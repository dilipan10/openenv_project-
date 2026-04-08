# Content Creation RL Environment

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ContentCreationAction, ContentCreationObservation
except ImportError:
    from models import ContentCreationAction, ContentCreationObservation


class ContentCreationEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    PERSONAS = [
        {
            "name": "gaming_content",
            "raw_transcript": "yo so today we played ranked and it was insane bro like we won 5 games straight and my aim was cracked the whole time",
            "audience": "teenagers and young adults aged 13-25",
            "color": "neon green and black",
            "platform": "YouTube",
            "mandatory_hashtags": ["#Gaming", "#Ranked", "#FPS"],
            "keywords": ["gaming", "ranked", "winning streak"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "study_content",
            "raw_transcript": "in this video we cover the entire chapter 5 of physics including newtons laws motion and some solved examples for board exams",
            "audience": "students aged 15-20 preparing for board exams",
            "color": "clean white and blue minimal design",
            "platform": "YouTube",
            "mandatory_hashtags": ["#Physics", "#BoardExam", "#Study"],
            "keywords": ["physics", "newton", "board exam", "study"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "beauty_content",
            "raw_transcript": "hey guys today im doing a full glam makeup tutorial using only drugstore products and it turned out so good omg",
            "audience": "women aged 18-35 interested in affordable beauty",
            "color": "pink and gold vibrant",
            "platform": "Instagram Reels and YouTube",
            "mandatory_hashtags": ["#Makeup", "#Drugstore", "#BeautyTips"],
            "keywords": ["makeup", "drugstore", "affordable", "glam"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "food_content",
            "raw_transcript": "today i made the easiest butter chicken recipe at home with simple ingredients and it tasted just like restaurant style",
            "audience": "home cooks aged 25-45 who love Indian food",
            "color": "warm orange and brown appetizing",
            "platform": "YouTube",
            "mandatory_hashtags": ["#ButterChicken", "#HomeCooking", "#Recipe"],
            "keywords": ["butter chicken", "recipe", "homemade", "restaurant style"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "tech_content",
            "raw_transcript": "today i reviewed the new macbook pro m4 chip and honestly the performance benchmarks blew me away especially for video editing and machine learning tasks",
            "audience": "tech enthusiasts and professionals aged 20-40",
            "color": "sleek silver and dark grey minimal",
            "platform": "YouTube",
            "mandatory_hashtags": ["#MacBook", "#TechReview", "#Apple"],
            "keywords": ["macbook", "m4", "review", "performance"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "finance_content",
            "raw_transcript": "in this video i explain how i saved my first 10 lakhs in 2 years on a 30k salary by following strict budgeting and investing in index funds",
            "audience": "young working professionals aged 22-35 interested in personal finance",
            "color": "clean green and white professional",
            "platform": "YouTube",
            "mandatory_hashtags": ["#PersonalFinance", "#Investing", "#Savings"],
            "keywords": ["savings", "investing", "budget", "index funds"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "travel_content",
            "raw_transcript": "we spent 5 days in bali on a budget of 30000 rupees and visited all the major temples beaches and tried the local street food it was absolutely worth it",
            "audience": "budget travellers aged 20-35 who love exploring Asia",
            "color": "tropical blue and green vibrant",
            "platform": "YouTube and Instagram",
            "mandatory_hashtags": ["#Bali", "#BudgetTravel", "#Travel"],
            "keywords": ["bali", "budget travel", "temples", "street food"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
        {
            "name": "fitness_content",
            "raw_transcript": "today i show you my complete home workout routine that i used to lose 10 kg in 3 months with no gym equipment just bodyweight exercises",
            "audience": "fitness beginners aged 18-40 who want to work out at home",
            "color": "bold red and black energetic",
            "platform": "YouTube",
            "mandatory_hashtags": ["#HomeWorkout", "#WeightLoss", "#Fitness"],
            "keywords": ["home workout", "weight loss", "bodyweight", "no gym"],
            "title_max_chars": 60,
            "description_max_chars": 300,
        },
    ]

    # Map task name → persona for deterministic task runs
    PERSONA_MAP = {p["name"]: p for p in PERSONAS}

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_persona = None
        self._turn_number = 0
        self._asked_about_audience = False
        self._asked_about_color = False
        self._asked_about_platform = False
        self._asked_about_keywords = False
        self._task_name = None

    def reset(self, task_name: str = None) -> ContentCreationObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._turn_number = 0
        self._asked_about_audience = False
        self._asked_about_color = False
        self._asked_about_platform = False
        self._asked_about_keywords = False
        self._task_name = task_name

        # Use specific persona if task_name given, else random
        if task_name and task_name in self.PERSONA_MAP:
            self._current_persona = self.PERSONA_MAP[task_name]
        else:
            self._current_persona = random.choice(self.PERSONAS)

        initial_message = (
            f"Hi! I want to upload a video. Here's my raw transcript:\n\n"
            f'"{self._current_persona["raw_transcript"]}"\n\n'
            f"Give me the SEO metadata and a thumbnail visual brief for this video."
        )

        return ContentCreationObservation(
            client_response=initial_message,
            turn_number=0,
            reward=None,
            done=False,
            feedback="Episode started. Ask clarifying questions then submit final answer."
        )

    def step(self, action: ContentCreationAction) -> ContentCreationObservation:
        self._state.step_count += 1
        self._turn_number += 1
        p = self._current_persona

        if action.is_final_submission:
            reward, feedback = self._grade_submission(action.message)
            return ContentCreationObservation(
                client_response="Thank you for your submission! Here is your score.",
                turn_number=self._turn_number,
                reward=reward,
                done=True,
                feedback=feedback
            )

        if self._turn_number > 6:
            return ContentCreationObservation(
                client_response="Please submit your final answer now. You've asked enough questions.",
                turn_number=self._turn_number,
                reward=None,
                done=False,
                feedback="Warning: Too many questions. Submit your final answer."
            )

        message_lower = action.message.lower()
        response = self._generate_client_response(message_lower, p)

        return ContentCreationObservation(
            client_response=response,
            turn_number=self._turn_number,
            reward=None,
            done=False,
            feedback=""
        )

    def _generate_client_response(self, message: str, p: dict) -> str:
        if any(w in message for w in ["audience", "viewers", "who", "demographic", "age", "target"]):
            self._asked_about_audience = True
            return f"My target audience is {p['audience']}."

        if any(w in message for w in ["color", "colour", "theme", "style", "thumbnail", "visual", "design"]):
            self._asked_about_color = True
            return f"For the thumbnail, use {p['color']} colors."

        if any(w in message for w in ["platform", "where", "upload", "channel", "youtube", "instagram", "social"]):
            self._asked_about_platform = True
            return f"I'm uploading this on {p['platform']}."

        if any(w in message for w in ["keyword", "topic", "tag", "hashtag", "focus", "main", "seo"]):
            self._asked_about_keywords = True
            return f"The main keywords I want to focus on are: {', '.join(p['keywords'])}."

        return "I'm not sure about that. Just do your best based on the transcript!"

    def _grade_submission(self, submission: str) -> tuple:
        """
        Scoring breakdown:
        - Asked about audience       → +0.10
        - Asked about colors         → +0.10
        - Asked about platform       → +0.10
        - Asked about keywords       → +0.10
        - All mandatory hashtags     → +0.20
        - Thumbnail color mentioned  → +0.10
        - 2+ keywords in submission  → +0.15
        - Title present & ≤60 chars  → +0.10
        - Description present        → +0.05
        Total possible               → 1.00
        """
        p = self._current_persona
        reward = 0.0
        feedback_parts = []
        sub = submission.lower()

        # Clarifying questions asked
        if self._asked_about_audience:
            reward += 0.10
            feedback_parts.append("✅ Asked about target audience (+0.10)")
        else:
            feedback_parts.append("❌ Never asked about target audience (0.00)")

        if self._asked_about_color:
            reward += 0.10
            feedback_parts.append("✅ Asked about thumbnail colors (+0.10)")
        else:
            feedback_parts.append("❌ Never asked about thumbnail colors (0.00)")

        if self._asked_about_platform:
            reward += 0.10
            feedback_parts.append("✅ Asked about platform (+0.10)")
        else:
            feedback_parts.append("❌ Never asked about platform (0.00)")

        if self._asked_about_keywords:
            reward += 0.10
            feedback_parts.append("✅ Asked about keywords (+0.10)")
        else:
            feedback_parts.append("❌ Never asked about keywords (0.00)")

        # Mandatory hashtags
        hashtags_found = all(tag.lower() in sub for tag in p["mandatory_hashtags"])
        if hashtags_found:
            reward += 0.20
            feedback_parts.append("✅ All mandatory hashtags included (+0.20)")
        else:
            missing = [t for t in p["mandatory_hashtags"] if t.lower() not in sub]
            feedback_parts.append(f"❌ Missing hashtags: {missing} (0.00)")

        # Thumbnail color
        color_word = p["color"].split()[0].lower()
        if color_word in sub:
            reward += 0.10
            feedback_parts.append("✅ Thumbnail color mentioned (+0.10)")
        else:
            feedback_parts.append("❌ Thumbnail color missing (0.00)")

        # Keywords in submission
        kw_found = sum(1 for kw in p["keywords"] if kw.lower() in sub)
        if kw_found >= 2:
            reward += 0.15
            feedback_parts.append(f"✅ {kw_found} keywords found (+0.15)")
        else:
            feedback_parts.append(f"❌ Only {kw_found} keywords found, need 2+ (0.00)")

        # Title length check (look for a line ≤60 chars that isn't a hashtag line)
        lines = submission.split("\n")
        title_lines = [l.strip() for l in lines if 5 < len(l.strip()) <= 60 and not l.strip().startswith("#")]
        if title_lines:
            reward += 0.10
            feedback_parts.append(f"✅ SEO title found within 60 chars (+0.10)")
        else:
            feedback_parts.append("❌ No valid SEO title ≤60 chars found (0.00)")

        # Description present (any line > 60 chars)
        desc_lines = [l.strip() for l in lines if len(l.strip()) > 60]
        if desc_lines:
            reward += 0.05
            feedback_parts.append("✅ Description present (+0.05)")
        else:
            feedback_parts.append("❌ No description found (0.00)")

        reward = round(min(reward, 1.0), 2)
        feedback = "\n".join(feedback_parts)
        feedback += f"\n\n🎯 FINAL SCORE: {reward}/1.0"

        return reward, feedback

    @property
    def state(self) -> State:
        return self._state
