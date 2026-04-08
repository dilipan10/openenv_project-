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
        - Audience info present      → +0.10
        - Thumbnail style present    → +0.10
        - Platform info present      → +0.10
        - Keywords/hashtags present  → +0.10
        - Mandatory hashtags (case-insensitive, partial credit) → +0.20
        - Thumbnail color word       → +0.10
        - 2+ keywords in submission  → +0.15
        - Title present & ≤60 chars  → +0.10
        - Description present        → +0.05
        Total possible               → 1.00
        """
        p = self._current_persona
        reward = 0.0
        feedback_parts = []
        sub = submission.lower()

        # Audience — asked during conversation OR mentioned in submission
        audience_in_sub = any(w in sub for w in ["audience", "viewers", "demographic", "aged", "age", "target audience", "for"])
        if self._asked_about_audience or audience_in_sub:
            reward += 0.10
            feedback_parts.append("✅ Target audience addressed (+0.10)")
        else:
            feedback_parts.append("❌ Target audience not addressed (0.00)")

        # Thumbnail — asked OR mentioned in submission
        color_in_sub = any(w in sub for w in ["thumbnail", "color", "colour", "visual", "style", "design", "background"])
        if self._asked_about_color or color_in_sub:
            reward += 0.10
            feedback_parts.append("✅ Thumbnail style addressed (+0.10)")
        else:
            feedback_parts.append("❌ Thumbnail style not addressed (0.00)")

        # Platform — asked OR mentioned in submission
        platform_in_sub = any(w in sub for w in ["youtube", "instagram", "tiktok", "platform", "reels", "shorts", "channel"])
        if self._asked_about_platform or platform_in_sub:
            reward += 0.10
            feedback_parts.append("✅ Platform addressed (+0.10)")
        else:
            feedback_parts.append("❌ Platform not addressed (0.00)")

        # Keywords section present
        keywords_in_sub = any(w in sub for w in ["keyword", "keywords", "seo", "#"])
        if self._asked_about_keywords or keywords_in_sub:
            reward += 0.10
            feedback_parts.append("✅ Keywords addressed (+0.10)")
        else:
            feedback_parts.append("❌ Keywords not addressed (0.00)")

        # Mandatory hashtags — case-insensitive, partial credit
        # Strip # and compare lowercase
        sub_tags = set()
        for word in sub.split():
            clean = word.strip(".,!?()[]").lstrip("#")
            sub_tags.add(clean.lower())

        mandatory_lower = [t.lstrip("#").lower() for t in p["mandatory_hashtags"]]
        matched = sum(1 for t in mandatory_lower if t in sub_tags or t in sub)
        total   = len(mandatory_lower)

        if matched == total:
            reward += 0.20
            feedback_parts.append("✅ All mandatory hashtags included (+0.20)")
        elif matched >= 1:
            partial = round(0.20 * matched / total, 2)
            reward += partial
            missing = [p["mandatory_hashtags"][i] for i, t in enumerate(mandatory_lower) if t not in sub_tags and t not in sub]
            feedback_parts.append(f"⚠️ {matched}/{total} hashtags found, partial (+{partial}) — missing: {missing}")
        else:
            missing = p["mandatory_hashtags"]
            feedback_parts.append(f"❌ Missing hashtags: {missing} (0.00)")

        # Thumbnail color word from persona — check all words in color string
        color_words = p["color"].lower().split()
        color_hit = any(cw in sub for cw in color_words if len(cw) > 3)
        if color_hit:
            reward += 0.10
            feedback_parts.append("✅ Thumbnail color mentioned (+0.10)")
        else:
            feedback_parts.append("❌ Thumbnail color missing (0.00)")

        # Keywords in submission — partial credit, case-insensitive
        kw_found = sum(1 for kw in p["keywords"] if kw.lower() in sub)
        if kw_found >= 2:
            reward += 0.15
            feedback_parts.append(f"✅ {kw_found} keywords found (+0.15)")
        elif kw_found == 1:
            reward += 0.07
            feedback_parts.append(f"⚠️ Only 1 keyword found, partial (+0.07)")
        else:
            feedback_parts.append(f"❌ No keywords found (0.00)")

        # Title length check — look for SEO Title: line or any short line ≤60 chars
        lines = submission.split("\n")
        title_found = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("seo title:"):
                title_text = stripped[10:].strip()
                if 5 < len(title_text) <= 60:
                    title_found = True
                    break
            elif 5 < len(stripped) <= 60 and not stripped.startswith("#") and ":" not in stripped[:15]:
                title_found = True
                break
        if title_found:
            reward += 0.10
            feedback_parts.append("✅ SEO title found within 60 chars (+0.10)")
        else:
            feedback_parts.append("❌ No valid SEO title ≤60 chars found (0.00)")

        # Description present — look for Description: line or any long line
        desc_found = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("description:"):
                desc_text = stripped[12:].strip()
                if len(desc_text) > 30:
                    desc_found = True
                    break
            elif len(stripped) > 80:
                desc_found = True
                break
        if desc_found:
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
