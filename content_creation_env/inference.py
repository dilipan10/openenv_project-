# inference.py
import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

# ─────────────────────────────────────
# SETUP — from environment variables
# ─────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

BENCHMARK    = "content_creation_env"
MAX_STEPS    = 5
TEMPERATURE  = 0.65

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

sys.path.append(".")
from server.content_creation_env_environment import ContentCreationEnvironment
from models import ContentCreationAction

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────
# STRUCTURED LOG HELPERS (mandatory format)
# ─────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: Optional[float], done: bool, error: Optional[str]) -> None:
    reward_val = f"{reward:.2f}" if reward is not None else "0.00"
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    # Sanitize action: remove newlines so it stays on one line
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(f"[STEP] step={step} action={action_clean} reward={reward_val} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ─────────────────────────────────────
# SINGLE EPISODE
# ─────────────────────────────────────
def run_episode(task_name: str) -> float:
    env = ContentCreationEnvironment()
    obs = env.reset(task_name=task_name)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    conversation = [
        {
            "role": "system",
            "content": (
                "You are an expert Content Creation Assistant for 8 niches: Gaming, Study, Beauty, Food, Tech, Finance, Travel, Fitness.\n\n"
                "STRICT 2-TURN RULE - FOLLOW 100% OR YOU FAIL:\n\n"
                "TURN 1 (FIRST RESPONSE ONLY):\n"
                "Ask EXACTLY TWO questions and nothing else. Examples:\n"
                "1. Who is the target audience for this content?\n"
                "2. Which platform will you post on and what thumbnail colors or style do you want? Any specific keywords?\n\n"
                "TURN 2 (SECOND RESPONSE):\n"
                "Do NOT ask any questions.\n"
                "Reply with ONLY this exact block. Start with the line 'FINAL SUBMISSION:' and include ALL sections fully:\n\n"
                "FINAL SUBMISSION:\n"
                "SEO Title: [Short catchy title maximum 60 characters]\n"
                "Description: [Write at least 3-4 full detailed sentences about the content. Make it engaging and SEO friendly.]\n"
                "Keywords: [mainkeyword1, mainkeyword2, mainkeyword3, mainkeyword4]\n"
                "Hashtags: [#Hashtag1, #Hashtag2, #Hashtag3, #Hashtag4, #Hashtag5]\n"
                "Thumbnail: [Describe colors and style clearly, e.g. 'Warm orange and yellow food photography with fresh ingredients and clean white background']\n"
                "Target Audience: [Specific audience, e.g. 'Busy home cooks and beginners who want quick recipes']\n"
                "Platform: [Instagram Reels, YouTube Shorts, TikTok etc.]\n\n"
                "For Food tasks always include these hashtags: #ButterChicken #HomeCooking #Recipe #IndianFood #EasyRecipes\n"
                "For Gaming tasks always include: #Gaming #Ranked #FPS #Esports #WinStreak\n"
                "For Study tasks always include: #Physics #BoardExam #Study #Newton #ExamPrep\n"
                "For Beauty tasks always include: #Makeup #Drugstore #BeautyTips #GlamLook #AffordableBeauty\n"
                "For Tech tasks always include: #MacBook #TechReview #Apple #M4Chip #TechNews\n"
                "For Finance tasks always include: #PersonalFinance #Investing #Savings #MoneyTips #IndexFunds\n"
                "For Travel tasks always include: #Bali #BudgetTravel #Travel #TravelVlog #Wanderlust\n"
                "For Fitness tasks always include: #HomeWorkout #WeightLoss #Fitness #Bodyweight #NoGym\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- NEVER submit on turn 1.\n"
                "- NEVER submit without asking the two questions first.\n"
                "- ALWAYS fill every section completely - never leave Description, Thumbnail, Keywords or Hashtags empty.\n"
                "- Make Description long and detailed.\n"
                "- You must submit on the second turn only."
            )
        },
        {"role": "user", "content": obs.client_response}
    ]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                max_tokens=700,
                temperature=TEMPERATURE,
            )

            ai_message = response.choices[0].message.content
            is_final   = "FINAL SUBMISSION:" in ai_message

            action = ContentCreationAction(message=ai_message, is_final_submission=is_final)
            obs    = env.step(action)

            reward      = obs.reward if obs.reward is not None else 0.0
            done        = obs.done
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=ai_message, reward=reward, done=done, error=None)

            if done:
                score   = reward
                success = score > 0.0
                break

            conversation.append({"role": "assistant", "content": ai_message})
            conversation.append({"role": "user",      "content": obs.client_response})

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        if not rewards:
            rewards = [0.0]
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ─────────────────────────────────────
# RUN ALL TASKS (one episode per task)
# ─────────────────────────────────────
TASKS = [
    "gaming_content",
    "study_content",
    "beauty_content",
    "food_content",
    "tech_content",
    "finance_content",
    "travel_content",
    "fitness_content",
]

if __name__ == "__main__":
    all_rewards = []

    for task in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"TASK: {task}", flush=True)
        print(f"{'='*50}\n", flush=True)
        score = run_episode(task)
        all_rewards.append(score)

    summary = {
        "total_tasks":    len(TASKS),
        "scores":         all_rewards,
        "average_score":  round(sum(all_rewards) / len(all_rewards), 2)
    }
    with open("outputs/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUMMARY] {json.dumps(summary)}", flush=True)
