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
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

BENCHMARK    = "content_creation_env"
MAX_STEPS    = 6

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
                "You are an expert Content Creation Assistant specialized in 8 niches: "
                "Gaming, Study, Beauty, Food, Tech, Finance, Travel, Fitness.\n\n"
                "You MUST follow these strict rules in every conversation:\n\n"
                "1. ALWAYS ask EXACTLY 2 clarifying questions in your FIRST response.\n"
                "   The two questions MUST cover:\n"
                "   - Target audience\n"
                "   - Platform (Instagram, YouTube, TikTok, etc.)\n"
                "   - Thumbnail colors or style\n"
                "   - Desired keywords or topic details\n\n"
                "2. After the user answers, create high-quality content based on their input.\n\n"
                "3. When you are ready to submit (on your second or third message), you MUST "
                "start your message with exactly this line:\n"
                "FINAL SUBMISSION:\n\n"
                "4. In the FINAL SUBMISSION, you MUST include ALL these sections in this exact format:\n\n"
                "FINAL SUBMISSION:\n"
                "SEO Title: [Short catchy title - maximum 60 characters]\n"
                "Description: [Detailed, engaging description with SEO]\n"
                "Keywords: [keyword1, keyword2, keyword3]\n"
                "Hashtags: [#Hashtag1, #Hashtag2, #Hashtag3, ...]\n"
                "Thumbnail: [Suggested thumbnail colors and style]\n"
                "Target Audience: [Who this content is for]\n"
                "Platform: [Chosen platform]\n\n"
                "Special rule for Beauty content:\n"
                "- Always include these hashtags: #Makeup, #Drugstore, #BeautyTips\n"
                "- Always suggest attractive thumbnail colors (e.g. pastel pink, rose gold, soft beige)\n"
                "- Focus on makeup, skincare, drugstore finds, or beauty tips\n\n"
                "You are not allowed to submit without asking the 2 clarifying questions first.\n"
                "You must submit the FINAL SUBMISSION by the 3rd or 4th turn at latest.\n"
                "Be professional, creative, and helpful."
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
                max_tokens=500,
            )

            ai_message = response.choices[0].message.content
            is_final   = ai_message.strip().startswith("FINAL SUBMISSION:")

            action = ContentCreationAction(message=ai_message, is_final_submission=is_final)
            obs    = env.step(action)

            reward     = obs.reward if obs.reward is not None else 0.0
            done       = obs.done
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=ai_message, reward=reward, done=done, error=None)

            if done:
                score   = reward
                success = score > 0.0
                break

            conversation.append({"role": "assistant", "content": ai_message})
            conversation.append({"role": "user",      "content": obs.client_response})

        if not rewards:
            rewards = [0.0]

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

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
        "average_score":  round(sum(all_rewards) / len(all_rewards), 3)
    }
    with open("outputs/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUMMARY] {json.dumps(summary)}", flush=True)
