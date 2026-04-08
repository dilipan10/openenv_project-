from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional

class ContentCreationAction(Action):
    message: str = Field(..., description="AI's question or final submission")
    is_final_submission: bool = Field(
        default=False,
        description="True if AI is submitting final answer, False if asking question"
    )

class ContentCreationObservation(Observation):
    client_response: str = Field(
        default="",
        description="Simulated client's response to AI's question"
    )
    turn_number: int = Field(
        default=1,
        description="Current turn in the conversation"
    )
    reward: Optional[float] = Field(
        default=None,
        description="Reward score between 0 and 1. Only given at end."
    )
    done: bool = Field(
        default=False,
        description="True when episode is complete"
    )
    feedback: str = Field(
        default="",
        description="Explanation of what AI did right or wrong"
    )