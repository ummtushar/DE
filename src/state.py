from dataclasses import dataclass
from typing import Literal
from typing_extensions import NotRequired

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ── Structured response (enforced on the final model output) ───────────────

class SupportResponse(BaseModel):
    """Final response from the Chinook Music Store support agent.

    Every agent turn must produce this structure.  The ``topic`` field lets
    downstream UI code render the right card (invoice table, recommendation
    carousel, or plain help text).
    """

    topic: Literal["order_lookup", "music_recommendation", "general_help"] = Field(
        description="Which support area this response addresses"
    )
    answer: str = Field(
        description="The response to the customer — be conversational but concise"
    )


# ── Routing types ──────────────────────────────────────────────────────────

Intent = Literal["transaction_lookup", "music_recommendation", "general_help"]


# ── Agent state (persisted across turns) ───────────────────────────────────

class SupportState(MessagesState):
    """State for the music store customer support workflow.

    Extends ``MessagesState`` (required for LangSmith Studio chat mode)
    rather than ``AgentState``.  The ``jump_to`` and ``structured_response``
    fields are hoisted from AgentState so the agent factory continues to work.

    Tracks customer identity, current intent, and workflow progress
    to enable state-machine-based routing for deterministic behaviour.
    """

    intent: NotRequired[Intent]
    authenticated_customer_id: NotRequired[int]
    customer_first_name: NotRequired[str]
    customer_last_name: NotRequired[str]
    customer_email: NotRequired[str]
    # Hoisted from AgentState (needed by create_agent internals)
    jump_to: NotRequired[Literal["tools", "model", "end"]]
    structured_response: NotRequired[SupportResponse]


# ── Runtime context (immutable, set at invocation by the app) ──────────────

@dataclass
class SupportContext:
    """Immutable context injected at invocation time.

    In the demo this is optional (the agent asks for email).  In production
    the app authenticates the user and passes this directly — the email-lookup
    conversation step disappears entirely.
    """

    customer_id: int | None = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
