import logging
from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    after_agent,
    before_agent,
    before_model,
)
from langchain.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.prompts import (
    BASE_SYSTEM_PROMPT,
    IDENTIFY_CUSTOMER_PROMPT,
    TRANSACTION_LOOKUP_PROMPT,
    MUSIC_RECOMMENDATION_PROMPT,
    GENERAL_HELP_PROMPT,
)
from src.state import SupportState

logger = logging.getLogger(__name__)

# ── Tool sets for each workflow ────────────────────────────────────────────

IDENTITY_TOOLS = {"identify_customer", "classify_intent"}

TRANSACTION_TOOLS = IDENTITY_TOOLS | {
    "get_my_recent_invoices",
    "get_invoice_detail",
    "get_my_spending_summary",
}

RECOMMENDATION_TOOLS = IDENTITY_TOOLS | {
    "get_my_top_genres",
    "get_my_top_artists",
    "recommend_by_genre",
    "recommend_by_artist",
    "get_popular_in_store",
}

# ── Off-topic keyword list ─────────────────────────────────────────────────

OFF_TOPIC_KEYWORDS = [
    "bomb", "hack", "password", "exploit", "weapon",
    "politics", "election", "religion", "stock tip", "crypto",
    "write code", "generate image", "translate", "essay",
    "homework", "recipe", "weather", "news", "sports score",
]


# ── Shared helper: resolve state-machine config ────────────────────────────

def _resolve_state_config(state: dict[str, Any]):
    """Return (formatted_system_prompt, allowed_tool_names) based on state."""
    customer_id = state.get("authenticated_customer_id")
    customer_first = state.get("customer_first_name", "")
    customer_last = state.get("customer_last_name", "")
    intent = state.get("intent")

    fmt = {
        "authenticated_customer_id": customer_id or "unknown",
        "customer_first_name": customer_first,
        "customer_last_name": customer_last,
        "customer_email": state.get("customer_email", ""),
    }

    if not customer_id:
        system_prompt = IDENTIFY_CUSTOMER_PROMPT.format(**fmt)
        allowed_tool_names = IDENTITY_TOOLS
    elif intent == "transaction_lookup":
        system_prompt = TRANSACTION_LOOKUP_PROMPT.format(**fmt)
        allowed_tool_names = TRANSACTION_TOOLS
    elif intent == "music_recommendation":
        system_prompt = MUSIC_RECOMMENDATION_PROMPT.format(**fmt)
        allowed_tool_names = RECOMMENDATION_TOOLS
    else:
        system_prompt = GENERAL_HELP_PROMPT.format(**fmt)
        allowed_tool_names = IDENTITY_TOOLS

    return system_prompt, allowed_tool_names


# ── Hook 1: @before_agent — inject identity from context ───────────────────

@before_agent
def bootstrap_identity(
    state: SupportState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """Resolve caller identity from runtime context into agent state.

    Production: the calling app passes ``context=SupportContext(customer_id=...)``
    and the agent skips the email-lookup roundtrip entirely.

    Demo/Studio: context is empty → falls through to email lookup.
    """
    ctx = runtime.context
    if not ctx or not getattr(ctx, "customer_id", None):
        return None
    return {
        "authenticated_customer_id": ctx.customer_id,
        "customer_first_name": ctx.first_name or "",
        "customer_last_name": ctx.last_name or "",
        "customer_email": ctx.email or "",
    }


# ── Hook 2: @before_model — off-topic guard ────────────────────────────────

@before_model(can_jump_to=["end"])
def reject_off_topic(
    state: SupportState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """Hard-reject messages outside music-store support scope.

    Runs before every model call.  If the last user message contains
    clearly-off-topic keywords the turn ends without an LLM call —
    saving cost and preventing the model from being led off-topic.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    raw = last_msg.content if hasattr(last_msg, "content") else ""
    if isinstance(raw, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        ).lower()
    else:
        content = str(raw).lower()

    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in content:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I'm Chinook Music Store's support agent — I can "
                            "help with order history, invoice lookup, and "
                            "music recommendations. I can't help with that "
                            "request."
                        )
                    )
                ],
                "jump_to": "end",
            }
    return None


# ── Hook 2: IdentityGuardMiddleware — state-machine routing ────────────────

class IdentityGuardMiddleware(AgentMiddleware):
    """State-machine middleware: selects prompt + tools based on state.

    Reads ``authenticated_customer_id`` and ``intent`` from state (set
    deterministically by tools returning ``Command``) and applies the
    matching configuration.

    Implements both sync and async paths so it works with LangGraph API
    (async) and direct Python invocation (sync).
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return self._apply_guard(request, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return await self._apply_guard(request, handler)

    @staticmethod
    def _apply_guard(request: ModelRequest, handler: Callable) -> ModelResponse:
        system_prompt, allowed_tool_names = _resolve_state_config(request.state)
        allowed_tools = [t for t in request.tools if t.name in allowed_tool_names]
        full_prompt = BASE_SYSTEM_PROMPT + "\n\n" + system_prompt
        return handler(
            request.override(system_prompt=full_prompt, tools=allowed_tools)
        )


# ── Hook 3: AuditToolMiddleware — audit trail ─────────────────────────────

class AuditToolMiddleware(AgentMiddleware):
    """Log every tool call with customer context, with graceful error handling.

    Implements both sync and async paths for LangGraph API compatibility.
    Tool failures are caught and returned as informative ToolMessages rather
    than crashing the agent loop.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable,
    ) -> ToolMessage | Command:
        return self._apply_audit(request, handler)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable,
    ) -> ToolMessage | Command:
        return await self._apply_audit(request, handler)

    @staticmethod
    def _apply_audit(request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})
        customer_id = request.state.get("authenticated_customer_id", "unknown")
        logger.info(
            "AUDIT: customer=%s tool=%s args=%s",
            customer_id, tool_name, tool_args,
        )
        try:
            return handler(request)
        except Exception as exc:
            logger.error(
                "AUDIT: customer=%s tool=%s FAILED: %s",
                customer_id, tool_name, exc,
            )
            return ToolMessage(
                content=f"Tool '{tool_name}' failed: {exc}. Inform the user and suggest retrying.",
                tool_call_id=request.tool_call["id"],
            )


# ── Hook 4: @after_agent — show answer text in chat bubble ───────────────

@after_agent
async def show_answer_in_bubble(
    state: SupportState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """Append the structured response answer as a chat-bubble AIMessage.

    ToolStrategy stores output in ``structured_response`` and appends a
    ToolMessage — neither renders as a chat bubble in Studio.  This hook
    fires after the agent loop completes (when ``structured_response`` is
    guaranteed to be set) and appends a plain AIMessage with just the answer.
    """
    sr = state.get("structured_response")
    if sr is None:
        return None
    answer = sr.answer if hasattr(sr, "answer") else sr.get("answer")
    if not answer:
        return None
    return {"messages": [AIMessage(content=answer)]}


# ── Instantiate class-based middleware ─────────────────────────────────────

identity_guard = IdentityGuardMiddleware()
audit_tool_calls = AuditToolMiddleware()
