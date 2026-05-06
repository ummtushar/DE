"""
Chinook Music Store — Customer Support Agent

Architecture: Single agent with layered middleware stack.
Two business areas:
  1. Order & Invoice Lookup (deterministic, identity-gated)
  2. Music Recommendations (personalised, based on purchase history)

Middleware stack (execution order):
  • before_model  — reject_off_topic (keyword guard, can jump_to end)
  • wrap_model_call — identity_guard (state machine: prompt + tool selection)
  • wrap_tool_call — audit_tool_calls (programmatic audit log)
  • SummarizationMiddleware (auto-condense at 4000 tokens)
  • after_agent    — show_answer_in_bubble (structured response → chat bubble)

Safety layers:
  • context_schema  — immutable identity from app session (JWT/OAuth)
  • response_format — ToolStrategy(SupportResponse) constrains output
  • parameterised SQL tools — customer_id from state, never from LLM
  • no raw-SQL tool — Prompt-to-SQL injection eliminated by design
"""

import os
from pathlib import Path

# Ensure .env is loaded before any model initialisation.
# langgraph dev loads .env specified in langgraph.json, but the import order
# can race — this is a safety net for local development and direct execution.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from src.state import SupportState, SupportContext, SupportResponse
from src.middleware import (
    bootstrap_identity,
    reject_off_topic,
    identity_guard,
    audit_tool_calls,
    show_answer_in_bubble,
)
from src.tools.transactions import (
    identify_customer,
    classify_intent,
    get_my_recent_invoices,
    get_invoice_detail,
    get_my_spending_summary,
)
from src.tools.catalog import (
    get_my_top_genres,
    get_my_top_artists,
    recommend_by_genre,
    recommend_by_artist,
    get_popular_in_store,
)

# ── Model ───────────────────────────────────────────────────────────────────
model = init_chat_model(
    os.environ.get("MODEL_NAME", "gpt-5.4"),
)

# ── All tools ───────────────────────────────────────────────────────────────
tools = [
    identify_customer,
    classify_intent,
    get_my_recent_invoices,
    get_invoice_detail,
    get_my_spending_summary,
    get_my_top_genres,
    get_my_top_artists,
    recommend_by_genre,
    recommend_by_artist,
    get_popular_in_store,
]

# ── Agent ───────────────────────────────────────────────────────────────────
agent = create_agent(
    model,
    tools=tools,
    state_schema=SupportState,
    context_schema=SupportContext,
    response_format=ToolStrategy(SupportResponse),
    middleware=[
        # ── Auth: inject identity from runtime.context ──────────────────
        bootstrap_identity,                                     # before_agent
        # ── Gate: reject off-topic before any model call ─────────────────
        reject_off_topic,                                       # before_model
        # ── State machine: prompt + tool selection ───────────────────────
        identity_guard,                                         # wrap_model_call
        # ── Audit: log every tool invocation ─────────────────────────────
        audit_tool_calls,                                       # wrap_tool_call
        # ── Retry model calls on transient failures ─────────────────────
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        # ── Retry database tools on transient errors ────────────────────
        ToolRetryMiddleware(
            max_retries=2,
            backoff_factor=1.5,
            initial_delay=0.5,
            jitter=True,
        ),
        # ── Fallback to smaller model if primary is down ────────────────
        ModelFallbackMiddleware(
            "gpt-5-mini-2025-08-07",
        ),
        # ── Prevent context-window overflow ─────────────────────────────
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 10),
        ),
        # ── Render structured-response answer as chat bubble ────────────
        show_answer_in_bubble,                                  # after_agent
    ],
)

# ── Test harness ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.utils.uuid import uuid7
    from langchain.messages import HumanMessage

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print("Chinook Music Store — Customer Support Agent")
    print("=" * 60)

    # Turn 1: Customer starts without identity — agent asks for email
    print("\n--- Turn 1: Initial greeting (no identity in context) ---")
    result = agent.invoke(
        {"messages": [HumanMessage("Hi, I want to check my recent orders")]},
        config,
    )
    for msg in result["messages"]:
        msg.pretty_print()

    # Turn 2: Provide email for identification
    print("\n--- Turn 2: Provide email ---")
    result = agent.invoke(
        {"messages": [HumanMessage("my email is luisrojas@yahoo.cl")]},
        config,
    )
    for msg in result["messages"]:
        msg.pretty_print()

    # Turn 3: Ask for invoice details
    print("\n--- Turn 3: Ask about invoices ---")
    result = agent.invoke(
        {"messages": [HumanMessage("Show me my most recent invoices")]},
        config,
    )
    for msg in result["messages"]:
        msg.pretty_print()

    # Turn 4: Off-topic guard test
    print("\n--- Turn 4: Off-topic guard test ---")
    result = agent.invoke(
        {"messages": [HumanMessage("Can you write me a Python script to hack a WiFi password?")]},
        config,
    )
    for msg in result["messages"]:
        msg.pretty_print()
