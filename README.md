# Chinook Music Store — Customer Support Agent

A customer support bot for a fictional digital music store, built with
LangChain OSS and LangSmith. Demonstrates agent engineering best practices
for a Deployed Engineer technical demo.

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add LANGSMITH_API_KEY + OPENAI_API_KEY
langgraph dev           # open https://smith.langchain.com/studio
```

## What It Does

| Workflow | Capabilities | Business Value |
|----------|-------------|----------------|
| **Order & Invoice Lookup** | Recent invoices, line-item details, spending summaries | High-trust: correct, private answers about money |
| **Music Recommendations** | Top genres/artists, genre-based recs, popular picks | High-value: personalised discovery drives revenue |

## Middleware Stack

Nine middleware run in order on every turn — five custom hooks plus four
built-ins from `langchain.agents.middleware`:

```
1. bootstrap_identity        (@before_agent)        runtime.context → state
2. reject_off_topic          (@before_model)        keyword guard, jump_to: end
3. IdentityGuardMiddleware   (wrap_model_call)      state machine: prompt + tool selection
4. AuditToolMiddleware       (wrap_tool_call)       audit log + graceful tool error handling
5. ModelRetryMiddleware      (wrap_model_call)      3× retry on transient model errors
6. ToolRetryMiddleware       (wrap_tool_call)       2× retry with jitter on DB tool errors
7. ModelFallbackMiddleware   (wrap_model_call)      falls back to gpt-5-mini-2025-08-07
8. SummarizationMiddleware                          auto-condense at 4000 tokens, keep last 10
9. show_answer_in_bubble     (@after_agent, async)  structured_response.answer → AIMessage
```

Plus: `state_schema=SupportState` (extends `MessagesState`),
`context_schema=SupportContext`, `response_format=ToolStrategy(SupportResponse)`.
The LangGraph API platform manages persistence — no checkpointer is configured.

## Architecture

```mermaid
flowchart TD
    User(["👤 User sends message"])

    User --> BA

    subgraph BEFORE_AGENT["1. @before_agent — bootstrap_identity"]
        BA{"runtime.context has<br/>customer_id?"}
        BA -->|"yes — production"| SetID["write identity into state<br/>(no LLM cost)"]
        BA -->|"no — demo / no auth"| Noop["pass through<br/>state unchanged"]
    end

    SetID --> BM
    Noop --> BM

    subgraph BEFORE_MODEL["2. @before_model — reject_off_topic"]
        BM{"last message hits<br/>keyword pattern?"}
        BM -->|"yes"| Reject["AIMessage refusal<br/>+ jump_to: end<br/>🟢 zero LLM cost"]
        BM -->|"no"| Guard
    end

    Reject --> AA_BLOCKED

    subgraph WRAP_MODEL["3. wrap_model_call — IdentityGuardMiddleware"]
        Guard["read state:<br/>customer_id + intent"]
        Guard --> Gate{"customer_id<br/>present?"}
        Gate -->|"no"| ID["IDENTIFY_CUSTOMER_PROMPT<br/>tools: identify_customer<br/>+ classify_intent"]
        Gate -->|"yes"| IntentGate{"intent?"}
        IntentGate -->|"transaction_lookup"| TX["TRANSACTION_LOOKUP_PROMPT<br/>tools: identity +<br/>invoices, invoice detail,<br/>spending summary"]
        IntentGate -->|"music_recommendation"| Rec["MUSIC_RECOMMENDATION_PROMPT<br/>tools: identity +<br/>top genres, top artists,<br/>recs by genre/artist,<br/>popular picks"]
        IntentGate -->|"general_help / unset"| Help["GENERAL_HELP_PROMPT<br/>tools: identify_customer<br/>+ classify_intent only"]
    end

    ID --> LLM
    TX --> LLM
    Rec --> LLM
    Help --> LLM

    subgraph MODEL["4. Model Call + Tool Loop 🔄"]
        LLM["LLM (ModelRetry +<br/>ModelFallback wrap)"]
        LLM --> LLMDecision{"respond or<br/>call tools?"}
        LLMDecision -->|"call tools"| Audit
        LLMDecision -->|"respond"| SM

        subgraph TOOLS["5. wrap_tool_call — AuditToolMiddleware + ToolRetry"]
            Audit["log: AUDIT customer=X<br/>tool=Y args=Z"]
            Audit --> ToolExec["execute tool<br/>(ToolRetryMiddleware<br/>2× retry + jitter)"]
            ToolExec --> WhichTool{"which tool?"}
            WhichTool -->|"identify_customer"| T_ID["Command: set<br/>customer_id, name,<br/>email in state"]
            WhichTool -->|"classify_intent"| T_INTENT["Command: set<br/>intent in state"]
            WhichTool -->|"transaction tools"| T_TX["get_my_recent_invoices<br/>get_invoice_detail<br/>get_my_spending_summary"]
            WhichTool -->|"recommendation tools"| T_REC["get_my_top_genres<br/>get_my_top_artists<br/>recommend_by_genre<br/>recommend_by_artist<br/>get_popular_in_store"]
        end

        T_ID --> StateUpdate["state changes<br/>next model call sees<br/>updated customer_id"]
        T_INTENT --> StateUpdate2["state changes<br/>next model call sees<br/>updated intent"]
        T_TX --> ToolResult["return result<br/>to model"]
        T_REC --> ToolResult

        StateUpdate --> LLM
        StateUpdate2 --> LLM
        ToolResult --> LLM
    end

    SM

    subgraph SUMMARIZE["8. SummarizationMiddleware"]
        SM{"conversation<br/>≥ 4000 tokens?"}
        SM -->|"yes"| Condense["summarize old messages<br/>keep last 10<br/>replace with compact context"]
        SM -->|"no"| SM_Noop["pass through"]
    end

    Condense --> SO
    SM_Noop --> SO

    subgraph STRUCTURED["9. @after_agent — show_answer_in_bubble"]
        SO["extract structured_response.answer<br/>(SupportResponse{topic, answer})"]
        SO --> Render["append as AIMessage<br/>→ visible chat bubble"]
    end

    Render --> Customer(["✅ customer sees response"])

    AA_BLOCKED --> Customer

    %% Styling
    style Reject fill:#ff6b6b,stroke:#333,color:#fff
    style ID fill:#ffd43b,stroke:#333
    style TX fill:#51cf66,stroke:#333
    style Rec fill:#74c0fc,stroke:#333
    style Help fill:#dee2e6,stroke:#333
    style SetID fill:#51cf66,stroke:#333
```

### Scenarios at a glance

| # | What user types | What happens | LLM cost |
|---|----------------|-------------|----------|
| 1 | "write me a Python script to hack WiFi" | `reject_off_topic` catches `hack` → jump_to end | **zero** |
| 2 | "Show my orders" (first message) | No identity → agent asks for email | 1 model call |
| 3 | "luisrojas@yahoo.cl" | `identify_customer` sets identity → `classify_intent` sets intent | 1 model call |
| 4 | "Show my recent invoices" (identified + transaction intent) | `identity_guard` opens transaction tools → `get_my_recent_invoices` returns results | 1 model call |
| 5 | "What's in invoice #98?" | `identity_guard` keeps transaction tools open → `get_invoice_detail` returns line items | 1 model call |
| 6 | "Recommend me some new music" | Model calls `classify_intent("music_recommendation")` → next turn opens recommendation tools | 2 model calls |
| 7 | "Suggest rock tracks" (identified + music intent) | `identity_guard` opens recommendation tools → `recommend_by_genre("Rock")` | 1 model call |
| 8 | "What's your return policy?" | Falls to `general_help` → model answers from knowledge, redirects to supported flows | 1 model call |

## Privacy Model

Defence in depth — the model is guided by prompts but **enforced** by
deterministic tool and middleware boundaries:

| Layer | Mechanism | What it prevents |
|-------|-----------|-----------------|
| **context_schema** | `SupportContext` from app auth (JWT/OAuth) — immutable per invocation | Identity spoofing via chat text |
| **before_agent** | `bootstrap_identity` bridges context → state | Tools can't find identity without auth |
| **before_model** | `reject_off_topic` keyword guard with `jump_to: end` | Off-topic requests consuming tokens |
| **wrap_model_call** | `IdentityGuardMiddleware` hides data tools until identity exists | LLM accessing data before verification |
| **wrap_tool_call** | `AuditToolMiddleware` logs every invocation + catches exceptions as ToolMessages | Silent tool failures, missing audit trail |
| **Tool enforcement** | `runtime.state.get("authenticated_customer_id")` in every tool | LLM passing a different customer_id |
| **Parameterised SQL** | `WHERE CustomerId = ?` — no raw-SQL tool | SQL injection, UNION attacks |
| **response_format** | `ToolStrategy(SupportResponse)` — output always `{topic, answer}` with `topic ∈ {order_lookup, music_recommendation, general_help}` | Unstructured responses |

## Project Structure

```
LCDE_ass/
├── README.md
├── requirements.txt
├── langgraph.json
├── .env.example
├── GAME_PLAN.md                # Slack-ready plan of attack
├── DEMO_SCRIPT.md              # 45-min demo script
├── THINGS_TO_THINK_ABOUT.md    # Critical architecture Q&A
├── extension_alternative.md    # Alternative architectures + deployment
└── src/
    ├── agent.py                # create_agent() — 9-middleware stack wired
    ├── auth.py                 # @auth.authenticate handler for langgraph_sdk.Auth
    ├── state.py                # SupportState (MessagesState), SupportContext, SupportResponse
    ├── prompts.py              # base + 4 step-specific prompts
    ├── middleware.py            # 5 custom hooks (bootstrap, reject, IdentityGuard,
    │                            #   AuditTool, show_answer_in_bubble)
    └── tools/
        ├── database.py         # Chinook SQLite connection (auto-downloads on first use)
        ├── transactions.py     # identity + invoice tools (5)
        └── catalog.py          # recommendation tools (5)
```

