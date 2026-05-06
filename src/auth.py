"""Custom authentication for Chinook Support Agent.

In production, the user is already authenticated by the app (JWT/OAuth/API key)
before reaching the agent.  This handler extracts their identity and makes it
available via ``config["configurable"]["langgraph_auth_user"]``.

The ``bootstrap_identity`` middleware then reads this into agent state so tools
can enforce per-customer data access without an email-lookup roundtrip.
"""

from langgraph_sdk import Auth


auth = Auth()


@auth.authenticate
async def authenticate(headers: dict) -> dict:
    """Resolve the caller's identity from request headers.

    Production: validate a JWT / OAuth token and return the user's profile.
    Demo: accept an optional ``x-customer-email`` header and look up the
    customer in Chinook; falls back to anonymous (email-lookup flow kicks in).
    """
    email = headers.get(b"x-customer-email")

    if email:
        email = email.decode() if isinstance(email, bytes) else email
    else:
        # Anonymous — agent will fall through to email-lookup flow
        return {"identity": "anonymous"}

    # Look up customer in Chinook
    from src.tools.database import get_raw_connection

    conn = get_raw_connection()
    row = conn.execute(
        "SELECT CustomerId, FirstName, LastName, Email FROM Customer WHERE Email = ?",
        (email,),
    ).fetchone()
    conn.close()

    if row is None:
        return {"identity": email}

    return {
        "identity": str(row["CustomerId"]),
        "customer_id": row["CustomerId"],
        "first_name": row["FirstName"],
        "last_name": row["LastName"],
        "email": row["Email"],
    }
