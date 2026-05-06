from typing import Literal

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

from src.state import SupportState
from src.tools.database import get_raw_connection


@tool
def identify_customer(
    email: str,
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Look up a customer by their email address and persist their identity to state.

    Use this to verify who you're talking to before accessing any account data.
    The customer ID is deterministically set by this tool — the model cannot choose it.

    Args:
        email: The customer's email address
    """
    conn = get_raw_connection()
    row = conn.execute(
        "SELECT CustomerId, FirstName, LastName, Email FROM Customer WHERE Email = ?",
        (email,),
    ).fetchone()
    conn.close()

    if not row:
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"No customer found with email {email}. Ask the customer to verify their email.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    return Command(update={
        "messages": [
            ToolMessage(
                content=(
                    f"Customer identified: {row['FirstName']} {row['LastName']} "
                    f"(ID: {row['CustomerId']}, Email: {row['Email']}). "
                    f"Now use classify_intent to route them."
                ),
                tool_call_id=runtime.tool_call_id,
            )
        ],
        "authenticated_customer_id": row["CustomerId"],
        "customer_first_name": row["FirstName"],
        "customer_last_name": row["LastName"],
        "customer_email": row["Email"],
    })


@tool
def classify_intent(
    intent: Literal["transaction_lookup", "music_recommendation", "general_help"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's intent to determine which workflow and tools to activate.

    Args:
        intent: The customer's goal —
            'transaction_lookup' for orders/invoices,
            'music_recommendation' for music suggestions,
            'general_help' for everything else
    """
    return Command(update={
        "messages": [
            ToolMessage(
                content=f"Intent set to: {intent}. You are now in the {intent} workflow.",
                tool_call_id=runtime.tool_call_id,
            )
        ],
        "intent": intent,
    })


@tool
def get_my_recent_invoices(
    limit: int = 5,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get the current authenticated customer's most recent invoices.

    Args:
        limit: Number of recent invoices to return (default 5)
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    rows = conn.execute(
        """SELECT InvoiceId, InvoiceDate, Total
           FROM Invoice
           WHERE CustomerId = ?
           ORDER BY InvoiceDate DESC
           LIMIT ?""",
        (customer_id, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return "No invoices found for your account."

    lines = [f"Your {len(rows)} most recent invoice(s):"]
    for r in rows:
        lines.append(f"  Invoice #{r['InvoiceId']} — {r['InvoiceDate']} — ${r['Total']:.2f}")
    return "\n".join(lines)


@tool
def get_invoice_detail(
    invoice_id: int,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get line items for a specific invoice. Only works for invoices belonging to the authenticated customer.

    Args:
        invoice_id: The invoice ID to look up
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()

    # Verify ownership — the invoice must belong to the authenticated customer
    invoice = conn.execute(
        "SELECT InvoiceId, InvoiceDate, Total FROM Invoice WHERE InvoiceId = ? AND CustomerId = ?",
        (invoice_id, customer_id),
    ).fetchone()

    if not invoice:
        conn.close()
        return f"Invoice #{invoice_id} not found for your account."

    # Get line items with track info
    items = conn.execute(
        """SELECT il.InvoiceLineId, t.Name AS TrackName, ar.Name AS ArtistName,
                  al.Title AS AlbumTitle, il.UnitPrice, il.Quantity
           FROM InvoiceLine il
           JOIN Track t ON il.TrackId = t.TrackId
           LEFT JOIN Album al ON t.AlbumId = al.AlbumId
           LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
           WHERE il.InvoiceId = ?
           ORDER BY il.InvoiceLineId""",
        (invoice_id,),
    ).fetchall()
    conn.close()

    lines = [
        f"Invoice #{invoice['InvoiceId']} — {invoice['InvoiceDate']} — Total: ${invoice['Total']:.2f}",
        f"{'─' * 60}",
    ]
    for item in items:
        artist = item["ArtistName"] or "Unknown Artist"
        lines.append(
            f"  {item['TrackName']} by {artist} "
            f"(${item['UnitPrice']:.2f} × {item['Quantity']})"
        )
    return "\n".join(lines)


@tool
def get_my_spending_summary(
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get a spending summary for the authenticated customer — total spent, order count, and first/last order dates."""
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    row = conn.execute(
        """SELECT COUNT(*) AS order_count,
                  COALESCE(SUM(Total), 0) AS total_spent,
                  MIN(InvoiceDate) AS first_order,
                  MAX(InvoiceDate) AS last_order
           FROM Invoice
           WHERE CustomerId = ?""",
        (customer_id,),
    ).fetchone()
    conn.close()

    return (
        f"Spending Summary:\n"
        f"  Total orders: {row['order_count']}\n"
        f"  Total spent: ${row['total_spent']:.2f}\n"
        f"  First order: {row['first_order']}\n"
        f"  Last order: {row['last_order']}"
    )
