from langchain.tools import tool, ToolRuntime

from src.state import SupportState
from src.tools.database import get_raw_connection


@tool
def get_my_top_genres(
    limit: int = 5,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get the authenticated customer's most-purchased music genres based on their invoice history.

    Args:
        limit: Number of top genres to return (default 5)
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    rows = conn.execute(
        """SELECT g.Name AS Genre, COUNT(*) AS PurchaseCount
           FROM Invoice i
           JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
           JOIN Track t ON il.TrackId = t.TrackId
           JOIN Genre g ON t.GenreId = g.GenreId
           WHERE i.CustomerId = ?
           GROUP BY g.Name
           ORDER BY PurchaseCount DESC
           LIMIT ?""",
        (customer_id, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return "No purchase history found for your account yet."

    lines = ["Your top genres based on purchase history:"]
    for r in rows:
        lines.append(f"  {r['Genre']} — {r['PurchaseCount']} tracks purchased")
    return "\n".join(lines)


@tool
def get_my_top_artists(
    limit: int = 5,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get the authenticated customer's most-purchased artists based on their invoice history.

    Args:
        limit: Number of top artists to return (default 5)
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    rows = conn.execute(
        """SELECT ar.Name AS Artist, COUNT(*) AS PurchaseCount
           FROM Invoice i
           JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
           JOIN Track t ON il.TrackId = t.TrackId
           JOIN Album al ON t.AlbumId = al.AlbumId
           JOIN Artist ar ON al.ArtistId = ar.ArtistId
           WHERE i.CustomerId = ?
           GROUP BY ar.Name
           ORDER BY PurchaseCount DESC
           LIMIT ?""",
        (customer_id, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return "No purchase history found for your account yet."

    lines = ["Your top artists based on purchase history:"]
    for r in rows:
        lines.append(f"  {r['Artist']} — {r['PurchaseCount']} tracks purchased")
    return "\n".join(lines)


@tool
def recommend_by_genre(
    genre: str,
    limit: int = 5,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get top tracks in a specific genre that the authenticated customer hasn't purchased yet.

    Args:
        genre: Genre name (e.g., 'Rock', 'Jazz', 'Metal')
        limit: Number of recommendations (default 5)
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    rows = conn.execute(
        """SELECT t.Name AS Track, ar.Name AS Artist, al.Title AS Album, g.Name AS Genre
           FROM Track t
           JOIN Genre g ON t.GenreId = g.GenreId
           LEFT JOIN Album al ON t.AlbumId = al.AlbumId
           LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
           WHERE g.Name = ?
             AND t.TrackId NOT IN (
                 SELECT il.TrackId FROM InvoiceLine il
                 JOIN Invoice i ON il.InvoiceId = i.InvoiceId
                 WHERE i.CustomerId = ?
             )
           ORDER BY RANDOM()
           LIMIT ?""",
        (genre, customer_id, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return f"No new tracks found in '{genre}' that you haven't already purchased."

    lines = [f"Recommended {genre} tracks you might like:"]
    for r in rows:
        artist = r["Artist"] or "Unknown Artist"
        album = r["Album"] or "Unknown Album"
        lines.append(f"  {r['Track']} by {artist} (from {album})")
    return "\n".join(lines)


@tool
def recommend_by_artist(
    artist_name: str,
    limit: int = 5,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get tracks by a specific artist that the authenticated customer hasn't purchased yet.

    Args:
        artist_name: Artist name to get recommendations from
        limit: Number of recommendations (default 5)
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    rows = conn.execute(
        """SELECT t.Name AS Track, ar.Name AS Artist, al.Title AS Album
           FROM Track t
           JOIN Album al ON t.AlbumId = al.AlbumId
           JOIN Artist ar ON al.ArtistId = ar.ArtistId
           WHERE ar.Name = ?
             AND t.TrackId NOT IN (
                 SELECT il.TrackId FROM InvoiceLine il
                 JOIN Invoice i ON il.InvoiceId = i.InvoiceId
                 WHERE i.CustomerId = ?
             )
           ORDER BY RANDOM()
           LIMIT ?""",
        (artist_name, customer_id, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return f"No new tracks found by '{artist_name}' that you haven't already purchased."

    lines = [f"Tracks by {artist_name} you might like:"]
    for r in rows:
        album = r["Album"] or "Unknown Album"
        lines.append(f"  {r['Track']} (from {album})")
    return "\n".join(lines)


@tool
def get_popular_in_store(
    limit: int = 5,
    runtime: ToolRuntime[None, SupportState] = None,
) -> str:
    """Get the most popular tracks in the store that the authenticated customer hasn't purchased yet.

    Args:
        limit: Number of recommendations (default 5)
    """
    customer_id = runtime.state.get("authenticated_customer_id")
    if not customer_id:
        return "ERROR: No authenticated customer. Identify the customer first."

    conn = get_raw_connection()
    rows = conn.execute(
        """SELECT t.Name AS Track, ar.Name AS Artist, g.Name AS Genre,
                  COUNT(il.TrackId) AS SalesCount
           FROM Track t
           JOIN InvoiceLine il ON t.TrackId = il.TrackId
           LEFT JOIN Album al ON t.AlbumId = al.AlbumId
           LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
           LEFT JOIN Genre g ON t.GenreId = g.GenreId
           WHERE t.TrackId NOT IN (
               SELECT il2.TrackId FROM InvoiceLine il2
               JOIN Invoice i2 ON il2.InvoiceId = i2.InvoiceId
               WHERE i2.CustomerId = ?
           )
           GROUP BY t.TrackId
           ORDER BY SalesCount DESC
           LIMIT ?""",
        (customer_id, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return "No popular tracks available to recommend."

    lines = ["Popular tracks in the store you haven't purchased yet:"]
    for r in rows:
        artist = r["Artist"] or "Unknown Artist"
        genre = r["Genre"] or "Unknown"
        lines.append(f"  {r['Track']} by {artist} ({genre}) — {r['SalesCount']} sales")
    return "\n".join(lines)
