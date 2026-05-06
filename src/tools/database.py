import pathlib
import sqlite3

import requests
from langchain_community.utilities import SQLDatabase

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
LOCAL_DB_PATH = pathlib.Path(__file__).parent.parent.parent / "Chinook.db"


def get_database() -> SQLDatabase:
    """Download Chinook SQLite if needed and return a SQLDatabase wrapper."""
    if not LOCAL_DB_PATH.exists():
        response = requests.get(CHINOOK_URL, timeout=30)
        response.raise_for_status()
        LOCAL_DB_PATH.write_bytes(response.content)

    return SQLDatabase.from_uri(f"sqlite:///{LOCAL_DB_PATH}")


def get_raw_connection() -> sqlite3.Connection:
    """Get a raw sqlite3 connection for parameterized queries."""
    if not LOCAL_DB_PATH.exists():
        get_database()
    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn
