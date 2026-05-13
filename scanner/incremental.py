import os
import sqlite3
import time


class IncrementalTracker:
    """
    Tracks which files have been indexed and when.
    Uses file mtime to decide if a file needs re-indexing.

    Usage:
        tracker = IncrementalTracker("index.db")
        if tracker.needs_reindex("/path/to/file.py"):
            # ... index it ...
            tracker.mark_indexed("/path/to/file.py")
    """

    def __init__(self, db_path: str = "index.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                path       TEXT PRIMARY KEY,
                mtime      REAL NOT NULL,
                indexed_at REAL NOT NULL
            )
        """)
        self.conn.commit()

    def needs_reindex(self, path: str) -> bool:
        """Return True if file is new or has been modified since last index."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False

        row = self.conn.execute(
            "SELECT mtime FROM file_index WHERE path = ?", (path,)
        ).fetchone()

        if row is None:
            return True  # never indexed

        return mtime > row[0]  # file changed since last index

    def mark_indexed(self, path: str) -> None:
        """Record that a file has been indexed at the current time."""
        mtime = os.path.getmtime(path)
        self.conn.execute("""
            INSERT OR REPLACE INTO file_index (path, mtime, indexed_at)
            VALUES (?, ?, ?)
        """, (path, mtime, time.time()))
        self.conn.commit()

    def remove(self, path: str) -> None:
        """Remove a file from the index (e.g. it was deleted)."""
        self.conn.execute("DELETE FROM file_index WHERE path = ?", (path,))
        self.conn.commit()

    def indexed_count(self) -> int:
        """Total number of files tracked."""
        row = self.conn.execute("SELECT COUNT(*) FROM file_index").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self.conn.close()
