"""
Database layer for g2api - SQLite backend
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiosqlite

_db: Optional[aiosqlite.Connection] = None
_initialized = False

async def init_db() -> aiosqlite.Connection:
    global _db, _initialized
    if _initialized and _db:
        return _db

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "data.sqlite3"
    _db = await aiosqlite.connect(db_path)

    await _db.execute("PRAGMA journal_mode=WAL;")
    await _db.execute("PRAGMA synchronous = NORMAL;")

    await _db.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            label TEXT,
            api_key TEXT,
            gummie_id TEXT,
            other TEXT,
            created_at TEXT,
            updated_at TEXT,
            enabled INTEGER DEFAULT 1,
            error_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0
        )
    """)
    await _db.execute("CREATE INDEX IF NOT EXISTS idx_accounts_enabled ON accounts (enabled);")
    await _db.commit()
    _initialized = True
    return _db

async def close_db():
    global _db, _initialized
    if _db:
        await _db.close()
        _db = None
    _initialized = False

async def get_db() -> aiosqlite.Connection:
    if not _db:
        await init_db()
    return _db

def row_to_dict(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    if d.get("other"):
        try:
            d["other"] = json.loads(d["other"])
        except:
            pass
    if "enabled" in d and d["enabled"] is not None:
        d["enabled"] = bool(int(d["enabled"]))
    return d

async def get_accounts(enabled: Optional[bool] = None, sort_by: str = "created_at", sort_order: str = "desc") -> List[Dict]:
    db = await get_db()
    db.row_factory = aiosqlite.Row

    query = "SELECT * FROM accounts"
    params = []
    if enabled is not None:
        query += " WHERE enabled = ?"
        params.append(1 if enabled else 0)

    if sort_by not in ("created_at", "success_count"):
        sort_by = "created_at"
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"
    query += f" ORDER BY {sort_by} {sort_order}"

    async with db.execute(query, params) as cursor:
        rows = await cursor.fetchall()
        return [row_to_dict(dict(row)) for row in rows]

async def get_account(account_id: str) -> Optional[Dict]:
    db = await get_db()
    db.row_factory = aiosqlite.Row
    async with db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)) as cursor:
        row = await cursor.fetchone()
        return row_to_dict(dict(row)) if row else None

async def create_account(data: Dict) -> Dict:
    import uuid
    from datetime import datetime

    db = await get_db()
    account_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat() + "Z"

    other_json = json.dumps(data.get("other")) if data.get("other") else None

    await db.execute("""
        INSERT INTO accounts (id, label, api_key, gummie_id, other, created_at, updated_at, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        account_id,
        data.get("label"),
        data.get("api_key"),
        data.get("gummie_id"),
        other_json,
        now, now,
        1 if data.get("enabled", True) else 0
    ))
    await db.commit()
    return await get_account(account_id)

async def update_account(account_id: str, data: Dict) -> Optional[Dict]:
    from datetime import datetime

    db = await get_db()
    existing = await get_account(account_id)
    if not existing:
        return None

    fields = []
    params = []

    for key in ("label", "api_key", "gummie_id"):
        if key in data:
            fields.append(f"{key} = ?")
            params.append(data[key])

    if "enabled" in data:
        fields.append("enabled = ?")
        params.append(1 if data["enabled"] else 0)

    if "other" in data:
        fields.append("other = ?")
        params.append(json.dumps(data["other"]) if data["other"] else None)

    if "error_count" in data:
        fields.append("error_count = ?")
        params.append(data["error_count"])

    if "success_count" in data:
        fields.append("success_count = ?")
        params.append(data["success_count"])

    if not fields:
        return existing

    fields.append("updated_at = ?")
    params.append(datetime.utcnow().isoformat() + "Z")
    params.append(account_id)

    await db.execute(f"UPDATE accounts SET {', '.join(fields)} WHERE id = ?", params)
    await db.commit()
    return await get_account(account_id)

async def delete_account(account_id: str) -> bool:
    db = await get_db()
    cursor = await db.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
    await db.commit()
    return cursor.rowcount > 0

async def increment_success(account_id: str):
    db = await get_db()
    await db.execute("UPDATE accounts SET success_count = success_count + 1 WHERE id = ?", (account_id,))
    await db.commit()

async def increment_error(account_id: str):
    db = await get_db()
    await db.execute("UPDATE accounts SET error_count = error_count + 1 WHERE id = ?", (account_id,))
    await db.commit()
