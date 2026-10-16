"""SQLite 数据层：逐题 upsert，不存任何绝对时间戳与 IP。

匿名口径来自设计定稿 §9：只记逐题作答时长。要加时间列必须先改设计并同步
test_db.py 的 schema 审查，不要先加列再放宽测试。
"""

import json
import sqlite3
import threading
from pathlib import Path

from server.anchors import DIM_NAMES, SCORE_LEVELS

# FastAPI 把同步路由丢进线程池执行，连接因此必然被跨线程使用；放行后要自己
# 把写操作串行化，否则两个评分者的 upsert 会互相踩。
_WRITE_LOCK = threading.Lock()


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(path)), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rater (
            code             TEXT PRIMARY KEY,
            expertise        TEXT,
            demographics_json TEXT,
            finished         INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS rating (
            code               TEXT NOT NULL,
            item_id            TEXT NOT NULL,
            dimension          TEXT NOT NULL,
            score              REAL NOT NULL,
            duration_sec       INTEGER NOT NULL,
            duration_truncated INTEGER NOT NULL DEFAULT 0,
            is_attention_check INTEGER NOT NULL DEFAULT 0,
            client_seq         INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (code, item_id)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def set_meta(conn, key, value) -> None:
    """记下这批数据是在哪份题池 / 哪批 code 上采的，分析侧据此断言。"""
    with _WRITE_LOCK, conn:
        conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (key, str(value)))


def get_meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def count_ratings(conn) -> int:
    """库里已有的评分条数。

    startup check 用它区分"空壳"与"真收过数据"：只有 `rating` 行会因换题池而作废，
    单有 `rater` 行（只填了自陈、还没打分）时改绑不损失任何科学数据。
    """
    return int(conn.execute("SELECT COUNT(*) FROM rating").fetchone()[0])


def ensure_meta(conn, key, value) -> None:
    """只在没记过的时候记：题池 sha 换了不能静默改写历史，必须人工处理。"""
    if get_meta(conn, key) is None:
        set_meta(conn, key, value)


def _ensure_rater(conn: sqlite3.Connection, code: str) -> None:
    with _WRITE_LOCK, conn:
        conn.execute("INSERT OR IGNORE INTO rater(code) VALUES (?)", (code,))


def upsert_rating(conn, code, item_id, dimension, score, duration_sec,
                   *, duration_truncated=0, is_attention_check=0, client_seq=None):
    """写入单题评分；同一 (code, item_id) 再写即覆盖，这是"提交前可改分"的实现。

    两道防线防的是真实发生过的静默错数据：
    1. **乱序写**：快速改选会并发多个 POST，后发先至时旧选择会覆盖新选择。
       `client_seq` 是前端每题递增的序号，只有更大序号的写入才生效。
    2. **脏耗时覆盖干净耗时**：刷新续答后该题起始时刻未知（truncated=1），
       这时的一次改分不能把首会话测到的真实耗时抹成 0。truncated 的写入
       只更新分值，不动时长。
    """
    if score not in SCORE_LEVELS:
        raise ValueError(f"score 必须是五档之一，收到 {score!r}")
    if dimension not in DIM_NAMES:
        raise ValueError(f"dimension 必须是 A/B/C/D，收到 {dimension!r}")

    _ensure_rater(conn, code)
    seq = int(client_seq) if client_seq is not None else 0
    keep_duration = ("CASE WHEN excluded.duration_truncated = 1 "
                     "THEN rating.duration_sec ELSE excluded.duration_sec END")
    keep_trunc = ("CASE WHEN excluded.duration_truncated = 1 "
                  "THEN rating.duration_truncated ELSE 0 END")
    with _WRITE_LOCK, conn:
        conn.execute(
            f"""
            INSERT INTO rating (code, item_id, dimension, score, duration_sec,
                                duration_truncated, is_attention_check, client_seq)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code, item_id) DO UPDATE SET
                score              = excluded.score,
                dimension          = excluded.dimension,
                duration_sec       = {keep_duration},
                duration_truncated = {keep_trunc},
                is_attention_check = excluded.is_attention_check,
                client_seq         = excluded.client_seq
            WHERE excluded.client_seq >= rating.client_seq
            """,
            (code, item_id, dimension, float(score), int(duration_sec),
             int(bool(duration_truncated)), int(bool(is_attention_check)), seq),
        )
    return True


_RATING_COLUMNS = ("code", "item_id", "dimension", "score", "duration_sec",
                   "duration_truncated", "is_attention_check")


def list_ratings(conn, code=None):
    """只回前端真正需要的列。新增内部列不会意外从 API 漏给持有 code 的人。"""
    cols = ", ".join(_RATING_COLUMNS)
    if code is None:
        rows = conn.execute(f"SELECT {cols} FROM rating ORDER BY code, item_id").fetchall()
    else:
        rows = conn.execute(f"SELECT {cols} FROM rating WHERE code = ? ORDER BY item_id",
                            (code,)).fetchall()
    return [dict(r) for r in rows]


def mark_finished(conn, code):
    """只置状态位，不记时间。"""
    _ensure_rater(conn, code)
    with _WRITE_LOCK, conn:
        conn.execute("UPDATE rater SET finished = 1 WHERE code = ?", (code,))


def get_rater(conn, code):
    row = conn.execute("SELECT * FROM rater WHERE code = ?", (code,)).fetchone()
    return dict(row) if row else None


def save_demographics(conn, code, *, degree, field, own_project):
    """存知情同意通过后的自陈信息，并据此划专业 / 非专业组。

    划组规则集中在这一个函数：控制方向且实际做过控制系统设计类课题才算专业组。
    """
    expertise = "expert" if (field == "control" and own_project == "yes") else "nonexpert"
    demographics = {"degree": degree, "field": field, "own_project": own_project}
    _ensure_rater(conn, code)
    with _WRITE_LOCK, conn:
        conn.execute(
            "UPDATE rater SET expertise = ?, demographics_json = ? WHERE code = ?",
            (expertise, json.dumps(demographics, ensure_ascii=False), code),
        )
    return expertise
