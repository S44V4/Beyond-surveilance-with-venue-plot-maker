from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CROWD_DATA_DIR", ROOT / "runtime")).resolve()


def uid():
    return uuid.uuid4().hex


@contextmanager
def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / "crowd.db", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS objects (kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, body TEXT NOT NULL, error TEXT, created REAL NOT NULL, updated REAL NOT NULL, idem TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS frames (run_id TEXT NOT NULL, idx INTEGER NOT NULL, timestamp REAL NOT NULL, body TEXT NOT NULL, PRIMARY KEY(run_id,idx));
        CREATE INDEX IF NOT EXISTS frame_time ON frames(run_id,timestamp);
        """)


def put(kind, id, body):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO objects VALUES (?,?,?,?)", (kind,id,json.dumps(body,allow_nan=False),time.time()))


def get(kind, id):
    with connect() as db:
        row = db.execute("SELECT body FROM objects WHERE kind=? AND id=?", (kind,id)).fetchone()
    return json.loads(row[0]) if row else None


def listing(kind, limit=50, offset=0):
    with connect() as db:
        rows = db.execute("SELECT body FROM objects WHERE kind=? ORDER BY created DESC LIMIT ? OFFSET ?",(kind,limit,offset)).fetchall()
    return [json.loads(r[0]) for r in rows]


def job(row):
    if row is None:
        return None
    value = dict(row)
    value["body"] = json.loads(value["body"])
    return value


def create_job(kind, body, idem, status="queued"):
    body=json.loads(json.dumps(body,allow_nan=False))
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO jobs(id,kind,status,body,created,updated,idem) VALUES (?,?,?,?,?,?,?)", (uid(),kind,status,json.dumps(body,allow_nan=False),time.time(),time.time(),idem))
        row = db.execute("SELECT * FROM jobs WHERE idem=?",(idem,)).fetchone()
    result = job(row)
    if result["kind"] != kind or result["body"] != body:
        raise ValueError("Idempotency key was already used for different inputs")
    return result


def get_job(id):
    with connect() as db:
        return job(db.execute("SELECT * FROM jobs WHERE id=?",(id,)).fetchone())


def jobs(kind=None, limit=50, offset=0):
    with connect() as db:
        rows = db.execute("SELECT * FROM jobs WHERE kind=? ORDER BY created DESC LIMIT ? OFFSET ?",(kind,limit,offset)).fetchall() if kind else db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ? OFFSET ?",(limit,offset)).fetchall()
    return [job(r) for r in rows]


def update_job(id, status, progress=None, error=None):
    with connect() as db:
        db.execute("UPDATE jobs SET status=?,progress=COALESCE(?,progress),error=?,updated=? WHERE id=?",(status,progress,error,time.time(),id))


def progress(id, value):
    with connect() as db:
        db.execute("UPDATE jobs SET progress=?,updated=? WHERE id=? AND status='running'",(value,time.time(),id))


def next_job():
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?",(time.time(),row["id"]))
    return job(row)


def add_frame(run, idx, body):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO frames VALUES(?,?,?,?)",(run,idx,body["timestamp"],json.dumps(body,allow_nan=False)))


def frames(run, offset=0, limit=300, lightweight=False):
    with connect() as db:
        rows = db.execute("SELECT body FROM frames WHERE run_id=? ORDER BY idx LIMIT ? OFFSET ?",(run,limit,offset)).fetchall()
        total = db.execute("SELECT COUNT(*) FROM frames WHERE run_id=?",(run,)).fetchone()[0]
    values = [json.loads(r[0]) for r in rows]
    if lightweight:
        values = [{k:v for k,v in x.items() if k not in {"heatmap","points","forecasts","learned"}} for x in values]
    return {"items":values,"total":total,"offset":offset}


def frame(run, idx):
    with connect() as db:
        row = db.execute("SELECT body FROM frames WHERE run_id=? AND idx=?",(run,idx)).fetchone()
    return json.loads(row[0]) if row else None
