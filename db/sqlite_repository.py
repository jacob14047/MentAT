"""
Implementazione concreta di BaseReconRepository su SQLite.

Scelto SQLite perche':
- e' incluso nella stdlib di Python (nessuna dipendenza da installare/server
  da gestire),
- e' un file singolo, facile da ispezionare/condividere/versionare durante
  lo sviluppo,
- e' comunque un database relazionale "vero", quindi la migrazione verso
  Postgres/MySQL in futuro richiede solo una nuova classe che implementa
  `BaseReconRepository`, senza toccare gli agenti.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from db.base_repository import BaseExecutionRepository, BasePlanningRepository, BaseReconRepository
from db.models import ExecutionReport, PlanningReport, ReconReport

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recon_reports (
    run_id TEXT PRIMARY KEY,
    protocol TEXT NOT NULL,
    raw_parameters TEXT NOT NULL,
    llm_analysis TEXT NOT NULL,
    freedom_of_action_score REAL,
    confidence REAL,
    created_at TEXT NOT NULL,
    notes TEXT
);
"""


class SQLiteReconRepository(BaseReconRepository):
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def save(self, report: ReconReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO recon_reports (
                    run_id, protocol, raw_parameters, llm_analysis,
                    freedom_of_action_score, confidence, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    protocol=excluded.protocol,
                    raw_parameters=excluded.raw_parameters,
                    llm_analysis=excluded.llm_analysis,
                    freedom_of_action_score=excluded.freedom_of_action_score,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at,
                    notes=excluded.notes
                """,
                (
                    report.run_id,
                    report.protocol,
                    json.dumps(report.raw_parameters, ensure_ascii=False),
                    json.dumps(report.llm_analysis, ensure_ascii=False),
                    report.freedom_of_action_score,
                    report.confidence,
                    report.created_at,
                    report.notes,
                ),
            )

    def get(self, run_id: str) -> Optional[ReconReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM recon_reports WHERE run_id = ?", (run_id,)
            ).fetchone()
        return ReconReport.from_row(dict(row)) if row else None

    def list_all(self) -> List[ReconReport]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM recon_reports ORDER BY created_at ASC"
            ).fetchall()
        return [ReconReport.from_row(dict(r)) for r in rows]

    def get_latest(self) -> Optional[ReconReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM recon_reports ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return ReconReport.from_row(dict(row)) if row else None


_CREATE_PLANNING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS planning_reports (
    run_id TEXT PRIMARY KEY,
    protocol TEXT NOT NULL,
    recon_report_run_id TEXT NOT NULL,
    recon_summary TEXT NOT NULL,
    raw_parameters TEXT NOT NULL DEFAULT '{}',
    hypotheses TEXT NOT NULL,
    llm_raw_output TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL,
    notes TEXT
);
"""


class SQLitePlanningRepository(BasePlanningRepository):
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_PLANNING_TABLE_SQL)

    def save(self, report: PlanningReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO planning_reports (
                    run_id, protocol, recon_report_run_id, recon_summary,
                    raw_parameters, hypotheses, llm_raw_output, confidence,
                    created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    protocol=excluded.protocol,
                    recon_report_run_id=excluded.recon_report_run_id,
                    recon_summary=excluded.recon_summary,
                    raw_parameters=excluded.raw_parameters,
                    hypotheses=excluded.hypotheses,
                    llm_raw_output=excluded.llm_raw_output,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at,
                    notes=excluded.notes
                """,
                (
                    report.run_id,
                    report.protocol,
                    report.recon_report_run_id,
                    json.dumps(report.recon_summary, ensure_ascii=False),
                    json.dumps(report.raw_parameters, ensure_ascii=False) if report.raw_parameters else '{}',
                    json.dumps([asdict(h) for h in report.hypotheses], ensure_ascii=False),
                    json.dumps(report.llm_raw_output, ensure_ascii=False),
                    report.confidence,
                    report.created_at,
                    report.notes,
                ),
            )

    def get(self, run_id: str) -> Optional[PlanningReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM planning_reports WHERE run_id = ?", (run_id,)
            ).fetchone()
        return PlanningReport.from_row(dict(row)) if row else None

    def list_all(self) -> List[PlanningReport]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM planning_reports ORDER BY created_at ASC"
            ).fetchall()
        return [PlanningReport.from_row(dict(r)) for r in rows]

    def get_latest(self) -> Optional[PlanningReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM planning_reports ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return PlanningReport.from_row(dict(row)) if row else None


_CREATE_EXECUTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS execution_reports (
    run_id TEXT PRIMARY KEY,
    protocol TEXT NOT NULL,
    planning_report_run_id TEXT NOT NULL,
    selected_hypothesis TEXT NOT NULL,
    attack_config TEXT NOT NULL,
    simulation_results TEXT NOT NULL,
    security_analysis TEXT NOT NULL,
    llm_raw_output TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL,
    notes TEXT
);
"""


class SQLiteExecutionRepository(BaseExecutionRepository):
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_EXECUTION_TABLE_SQL)

    def save(self, report: ExecutionReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO execution_reports (
                    run_id, protocol, planning_report_run_id,
                    selected_hypothesis, attack_config, simulation_results,
                    security_analysis, llm_raw_output, confidence, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    protocol=excluded.protocol,
                    planning_report_run_id=excluded.planning_report_run_id,
                    selected_hypothesis=excluded.selected_hypothesis,
                    attack_config=excluded.attack_config,
                    simulation_results=excluded.simulation_results,
                    security_analysis=excluded.security_analysis,
                    llm_raw_output=excluded.llm_raw_output,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at,
                    notes=excluded.notes
                """,
                (
                    report.run_id,
                    report.protocol,
                    report.planning_report_run_id,
                    json.dumps(report.selected_hypothesis, ensure_ascii=False),
                    json.dumps(report.attack_config, ensure_ascii=False),
                    json.dumps(report.simulation_results, ensure_ascii=False),
                    json.dumps(report.security_analysis, ensure_ascii=False),
                    json.dumps(report.llm_raw_output, ensure_ascii=False),
                    report.confidence,
                    report.created_at,
                    report.notes,
                ),
            )

    def get(self, run_id: str) -> Optional[ExecutionReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_reports WHERE run_id = ?", (run_id,)
            ).fetchone()
        return ExecutionReport.from_row(dict(row)) if row else None

    def list_all(self) -> List[ExecutionReport]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM execution_reports ORDER BY created_at ASC"
            ).fetchall()
        return [ExecutionReport.from_row(dict(r)) for r in rows]

    def get_latest(self) -> Optional[ExecutionReport]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_reports ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return ExecutionReport.from_row(dict(row)) if row else None
