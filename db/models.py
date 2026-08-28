"""
Modello dati del risultato prodotto dal Recon Agent.

Questo e' l'oggetto che viaggia tra `agents/recon_agent.py`, il database e,
in futuro, il Planning Agent: e' il "contratto" tra le due fasi.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ReconReport:
    run_id: str
    protocol: str
    raw_parameters: Dict[str, Any]
    llm_analysis: Dict[str, Any]
    freedom_of_action_score: Optional[float] = None
    confidence: Optional[float] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def __post_init__(self) -> None:
        # Comodo: se il punteggio non e' passato esplicitamente, prova a
        # leggerlo dall'analisi LLM (dove il prompt lo richiede).
        if self.freedom_of_action_score is None:
            self.freedom_of_action_score = self.llm_analysis.get(
                "freedom_of_action_score"
            )
        if self.confidence is None:
            self.confidence = self.llm_analysis.get("confidence")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ReconReport":
        return cls(
            run_id=row["run_id"],
            protocol=row["protocol"],
            raw_parameters=json.loads(row["raw_parameters"]),
            llm_analysis=json.loads(row["llm_analysis"]),
            freedom_of_action_score=row.get("freedom_of_action_score"),
            confidence=row.get("confidence"),
            created_at=row.get("created_at"),
            notes=row.get("notes", ""),
        )


@dataclass
class Hypothesis:
    id: int
    rank: int
    title: str
    description: str
    supporting_evidence: List[str]
    plausibility_score: float
    required_recon_data: List[str]
    next_steps: List[str]


@dataclass
class PlanningReport:
    run_id: str
    protocol: str
    recon_report_run_id: str
    recon_summary: Dict[str, Any]
    raw_parameters: Dict[str, Any]
    hypotheses: List[Hypothesis]
    llm_raw_output: Dict[str, Any]
    confidence: Optional[float] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["hypotheses"] = [asdict(h) for h in self.hypotheses]
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PlanningReport":
        hypotheses_raw = json.loads(row["hypotheses"])
        hypotheses = [
            Hypothesis(
                id=h["id"],
                rank=h["rank"],
                title=h["title"],
                description=h["description"],
                supporting_evidence=h["supporting_evidence"],
                plausibility_score=h["plausibility_score"],
                required_recon_data=h["required_recon_data"],
                next_steps=h["next_steps"],
            )
            for h in hypotheses_raw
        ]
        recon_summary = json.loads(row["recon_summary"])
        raw_params = row.get("raw_parameters")
        raw_parameters = json.loads(raw_params) if raw_params else {}
        return cls(
            run_id=row["run_id"],
            protocol=row["protocol"],
            recon_report_run_id=row["recon_report_run_id"],
            recon_summary=recon_summary,
            raw_parameters=raw_parameters,
            hypotheses=hypotheses,
            llm_raw_output=json.loads(row["llm_raw_output"]),
            confidence=row.get("confidence"),
            created_at=row.get("created_at"),
            notes=row.get("notes", ""),
        )


@dataclass
class AttackConfig:
    attack_type: str
    interception_rate: float
    eve_pns_enabled: bool
    eve_pns_block_ratio: float
    blinding_attack_active: bool
    trojan_horse_prob: float
    qber_tamper_active: bool
    qber_tamper_amount: float
    amplitude_damping_gamma: float
    depolarization_prob: float
    detector_efficiency: float
    dark_count_rate: float
    channel_loss: float
    mean_photon_num: float
    distance_km: float


@dataclass
class ExecutionReport:
    run_id: str
    protocol: str
    planning_report_run_id: str
    selected_hypothesis: Dict[str, Any]
    attack_config: Dict[str, Any]
    simulation_results: Dict[str, Any]
    security_analysis: Dict[str, Any]
    llm_raw_output: Dict[str, Any]
    confidence: Optional[float] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ExecutionReport":
        return cls(
            run_id=row["run_id"],
            protocol=row["protocol"],
            planning_report_run_id=row["planning_report_run_id"],
            selected_hypothesis=json.loads(row["selected_hypothesis"]),
            attack_config=json.loads(row["attack_config"]),
            simulation_results=json.loads(row["simulation_results"]),
            security_analysis=json.loads(row["security_analysis"]),
            llm_raw_output=json.loads(row["llm_raw_output"]),
            confidence=row.get("confidence"),
            created_at=row.get("created_at"),
            notes=row.get("notes", ""),
        )
