"""
Planning Agent.

Obiettivo: prendere le informazioni raccolte dal Recon Agent e generare
una lista di ipotesi d'attacco ordinate per plausibilita', che formino
il contesto perfetto per un successivo Execution Agent.

Design:
  - Crea autonomamente il proprio LMStudioClient: sessione LLM isolata
    dal Recon Agent, zero condivisione di contesto/token.
  - Utilizza un contesto compresso dal Recon Report: solo i dati
    essenziali (summary, weaknesses, constraints, focus areas), non
    il dump completo dei raw parameters.
  - Flusso:
    1. Recupera il Recon Report (da repository o passed directly).
    2. Estrae un riassunto strutturato (contesto compresso).
    3. Chiama l'LLM con il prompt di "planning".
    4. Parsa le ipotesi JSON restituite.
    5. Costruisce un PlanningReport e lo salva nel repository.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from config.settings import get_settings
from db.base_repository import BasePlanningRepository, BaseReconRepository
from db.models import Hypothesis, PlanningReport, ReconReport
from llm.lmstudio_client import LMStudioClient
from prompts.planning_prompt import PlanningPromptTemplate
from utils.logger import get_logger


class PlanningAgent(BaseAgent):
    def __init__(
        self,
        recon_repository: Optional[BaseReconRepository] = None,
        planning_repository: Optional[BasePlanningRepository] = None,
        recon_report: Optional[ReconReport] = None,
        raw_parameters: Optional[Dict[str, Any]] = None,
        max_llm_retries: int = 2,
        llm_base_url: str = "http://localhost:1234/v1",
        llm_model_name: str = "qwen/qwen3.6-35b-a3b",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 32678,
        llm_max_reasoning_tokens: int = 4092,
    ) -> None:
        # Crea un client LLM autonomo: nuova sessione isolata dal Recon Agent.
        # La temperatura piu' alta (0.3 vs 0.2 del recon) favorisce la
        # creativita' nella generazione di ipotesi.
        llm_client = LMStudioClient(
            base_url=llm_base_url,
            model_name=llm_model_name,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            max_reasoning_tokens=llm_max_reasoning_tokens,
        )
        super().__init__(
            llm_client=llm_client,
            prompt_template=PlanningPromptTemplate(),
            max_llm_retries=max_llm_retries,
        )
        self.recon_repository = recon_repository
        self.planning_repository = planning_repository
        self._recon_report = recon_report
        self._raw_parameters = raw_parameters or {}

    def _get_recon_report(self) -> ReconReport:
        if self._recon_report is not None:
            return self._recon_report

        if self.recon_repository is not None:
            report = self.recon_repository.get_latest()
            if report is not None:
                return report

        raise RuntimeError(
            "Nessun ReconReport disponibile: passare recon_report "
            "oppure configurare recon_repository."
        )

    def _build_context(self, recon_report: ReconReport) -> Dict[str, Any]:
        """Costruisce un contesto COMPRESSO dal Recon Report.

        Invece di passare tutto il dump dei raw parameters (che spreca token),
        estrae solo le informazioni strutturate dal llm_analysis del Recon Agent.
        """
        analysis = recon_report.llm_analysis

        return {
            "recon_analysis": {
                "attack_surface_summary": analysis.get("attack_surface_summary", ""),
                "freedom_of_action_score": analysis.get("freedom_of_action_score"),
                "confidence": analysis.get("confidence"),
                "rationale": analysis.get("rationale", ""),
                "exploitable_weaknesses": analysis.get("exploitable_weaknesses", []),
                "recommended_focus_areas_for_planning": analysis.get(
                    "recommended_focus_areas_for_planning", []
                ),
                "constraints_on_eve": analysis.get("constraints_on_eve", []),
            },
            "raw_parameters": {
                "qber": analysis.get("qber", recon_report.raw_parameters.get("qber")),
                "channel_loss": recon_report.raw_parameters.get("channel_loss"),
                "detector_efficiency": recon_report.raw_parameters.get("detector_efficiency"),
                "dark_count_rate": recon_report.raw_parameters.get("dark_count_rate"),
                "eve_present": recon_report.raw_parameters.get("eve_present"),
                "eve_strategy": recon_report.raw_parameters.get("eve_strategy"),
                "eve_interception_rate": recon_report.raw_parameters.get("eve_interception_rate"),
                "protocol_aborted": recon_report.raw_parameters.get("protocol_aborted"),
                "n_qubits_sent": recon_report.raw_parameters.get("n_qubits_sent"),
                "sifted_key_length": recon_report.raw_parameters.get("sifted_key_length"),
            },
            "metadata": {
                "run_id": recon_report.run_id,
                "protocol": recon_report.protocol,
            },
        }

    def _parse_hypotheses(self, llm_output: dict) -> list[Hypothesis]:
        hypotheses_raw = llm_output.get("hypotheses", [])[:2]
        hypotheses = []
        for i, h in enumerate(hypotheses_raw):
            hypothesis = Hypothesis(
                id=h.get("id", i + 1),
                rank=h.get("rank", i + 1),
                title=h.get("title", ""),
                description=h.get("description", ""),
                supporting_evidence=h.get("supporting_evidence", []),
                plausibility_score=float(h.get("plausibility_score", 0.0)),
                required_recon_data=h.get("required_recon_data", []),
                next_steps=h.get("next_steps", []),
            )
            hypotheses.append(hypothesis)
        return hypotheses

    def run(self) -> PlanningReport:
        self.logger.info("Recupero del Recon Report...")
        recon_report = self._get_recon_report()

        self.logger.info(
            "Costruzione del contesto compresso dal Recon Report (run_id=%s)...",
            recon_report.run_id,
        )
        context = self._build_context(recon_report)

        self.logger.info("Avvio nuova sessione LLM per planning...")
        print(f"\n[DEBUG PlanningAgent] raw_parameters received ({len(self._raw_parameters)} keys): {list(self._raw_parameters.keys())}")
        print(f"[DEBUG PlanningAgent] raw_parameters qber={self._raw_parameters.get('qber')} | sifted={self._raw_parameters.get('sifted_key_length')} | eve_present={self._raw_parameters.get('eve_present')}")
        print(f"[DEBUG PlanningAgent] recon_analysis freedom_score={context['recon_analysis'].get('freedom_of_action_score')} | confidence={context['recon_analysis'].get('confidence')}")
        llm_output = self._call_llm(context)
        print(f"[DEBUG PlanningAgent] LLM output hypotheses count={len(llm_output.get('hypotheses', []))}")
        for i, h in enumerate(llm_output.get('hypotheses', [])):
            print(f"[DEBUG PlanningAgent]   hypothesis[{i}]: rank={h.get('rank')} | id={h.get('id')} | title={h.get('title','?')[:50]} | plausibility={h.get('plausibility_score')}")

        self.logger.info("Parsing delle ipotesi generate (%d trovate)...", len(llm_output.get("hypotheses", [])))
        hypotheses = self._parse_hypotheses(llm_output)

        confidence = llm_output.get("confidence")

        planning_report = PlanningReport(
            run_id=uuid.uuid4().hex[:12],
            protocol=recon_report.protocol,
            recon_report_run_id=recon_report.run_id,
            recon_summary=recon_report.llm_analysis,
            raw_parameters=self._raw_parameters,
            hypotheses=hypotheses,
            llm_raw_output=llm_output,
            confidence=confidence,
        )

        if self.planning_repository is not None:
            self.logger.info("Salvataggio del Planning Report nel database...")
            self.planning_repository.save(planning_report)

        self.logger.info(
            "Planning completato. %d ipotesi generate, confidence=%s",
            len(hypotheses),
            confidence,
        )
        return planning_report
