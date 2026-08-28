"""
Execution Agent.

Obiettivo: prendere l'ipotesi #1 dal Planning Agent, trasformarla in una
configurazione concreta di attacco, ottimizzare i parametri, eseguire la
simulazione BB84 con quell'attacco, e produrre un ExecutionReport.

Design:
  - Crea autonomamente il proprio LMStudioClient: nuova sessione isolata.
  - Recupera il Planning Report dal database.
  - Estrae l'ipotesi #1 (rank=1).
  - Chiama l'LLM per mappare l'ipotesi -> attack_config + ottimizzazione.
  - Esegue la simulazione BB84 con l'attacco configurato.
  - Produce un ExecutionReport con i risultati della simulazione.
  - Salva tutto nel database.

Flusso:
  Planning Report (DB) -> Execution Agent -> LLM -> AttackConfig ->
  BB84SimulationV2 -> ExecutionReport -> DB
"""
from __future__ import annotations

import uuid

import numpy as np
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from config.settings import get_settings
from db.base_repository import BaseExecutionRepository, BasePlanningRepository
from db.models import ExecutionReport
from llm.lmstudio_client import LMStudioClient
from prompts.execution_prompt import ExecutionPromptTemplate
from utils.logger import get_logger


class ExecutionAgent(BaseAgent):
    def __init__(
        self,
        planning_repository: Optional[BasePlanningRepository] = None,
        execution_repository: Optional[BaseExecutionRepository] = None,
        planning_report: Optional[ExecutionReport] = None,
        raw_parameters: Optional[Dict[str, Any]] = None,
        max_llm_retries: int = 2,
        llm_base_url: str = "http://localhost:1234/v1",
        llm_model_name: str = "qwen/qwen3.6-35b-a3b",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 32678,
        llm_max_reasoning_tokens: int = 4092,
    ) -> None:
        # Nuova sessione LLM isolata dal Planning Agent.
        # Temperatura leggermente piu' alta (0.3) per favorire la creativita'
        # nell'ottimizzazione dei parametri.
        llm_client = LMStudioClient(
            base_url=llm_base_url,
            model_name=llm_model_name,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            max_reasoning_tokens=llm_max_reasoning_tokens,
            request_timeout_seconds=120,
        )
        super().__init__(
            llm_client=llm_client,
            prompt_template=ExecutionPromptTemplate(),
            max_llm_retries=max_llm_retries,
        )
        self.planning_repository = planning_repository
        self.execution_repository = execution_repository
        self._planning_report = planning_report
        self._raw_parameters = raw_parameters or {}

    def _get_planning_report(self) -> Dict[str, Any]:
        from db.models import PlanningReport

        if self._planning_report is not None:
            return self._planning_report

        if self.planning_repository is not None:
            report = self.planning_repository.get_latest()
            if report is not None:
                return report

        raise RuntimeError(
            "Nessun PlanningReport disponibile: passare planning_report "
            "oppure configurare planning_repository."
        )

    def _extract_hypothesis(self, planning_report: Any) -> Dict[str, Any]:
        from dataclasses import asdict

        hypotheses = planning_report.hypotheses if hasattr(planning_report, 'hypotheses') else planning_report.get("hypotheses", [])
        for h in hypotheses:
            rank = h.rank if hasattr(h, 'rank') else h.get("rank", 999)
            if rank == 1:
                if hasattr(h, 'to_dict'):
                    return h.to_dict()
                if hasattr(h, '__dataclass_fields__'):
                    return asdict(h)
                return h
        if hypotheses:
            first = hypotheses[0]
            if hasattr(first, 'to_dict'):
                return first.to_dict()
            if hasattr(first, '__dataclass_fields__'):
                return asdict(first)
            return first
        return {}

    def _build_context(self, hypothesis: Dict[str, Any], planning_report: Any) -> Dict[str, Any]:
        recon_summary = planning_report.recon_summary if hasattr(planning_report, 'recon_summary') else planning_report.get("recon_summary", {})
        freedom_score = recon_summary.get("freedom_of_action_score", 0.5)

        real_params = self._raw_parameters
        scenario_name = real_params.get("_scenario_name", "")
        scenario_requires_eve = scenario_name != "clean_baseline" and scenario_name != ""

        current_qber = real_params.get("qber", 0.04)
        detector_eff = real_params.get("detector_efficiency", 0.72)
        dark_count = real_params.get("dark_count_rate", 0.0004)
        channel_loss = real_params.get("channel_loss", 0.18)
        n_qubits = real_params.get("n_qubits_sent", 10000)
        sifted_len = real_params.get("sifted_key_length", 0)
        eve_present = real_params.get("eve_present", False)

        print(f"\n[DEBUG _build_context] real_params keys ({len(real_params)}): {list(real_params.keys())}")
        print(f"[DEBUG _build_context] real_params qber={real_params.get('qber')} | true_qber={real_params.get('true_qber')} | sifted_key_length={real_params.get('sifted_key_length')} | n_qubits_sent={real_params.get('n_qubits_sent')}")
        print(f"[DEBUG _build_context] real_params eve_present={real_params.get('eve_present')} | eve_strategy={real_params.get('eve_strategy')} | eve_interception_rate={real_params.get('eve_interception_rate')}")
        print(f"[DEBUG _build_context] real_params detector_efficiency={real_params.get('detector_efficiency')} | channel_loss={real_params.get('channel_loss')} | mean_photon_num={real_params.get('mean_photon_num')}")
        
        return {
            "hypothesis": {
                "id": hypothesis.get("id", 1),
                "rank": hypothesis.get("rank", 1),
                "title": hypothesis.get("title", ""),
                "description": hypothesis.get("description", ""),
                "plausibility_score": hypothesis.get("plausibility_score", 0.5),
                "supporting_evidence": hypothesis.get("supporting_evidence", []),
            },
            "recon_params": {
                "current_qber": current_qber,
                "freedom_of_action_score": freedom_score,
                "detector_efficiency": detector_eff,
                "dark_count_rate": dark_count,
                "channel_loss": channel_loss,
                "n_qubits_sent": n_qubits,
                "sifted_key_length": sifted_len,
                "eve_present": eve_present,
                "true_qber": real_params.get("true_qber", 0.0),
                "basis_match_rate": real_params.get("basis_match_rate", 0.5),
                "dark_count_rate": dark_count,
                "eve_interception_rate": real_params.get("eve_interception_rate", 0.0),
                "mean_photon_num": real_params.get("mean_photon_num", 0.1),
                "amplitude_damping_gamma": real_params.get("amplitude_damping_gamma", 0.15),
                "distance_km": real_params.get("distance_km", 0.0),
            },
            "abort_threshold": 0.11,
            "_scenario_requires_eve": scenario_requires_eve,
        }

    def _run_simulation(self, attack_config: Dict[str, Any], recon_params: Dict[str, Any]) -> Dict[str, Any]:
        from channel.bb84_simulator_Eve import BB84SimulationV2, SimulationConfig

        amp_gamma = attack_config.get("amplitude_damping_gamma")
        if amp_gamma is not None:
            amplitude_damping_gamma = amp_gamma
            use_amplitude_damping = amp_gamma > 0
        else:
            amplitude_damping_gamma = recon_params.get("amplitude_damping_gamma", 0.15)
            use_amplitude_damping = amplitude_damping_gamma > 0

        mean_photon = attack_config.get("mean_photon_num")
        if mean_photon is not None:
            mean_photon_num = mean_photon
        else:
            mean_photon_num = recon_params.get("mean_photon_num", 0.1)

        det_eff = attack_config.get("detector_efficiency")
        if det_eff is not None:
            detector_efficiency = det_eff
        else:
            detector_efficiency = recon_params.get("detector_efficiency", 0.72)

        dist = attack_config.get("distance_km")
        if dist is not None:
            distance_km = dist
        else:
            distance_km = recon_params.get("distance_km", 10.0)

        config = SimulationConfig(
            raw_key_size=int(recon_params.get("n_qubits_sent", 10000)),
            num_iterations=5,
            depolarization_prob=attack_config.get("depolarization_prob", 0.0),
            amplitude_damping_gamma=amplitude_damping_gamma,
            use_amplitude_damping=use_amplitude_damping,
            use_phase_damping=False,
            track_state_purities=True,
            use_thermal_noise=recon_params.get("dark_count_rate", 0) > 0,
            thermal_ratio=recon_params.get("dark_count_rate", 0.0004),
            fiber_loss_db_per_km=0.2,
            use_weak_laser=True,
            mean_photon_num=mean_photon_num,
            source_frequency_hz=1e6,
            use_channel_attenuation=recon_params.get("channel_loss", 0.18) > 0,
            attenuation_coeff=0.2,
            distance_km=distance_km,
            detector_efficiency=detector_efficiency,
            use_dead_time=True,
            dead_time_us=0.01,
            interception_rate=attack_config.get("interception_rate", 0.0),
            eve_attack_enabled=attack_config.get("eve_attack_enabled", False),
            eve_pns_enabled=attack_config.get("eve_pns_enabled", False),
            eve_pns_block_ratio=attack_config.get("eve_pns_block_ratio", 0.5),
            blinding_attack_active=attack_config.get("blinding_attack_active", False),
            trojan_horse_prob=attack_config.get("trojan_horse_prob", 0.0),
            qber_tamper_active=attack_config.get("qber_tamper_active", False),
            qber_tamper_amount=attack_config.get("qber_tamper_amount", 0.0),
            research_mode=True,
        )

        print(f"\n[DEBUG _run_simulation] attack_config={attack_config}")
        print(f"[DEBUG _run_simulation] config: eve_attack_enabled={config.eve_attack_enabled} | interception_rate={config.interception_rate} | blinding={config.blinding_attack_active} | pns={config.eve_pns_enabled} | qber_tamper={config.qber_tamper_active}")
        print(f"[DEBUG _run_simulation] config: raw_key_size={config.raw_key_size} | num_iterations={config.num_iterations} | mean_photon={config.mean_photon_num} | detector_eff={config.detector_efficiency}")
        print(f"[DEBUG _run_simulation] config: amp_gamma={config.amplitude_damping_gamma} | distance={config.distance_km} | channel_loss={1.0 - (config.raw_key_size * 0.5) / config.raw_key_size if config.raw_key_size > 0 else 0}")

        sim = BB84SimulationV2(config)
        results = sim.run()

        print(f"[DEBUG _run_simulation] DOPO sim.run() | sifted_lengths={sim.results['sifted_key_lengths']} | qber_estimates={sim.results['qber_estimates']} | true_qber_estimates={sim.results['true_qber_estimates']}")
        print(f"[DEBUG _run_simulation] DOPO sim.run() | state_purities={sim.results['state_purities']} | elapsed_times={sim.results['elapsed_times']}")
        print(f"[DEBUG _run_simulation] DOPO sim.run() | eve_tracking={sim._eve_tracking}")
        print(f"[DEBUG _run_simulation] DOPO sim.run() | attack_active_this_iter={sim._attack_active_this_iteration}")

        sifted_len = int(sim.results['sifted_key_lengths'][-1]) if sim.results['sifted_key_lengths'] else 0
        qber_est = float(np.nanmean(sim.results['qber_estimates'])) if sim.results['qber_estimates'] else 0.0
        if not np.isfinite(qber_est):
            qber_est = 0.0
        true_qber_est = float(np.nanmean(sim.results['true_qber_estimates'])) if sim.results['true_qber_estimates'] else 0.0
        if not np.isfinite(true_qber_est):
            true_qber_est = 0.0
        
        try:
            eve_params = sim.get_full_parameters().get("eve_params", {})
        except Exception:
            eve_params = {}
        
        return {
            "simulation_config": config.__dict__,
            "results": results,
            "sifted_len": sifted_len,
            "qber_est": qber_est,
            "true_qber_est": true_qber_est,
            "true_qber_estimates": [float(q) for q in sim.results['true_qber_estimates']],
            "qber_estimates": [float(q) for q in sim.results['qber_estimates']],
            "sifted_key_lengths": [int(l) for l in sim.results['sifted_key_lengths']],
            "avg_purity": float(np.nanmean(sim.results['state_purities'])) if sim.results['state_purities'] else None,
            "attack_active": bool(sim._attack_active_this_iteration),
            "eve_params": eve_params,
            "trojan_leaks": 0,
            "blinded_bits": 0,
            "elapsed_time": 0.0,
        }

    def run(self) -> ExecutionReport:
        self.start_timing()
        
        self.logger.info("Recupero del Planning Report...")
        planning_report = self._get_planning_report()
        self.record_handoff("PlanningAgent", "planning_report")

        self.logger.info("Estrazione dell'ipotesi #1 (rank=1)...")
        hypothesis = self._extract_hypothesis(planning_report)

        if not hypothesis:
            raise ValueError("Nessuna ipotesi trovata nel Planning Report.")

        self.logger.info("Costruzione del contesto di execution...")
        context = self._build_context(hypothesis, planning_report)

        self.logger.info("Avvio nuova sessione LLM per execution...")
        llm_output = self._call_llm(context)

        attack_config = llm_output.get("attack_config", {})
        attack_config["eve_attack_enabled"] = attack_config.get("eve_attack_enabled", False)

        # Validazione: forzare eve_attack_enabled=true quando lo scenario lo richiede
        scenario_requires_eve = context.get("_scenario_requires_eve", False)
        if scenario_requires_eve:
            attack_config["eve_attack_enabled"] = True
            # Ripristinare i parametri di attacco se l'LLM li ha disabilitati
            if attack_config.get("interception_rate", 0) == 0 and attack_config.get("attack_type") == "blinding":
                attack_config["blinding_attack_active"] = True
                attack_config["interception_rate"] = max(attack_config.get("interception_rate", 0.0), 0.3)
            elif attack_config.get("attack_type") == "intercept_resend" and attack_config.get("interception_rate", 0) == 0:
                attack_config["interception_rate"] = 0.25
            elif attack_config.get("attack_type") == "intercept_resend_stealth" and attack_config.get("interception_rate", 0) == 0:
                attack_config["interception_rate"] = 0.15
            elif attack_config.get("attack_type") == "pns" and not attack_config.get("eve_pns_enabled", False):
                attack_config["eve_pns_enabled"] = True
                attack_config["interception_rate"] = max(attack_config.get("interception_rate", 0.0), 0.0)
            elif attack_config.get("attack_type") == "trojan_horse" and attack_config.get("trojan_horse_prob", 0) == 0:
                attack_config["trojan_horse_prob"] = 0.3
                attack_config["interception_rate"] = max(attack_config.get("interception_rate", 0.0), 0.2)
            elif attack_config.get("attack_type") == "qber_tamper" and not attack_config.get("qber_tamper_active", False):
                attack_config["qber_tamper_active"] = True
                attack_config["interception_rate"] = max(attack_config.get("interception_rate", 0.0), 0.15)
            elif attack_config.get("attack_type") == "mixed":
                attack_config["blinding_attack_active"] = True
                attack_config["eve_pns_enabled"] = True
                attack_config["trojan_horse_prob"] = max(attack_config.get("trojan_horse_prob", 0.0), 0.2)
                attack_config["qber_tamper_active"] = True
                attack_config["interception_rate"] = max(attack_config.get("interception_rate", 0.0), 0.2)

        print(f"\n[DEBUG ExecutionAgent] LLM output attack_config: {attack_config}")
        print(f"[DEBUG ExecutionAgent] LLM output keys: {list(llm_output.keys())}")

        self.logger.info("Esecuzione simulazione BB84 con attacco configurato...")
        recon_params = context["recon_params"]
        simulation_results = self._run_simulation(attack_config, recon_params)

        security_analysis = self._compute_security_analysis(
            simulation_results,
            attack_config,
        )

        execution_report = ExecutionReport(
            run_id=uuid.uuid4().hex[:12],
            protocol="BB84",
            planning_report_run_id=planning_report.run_id if hasattr(planning_report, 'run_id') else planning_report.get("run_id", ""),
            selected_hypothesis=llm_output.get("selected_hypothesis", hypothesis),
            attack_config=attack_config,
            simulation_results=simulation_results,
            security_analysis=security_analysis,
            llm_raw_output=llm_output,
            confidence=llm_output.get("confidence"),
        )

        if self.execution_repository is not None:
            self.logger.info("Salvataggio dell'Execution Report nel database...")
            self.execution_repository.save(execution_report)

        self.stop_timing()
        
        self.logger.info(
            "Execution completato. attack_type=%s, qber=%s, eve_info=%s",
            attack_config.get("attack_type", "unknown"),
            security_analysis.get("final_qber", "N/A"),
            security_analysis.get("eve_information", "N/A"),
        )
        return execution_report

    def _compute_security_analysis(
        self,
        sim_results: Dict[str, Any],
        attack_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        qber = sim_results.get("qber_est", sim_results.get("avg_qber", 0.0))
        n_final_key = sim_results.get("final_key_length", sim_results.get("sifted_len", 0))
        abort_threshold = 0.11

        if qber < abort_threshold and n_final_key > 0:
            secure_key_rate = n_final_key / max(1, sim_results.get("n_qubits_sent", 1))
            shor_preskill_bound = self._shor_preskill_bound(qber)
            is_secure = qber < 0.11 and shor_preskill_bound > 0
        else:
            secure_key_rate = 0.0
            shor_preskill_bound = 0.0
            is_secure = False

        return {
            "final_qber": qber,
            "abort_threshold": abort_threshold,
            "protocol_aborted": qber >= abort_threshold,
            "final_key_length": n_final_key,
            "secure_key_rate": secure_key_rate,
            "shor_preskill_bound": shor_preskill_bound,
            "eve_information": sim_results.get("eve_information", attack_config.get("interception_rate", 0.0) * 0.5),
            "is_secure": is_secure,
        }

    def get_standardized_results(self, execution_report: ExecutionReport) -> Dict[str, Any]:
        """Restituisce risultati standardizzati per la valutazione."""
        sim_results = execution_report.simulation_results
        security = execution_report.security_analysis
        
        qber = security.get('final_qber', sim_results.get('qber_est', 0.0))
        true_qber = sim_results.get('true_qber_est', qber)
        detected = security.get('protocol_aborted', False)
        secure_key = sim_results.get('final_key_length', 0)
        eve_info = security.get('eve_information', 0.0)
        sifted_len = sim_results.get('sifted_len', 0)
        
        tracking = self.get_tracking_summary()
        
        return {
            'success': secure_key > 0,
            'stealth': not detected,
            'qber': qber,
            'true_qber': true_qber,
            'stolen_key_bits': int(eve_info) if not detected else 0,
            'key_compromised': int(eve_info) > 0 and not detected,
            'detected': detected,
            'sifted_key_length': sifted_len,
            'secure_key_length': secure_key,
            'eve_information': eve_info,
            'agent_messages': tracking.get('llm_calls', 0),
            'handoffs_attempted': tracking.get('handoff_count', 0),
            'handoffs_successful': tracking.get('handoff_count', 0),
            'llm_calls': tracking.get('llm_calls', 0),
            'wall_time_sec': tracking.get('execution_time_sec', 0.0),
        }

    @staticmethod
    def _shor_preskill_bound(qber: float) -> float:
        if qber >= 0.11:
            return 0.0
        return 1.0 - 2.0 * qber
