"""
AdaptiveTrialRunner: esegue trial BB84 con variabilità esplorativa.

Ogni trial:
1. Genera parametri del canale casuali (jitter realistico)
2. Esegue Recon → Planning → Execution
3. [Opzionale] Ciclo adattivo: Execution → Recon → Planning → Execution
4. Raccoglie metriche RQ3 (adattamento, esplorazione, collaborazione)

Design:
- Classe separata da EvaluationRunner per non rompere la valutazione esistente
- I parametri del canale variano tra trial per simulare dispositivi reali
- Il ciclo adattivo permette agli agenti di rispondere all'esito della simulazione
- L'Execution Agent è vincolato a rispettare il tipo di attacco scelto dal Planning
"""
import time
import uuid
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents.recon_agent import ReconAgent
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from channel.bb84_channel_source import BB84ChannelSource
from channel.bb84_simulator_Eve import BB84SimulationV2, SimulationConfig
from llm.lmstudio_client import LMStudioClient
from evaluation.trial_logger import TrialResult, TrialLogger


# ============================================================================
# Config
# ============================================================================

@dataclass
class AdaptiveConfig:
    """Configurazione per trial esplorativa con ciclo adattivo."""
    
    # Seed per riproducibilità
    seed: int = 42
    
    # Dimensione della chiave e iterazioni della simulazione
    raw_key_size: int = 5000
    num_iterations: int = 5
    
    # Jitter dei parametri del canale (distribuzione normale)
    detector_efficiency_mean: float = 0.72
    detector_efficiency_std: float = 0.05
    dark_count_rate_mean: float = 0.0004
    dark_count_rate_std: float = 0.00015
    channel_loss_mean: float = 0.18
    channel_loss_std: float = 0.04
    mean_photon_num_mean: float = 0.1
    mean_photon_num_std: float = 0.03
    distance_km_mean: float = 10.0
    distance_km_std: float = 3.0
    
    # Ciclo adattivo
    enable_adaptive_loop: bool = True
    max_adaptation_rounds: int = 3
    
    # Execution Agent temperature (più alta = più esplorazione)
    execution_temperature: float = 0.5
    
    # Directory output
    output_dir: str = "adaptive_output"
    
    # URL e modello LLM
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model_name: str = "qwen/qwen3.6-35b-a3b"
    llm_max_tokens: int = 32678
    llm_max_reasoning_tokens: int = 4092
    
    # Retry
    max_llm_retries: int = 2
    
    # QBER abort threshold
    qber_abort_threshold: float = 0.11

    # Fiber loss
    fiber_loss_db_per_km: float = 0.2
    
    # Sharing rate per QBER estimation
    sharing_rate: float = 0.1
    
    # Tracking calibrazione recon-driven
    calibrated_means: Dict[str, float] = field(default_factory=dict)
    calibration_updates_count: int = 0


# ============================================================================
# AdaptiveTrialRunner
# ============================================================================

class AdaptiveTrialRunner:
    """
    Esegue trial BB84 con variabilità esplorativa.
    
    Ogni trial parte da un canale con parametri casuali, esegue la pipeline
    Recon → Planning → Execution, e opzionalmente ripete il ciclo per
    adattare la strategia in base ai risultati.
    """
    
    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrialLogger(str(self.output_dir))
        self.calibrated_means: Dict[str, float] = {}
        self.calibration_updates_count: int = 0
    
    def run_trial(
        self,
        scenario_name: str,
        trial_idx: int,
        scenario_params: Optional[Dict[str, Any]] = None,
    ) -> TrialResult:
        """
        Esegue UNA trial esplorativa completa.
        
        Args:
            scenario_name: Nome dello scenario base (es. 'intercept_resend')
            trial_idx: Indice della trial (per logging)
            scenario_params: Parametri base dello scenario. Se None, usa SCENARIOS.
        
        Returns:
            TrialResult con metriche RQ1, RQ3, e dati adattivi.
        """
        start_time = time.time()
        run_id = f"adaptive_{scenario_name[:10]}_{trial_idx:04d}"
        
        print(f"\n{'='*70}")
        print(f"[DEBUG run_trial START] scenario={scenario_name} | trial={trial_idx} | run_id={run_id}")
        print(f"[DEBUG run_trial START] scenario_params={scenario_params}")
        
        # 1. Genera configurazione canale casuale
        channel_config = self._generate_random_channel_config(scenario_name, scenario_params)
        
        # 2. Primo ciclo: Recon → Planning → Execution
        recon_report = self._run_recon(channel_config, run_id)
        self._update_calibration_from_recon(recon_report)
        planning_report = self._run_planning(recon_report, run_id)
        
        # Inietta il nome dello scenario nei raw_parameters per la validazione
        if hasattr(planning_report, 'raw_parameters') and isinstance(planning_report.raw_parameters, dict):
            planning_report.raw_parameters["_scenario_name"] = scenario_name
        elif hasattr(planning_report, 'raw_parameters') and hasattr(planning_report.raw_parameters, '__setitem__'):
            planning_report.raw_parameters["_scenario_name"] = scenario_name
        
        execution_report = self._run_execution(planning_report, run_id)
        
        # 3. Inizializza tracking
        print(f"\n[DEBUG run_trial] DOPO prima execution | sifted_len={execution_report.simulation_results.get('sifted_len','?')} | qber_est={execution_report.simulation_results.get('qber_est','?')} | true_qber={execution_report.simulation_results.get('true_qber_est','?')}")
        if hasattr(execution_report, 'simulation_results') and execution_report.simulation_results:
            sr = execution_report.simulation_results
            if isinstance(sr, dict):
                print(f"[DEBUG run_trial] sim.results keys={list(sr.keys())} | sifted_lengths={sr.get('sifted_key_lengths','?')} | qber_estimates={sr.get('qber_estimates','?')} | true_qber_estimates={sr.get('true_qber_estimates','?')}")
        
        adaptation_history = []
        attack_types_tried = [execution_report.attack_config.get('attack_type', 'unknown')]
        qber_trajectory = [execution_report.security_analysis.get('final_qber', 0.0)]
        detected_trajectory = [execution_report.security_analysis.get('protocol_aborted', False)]
        
        llm_calls_per_agent = {
            'recon': recon_report.llm_analysis.get('_llm_calls', 0) if hasattr(recon_report, 'llm_analysis') else 0,
            'planning': planning_report.llm_raw_output.get('_llm_calls', 0) if hasattr(planning_report, 'llm_raw_output') else 0,
            'execution': execution_report.llm_raw_output.get('_llm_calls', 0) if hasattr(execution_report, 'llm_raw_output') else 0,
        }
        
        # Conta LLM calls dai tracking summary degli agenti
        llm_calls_per_agent = self._count_llm_calls(recon_report, planning_report, execution_report)
        
        # Raccogli token usage per agente
        def _get_tokens(report, agent_name):
            if hasattr(report, 'llm_analysis'):
                return report.llm_analysis.get('_total_tokens', 0)
            if hasattr(report, 'llm_raw_output'):
                return report.llm_raw_output.get('_total_tokens', 0)
            return 0
        
        tokens_per_agent = {
            'recon': _get_tokens(recon_report, 'recon'),
            'planning': _get_tokens(planning_report, 'planning'),
            'execution': _get_tokens(execution_report, 'execution'),
        }
        total_tokens = sum(tokens_per_agent.values())
        
        # 4. Ciclo adattivo (se abilitato)
        if self.config.enable_adaptive_loop:
            current_sim_results = execution_report.simulation_results
            current_security = execution_report.security_analysis
            current_attack_config = execution_report.attack_config
            
            for round_idx in range(self.config.max_adaptation_rounds):
                # Criterio di stop: Eve ha successo (information > 0 e non rilevata)
                eve_info_current = current_security.get('eve_information', 0.0)
                detected_current = current_security.get('protocol_aborted', False)
                if eve_info_current > 0 and not detected_current:
                    adaptation_history.append({
                        'round': round_idx,
                        'reason': 'eve_success',
                        'qber': current_security.get('final_qber', 0.0),
                    })
                    break
                
                # Recon sui risultati della simulazione
                print(f"\n[DEBUG adaptive loop round={round_idx}] PRIMA recon: qber_trajectory={qber_trajectory} | detected_trajectory={detected_trajectory}")
                recon_report = self._run_recon_on_results(
                    current_sim_results, current_security, current_attack_config, run_id
                )
                print(f"[DEBUG adaptive loop round={round_idx}] DOPO recon: raw_params count={len(recon_report.raw_parameters) if hasattr(recon_report, 'raw_parameters') else '?'} | qber={recon_report.raw_parameters.get('qber','?') if hasattr(recon_report, 'raw_parameters') else '?'}")
                self._update_calibration_from_recon(recon_report)
                
                # Planning aggiorna strategia
                planning_report = self._run_planning(recon_report, run_id)
                
                # Inietta il nome dello scenario per validazione (round adattivi)
                if hasattr(planning_report, 'raw_parameters') and isinstance(planning_report.raw_parameters, dict):
                    planning_report.raw_parameters["_scenario_name"] = scenario_name
                
                # Execution con strategia aggiornata
                execution_report = self._run_execution(planning_report, run_id)
                
                print(f"\n[DEBUG adaptive loop round={round_idx}] DOPO execution: qber={execution_report.security_analysis.get('final_qber')} | sifted_len={execution_report.simulation_results.get('sifted_len')} | attack_type={execution_report.attack_config.get('attack_type')} | attack_config={execution_report.attack_config}")
                
                # Aggiorna tracking
                attack_type = execution_report.attack_config.get('attack_type', 'unknown')
                if attack_type not in attack_types_tried:
                    attack_types_tried.append(attack_type)
                
                qber = execution_report.security_analysis.get('final_qber', 0.0)
                detected = execution_report.security_analysis.get('protocol_aborted', False)
                
                adaptation_history.append({
                    'round': round_idx,
                    'attack_type': attack_type,
                    'qber': qber,
                    'detected': detected,
                    'is_secure': execution_report.security_analysis.get('is_secure', False),
                })
                
                qber_trajectory.append(qber)
                detected_trajectory.append(detected)
                
                llm_calls_per_agent = self._count_llm_calls(recon_report, planning_report, execution_report)
                
                # Aggiorna token usage dopo round adattivo
                tokens_per_agent = {
                    'recon': _get_tokens(recon_report, 'recon'),
                    'planning': _get_tokens(planning_report, 'planning'),
                    'execution': _get_tokens(execution_report, 'execution'),
                }
                total_tokens = sum(tokens_per_agent.values())
                
                current_sim_results = execution_report.simulation_results
                current_security = execution_report.security_analysis
                current_attack_config = execution_report.attack_config
        
        # 5. Determina successo finale
        final_qber = qber_trajectory[-1] if qber_trajectory else 0.0
        final_detected = detected_trajectory[-1] if detected_trajectory else False
        final_eve_info = execution_report.security_analysis.get('eve_information', 0.0)
        final_secure_key = execution_report.security_analysis.get('secure_key_rate', 0.0)
        
        is_baseline = scenario_name == "clean_baseline"
        sifted = execution_report.simulation_results.get('sifted_len', 0)
        
        # Blinding attack: Eve controlla completamente la chiave quando non rilevata
        final_attack_cfg = execution_report.attack_config
        blinding_active = final_attack_cfg.get('blinding_attack_active', False) or final_attack_cfg.get('blinding', False)
        if blinding_active and not final_detected and sifted > 0:
            final_eve_info = float(sifted)
        
        success = (final_secure_key > 0) if is_baseline else (final_eve_info > 0 and not final_detected and sifted > 0)
        stealth = not final_detected
        
        print(f"\n{'='*70}")
        print(f"[DEBUG run_trial END] scenario={scenario_name} | trial={trial_idx}")
        print(f"[DEBUG run_trial END] final_qber={final_qber} | final_detected={final_detected} | final_eve_info={final_eve_info} | final_secure_key={final_secure_key}")
        print(f"[DEBUG run_trial END] sifted={sifted} | success={success} | stealth={stealth}")
        print(f"[DEBUG run_trial END] attack_types_tried={attack_types_tried} | qber_trajectory={qber_trajectory}")
        print(f"[DEBUG run_trial END] adaptation_history={adaptation_history}")
        
        
        # 6. Calcola wall time
        wall_time = time.time() - start_time
        
        # 7. Crea TrialResult
        result = TrialResult(
            run_id=run_id,
            seed=int(self.config.seed),
            scenario_name=scenario_name,
            attack_type=attack_types_tried[0] if attack_types_tried else "unknown",
            success=success,
            stealth=stealth,
            qber=final_qber,
            true_qber=self._get_true_qber(execution_report),
            stolen_key_bits=int(final_eve_info) if not final_detected else 0,
            key_compromised=int(final_eve_info) > 0 and not final_detected,
            detected=final_detected,
            sifted_key_length=execution_report.simulation_results.get('sifted_len', 0),
            secure_key_length=max(0, int(final_secure_key * self.config.raw_key_size)),
            eve_information=final_eve_info,
            agent_messages=len(adaptation_history) + 3,  # 3 per ciclo base
            handoffs_attempted=len(adaptation_history) + 2,
            handoffs_successful=len(adaptation_history) + 2,
            llm_calls=sum(llm_calls_per_agent.values()),
            wall_time_sec=wall_time,
            total_tokens=total_tokens,
            tokens_per_agent=tokens_per_agent,
            crashed=False,
            crash_reason="",
            extra_params={
                'adaptation_depth': len(adaptation_history),
                'attack_types_tried': attack_types_tried,
                'qber_trajectory': qber_trajectory,
                'detected_trajectory': detected_trajectory,
                'channel_config': self._channel_config_to_dict(channel_config),
                'llm_calls_per_agent': llm_calls_per_agent,
                'adaptation_history': adaptation_history,
                'is_adaptive': self.config.enable_adaptive_loop,
                'final_attack_config': execution_report.attack_config,
                'calibration_state': {
                    'updates': self.calibration_updates_count,
                    'current_means': dict(self.calibrated_means),
                },
            },
        )
        
        self.logger.log_trial(result)
        return result
    
    def _reset_calibration(self) -> None:
        """Resetta i mean di calibration ai valori di default per una nuova sessione di trial."""
        self.config.detector_efficiency_mean = 0.72
        self.config.dark_count_rate_mean = 0.0004
        self.config.channel_loss_mean = 0.18
        self.config.mean_photon_num_mean = 0.1
        self.config.distance_km_mean = 10.0
        self.calibration_updates_count = 0
        self.calibrated_means = {}
    
    def run_multiple_trials(
        self,
        scenario_names: Optional[List[str]] = None,
        trials_per_scenario: int = 5,
    ) -> List[TrialResult]:
        """
        Esegue multiple trial per diversi scenari.
        
        Args:
            scenario_names: Lista di nomi scenario. Se None, usa tutti SCENARIOS.
            trials_per_scenario: Numero di trial per scenario.
        
        Returns:
            Lista di TrialResult.
        """
        from evaluation.attack_scenarios import get_all_scenarios
        
        if scenario_names is None:
            scenario_names = list(get_all_scenarios().keys())
        
        all_results = []
        
        # Reset calibration prima di ogni scenario per evitare drift accumulato
        self._reset_calibration()
        
        for scenario_name in scenario_names:
            print(f"\n{'=' * 70}")
            print(f"Scenario: {scenario_name}")
            print(f"{'=' * 70}")
            print(f"[DEBUG run_multiple_trials] calibrated_means at start: detector_eff={self.config.detector_efficiency_mean:.4f} | dark_count={self.config.dark_count_rate_mean:.6f} | channel_loss={self.config.channel_loss_mean:.4f} | mean_photon={self.config.mean_photon_num_mean:.4f} | distance={self.config.distance_km_mean:.2f}")
            
            for trial_idx in range(trials_per_scenario):
                result = self.run_trial(scenario_name, trial_idx)
                all_results.append(result)
                
                status = "SUCCESS" if (not result.crashed and result.success) else \
                         "CRASH" if result.crashed else "FAIL"
                
                adaptation = result.extra_params.get('adaptation_depth', 0)
                attacks = result.extra_params.get('attack_types_tried', [])
                
                print(f"  [{scenario_name}] trial {trial_idx + 1}/{trials_per_scenario}: "
                      f"{status} | QBER={result.qber:.4f} | "
                      f"Adapt={adaptation} | Attacks={attacks} | "
                      f"Tempo={result.wall_time_sec:.2f}s")
                
                # Salvataggio intermedio dopo ogni trial
                self.logger.save_results_json()
            
            print()
        
        # Reset calibration tra scenari per evitare drift accumulato
        self._reset_calibration()
        
        # Salva risultati
        self.logger.save_results_json()
        self.logger.save_metrics_csv()
        
        summary = self.logger.get_summary()
        print(f"\n[AdaptiveTrialRunner] Valutazione completata: {len(all_results)} trial eseguite.")
        print(f"[AdaptiveTrialRunner] Success rate: {summary.get('success_rate', 0):.2%}")
        print(f"[AdaptiveTrialRunner] Crash rate: {summary.get('crash_rate', 0):.2%}")
        print(f"[AdaptiveTrialRunner] Risultati salvati in: {self.output_dir}")
        
        return all_results
    
    # ========================================================================
    # Private methods
    # ========================================================================
    
    def _generate_random_channel_config(
        self,
        scenario_name: str,
        scenario_params: Optional[Dict[str, Any]] = None,
    ) -> SimulationConfig:
        """
        Genera SimulationConfig con parametri del canale casuali.
        
        I parametri base vengono dagli scenario_params (o SCENARIOS),
        ma con jitter aggiunto per simulare dispositivi reali.
        """
        from evaluation.attack_scenarios import get_all_scenarios
        
        if scenario_params is None:
            all_scenarios = get_all_scenarios()
            scenario_params = all_scenarios[scenario_name].params.copy()
        
        channel_params = scenario_params.get('channel_params', {})
        
        # Genera parametri con jitter
        detector_eff = max(0.5, min(1.0,
            float(self._rng.normal(self.config.detector_efficiency_mean,
                                    self.config.detector_efficiency_std))))
        
        dark_count = max(0.0,
            float(self._rng.normal(self.config.dark_count_rate_mean,
                                   self.config.dark_count_rate_std)))
        
        ch_loss = max(0.05, min(0.50,
            float(self._rng.normal(self.config.channel_loss_mean,
                                   self.config.channel_loss_std))))
        
        mean_photon = max(0.05, min(0.5,
            float(self._rng.normal(self.config.mean_photon_num_mean,
                                   self.config.mean_photon_num_std))))
        
        distance = max(1.0, min(30.0,
            float(self._rng.normal(self.config.distance_km_mean,
                                   self.config.distance_km_std))))
        
        # Calcola amplitude_damping_gamma dalla distanza
        # 0.2 dB/km → trasmittanza = 10^(-0.2*d/10), gamma = 1 - trasmittanza
        transmission = 10 ** (-self.config.fiber_loss_db_per_km * distance / 10)
        amplitude_gamma = 1.0 - transmission

        # DEBUG: log configurazione canale generata
        print(f"\n[DEBUG channel_config] scenario={scenario_name}  | detector_eff={detector_eff:.4f} | dark_count={dark_count:.6f} | ch_loss={ch_loss:.4f} | mean_photon={mean_photon:.4f} | distance={distance:.2f} | amp_gamma={amplitude_gamma:.4f}")
        
        return SimulationConfig(
            raw_key_size=self.config.raw_key_size,
            num_iterations=self.config.num_iterations,
            
            # Eve attack params (dal scenario base)
            eve_attack_enabled=scenario_params.get('eve_attack_enabled', False),
            interception_rate=scenario_params.get('interception_rate', 0.0),
            eve_pns_enabled=scenario_params.get('eve_pns_enabled', False),
            eve_pns_block_ratio=scenario_params.get('eve_pns_block_ratio', 0.5),
            blinding_attack_active=scenario_params.get('blinding_attack_active', False),
            trojan_horse_prob=scenario_params.get('trojan_horse_prob', 0.0),
            qber_tamper_active=scenario_params.get('qber_tamper_active', False),
            qber_tamper_amount=scenario_params.get('qber_tamper_amount', 0.0),
            
            # Channel params con jitter
            depolarization_prob=channel_params.get('depolarization_prob', 0.0),
            amplitude_damping_gamma=amplitude_gamma,
            use_amplitude_damping=True,
            use_phase_damping=channel_params.get('use_phase_damping', False),
            phase_damping_lambda=channel_params.get('phase_damping_lambda', 0.05),
            
            # Source
            use_weak_laser=True,
            mean_photon_num=mean_photon,
            source_frequency_hz=1e6,
            
            # Detector
            detector_efficiency=detector_eff,
            use_dead_time=True,
            dead_time_us=0.01,
            use_thermal_noise=dark_count > 0,
            thermal_ratio=dark_count,
            
            # Channel loss
            use_channel_attenuation=True,
            attenuation_coeff=self.config.fiber_loss_db_per_km / 0.2,
            distance_km=distance,
            fiber_loss_db_per_km=self.config.fiber_loss_db_per_km,
            
            # Optical isolator
            optical_isolator_efficiency=channel_params.get('optical_isolator_efficiency', 0.95),
            
            # Security
            sharing_rate=self.config.sharing_rate,
            track_state_purities=False,
            research_mode=True,
        )
    
    def _channel_config_to_dict(self, config: SimulationConfig) -> Dict[str, Any]:
        """Converte SimulationConfig in dict per logging."""
        from dataclasses import fields as dc_fields
        return {
            f.name: getattr(config, f.name)
            for f in dc_fields(config)
        }
    
    def _update_calibration_from_recon(self, recon_report: Any) -> None:
        """
        Aggiorna i mean di AdaptiveConfig con i parametri reali
        osservati dal Recon Agent (Recon-Driven Calibration).
        
        Usa una media mobile esponenziale per stabilizzare il rumore:
            new_mean = alpha * observed + (1 - alpha) * old_mean
        con alpha = 0.3 (smoothing per evitare oscillazioni).
        """
        alpha = 0.3
        raw = recon_report.raw_parameters
        
        if "detector_efficiency" in raw:
            self.config.detector_efficiency_mean = alpha * raw["detector_efficiency"] + (1 - alpha) * self.config.detector_efficiency_mean
        if "dark_count_rate" in raw:
            self.config.dark_count_rate_mean = alpha * raw["dark_count_rate"] + (1 - alpha) * self.config.dark_count_rate_mean
        if "channel_loss" in raw:
            self.config.channel_loss_mean = alpha * raw["channel_loss"] + (1 - alpha) * self.config.channel_loss_mean
        if "mean_photon_num" in raw:
            self.config.mean_photon_num_mean = alpha * raw["mean_photon_num"] + (1 - alpha) * self.config.mean_photon_num_mean
        if "distance_km" in raw:
            self.config.distance_km_mean = alpha * raw["distance_km"] + (1 - alpha) * self.config.distance_km_mean
        
        self.calibration_updates_count += 1
        print(f"\n[DEBUG calibration_update] updates={self.calibration_updates_count} | detector_eff_mean={self.config.detector_efficiency_mean:.4f} | dark_count_mean={self.config.dark_count_rate_mean:.6f} | channel_loss_mean={self.config.channel_loss_mean:.4f} | mean_photon_mean={self.config.mean_photon_num_mean:.4f} | distance_mean={self.config.distance_km_mean:.2f}")
        
        self.calibrated_means = {
            "detector_efficiency_mean": self.config.detector_efficiency_mean,
            "channel_loss_mean": self.config.channel_loss_mean,
            "mean_photon_num_mean": self.config.mean_photon_num_mean,
            "distance_km_mean": self.config.distance_km_mean,
        }
    
    def _run_recon(self, channel_config: SimulationConfig, run_id: str) -> Any:
        """Esegue il Recon Agent con il canale generato."""
        print(f"\n[DEBUG _run_recon] PRIMA di from_simulation")
        
        channel_source = BB84ChannelSource.from_simulation(
            channel_config, run_id=run_id
        )
        
        raw_params = channel_source.get_raw_parameters()
        print(f"[DEBUG _run_recon] DA from_simulation: {len(raw_params)} campi | qber={raw_params.get('qber','?')} | sifted={raw_params.get('sifted_key_length','?')} | eve_present={raw_params.get('eve_present','?')} | eve_strategy={raw_params.get('eve_strategy','?')}")
        
        llm_client = LMStudioClient(
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
            temperature=0.2,
            max_tokens=self.config.llm_max_tokens,
            max_reasoning_tokens=4092,
            request_timeout_seconds=500,
        )
        
        recon_agent = ReconAgent(
            llm_client=llm_client,
            channel_source=channel_source,
            repository=None,  # No DB in adaptive mode
            qber_abort_threshold=self.config.qber_abort_threshold,
            max_llm_retries=self.config.max_llm_retries,
        )
        
        recon_report = recon_agent.run()
        
        # Aggiungi tracking LLM calls e token usage al report
        recon_report.llm_analysis['_llm_calls'] = recon_agent._llm_call_count
        usage = recon_agent.llm_client.last_usage
        if usage:
            recon_report.llm_analysis['_total_tokens'] = usage.get('total_tokens', 0)
        
        return recon_report
    
    def _run_planning(self, recon_report: Any, run_id: str) -> Any:
        """Esegue il Planning Agent."""
        llm_client = LMStudioClient(
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
            temperature=0.3,
            max_tokens=self.config.llm_max_tokens,
            max_reasoning_tokens=4092,
            request_timeout_seconds=500,
        )
        
        planning_agent = PlanningAgent(
            recon_repository=None,
            planning_repository=None,
            recon_report=recon_report,
            raw_parameters=recon_report.raw_parameters if hasattr(recon_report, 'raw_parameters') else {},
            max_llm_retries=self.config.max_llm_retries,
            llm_base_url=self.config.llm_base_url,
            llm_model_name=self.config.llm_model_name,
            llm_temperature=0.3,
            llm_max_tokens=self.config.llm_max_tokens,
            llm_max_reasoning_tokens=4092,
        )
        
        planning_report = planning_agent.run()
        
        planning_report.llm_raw_output['_llm_calls'] = planning_agent._llm_call_count
        usage = planning_agent.llm_client.last_usage
        if usage:
            planning_report.llm_raw_output['_total_tokens'] = usage.get('total_tokens', 0)
        
        return planning_report
    
    def _run_execution(self, planning_report: Any, run_id: str) -> Any:
        """Esegue l'Execution Agent con temperatura esplorativa."""
        execution_agent = ExecutionAgent(
            planning_repository=None,
            execution_repository=None,
            planning_report=planning_report,
            raw_parameters=planning_report.raw_parameters if hasattr(planning_report, 'raw_parameters') else {},
            max_llm_retries=self.config.max_llm_retries,
            llm_base_url=self.config.llm_base_url,
            llm_model_name=self.config.llm_model_name,
            llm_temperature=self.config.execution_temperature,
            llm_max_tokens=self.config.llm_max_tokens,
        )
        
        execution_report = execution_agent.run()
        
        execution_report.llm_raw_output['_llm_calls'] = execution_agent._llm_call_count
        usage = execution_agent.llm_client.last_usage
        if usage:
            execution_report.llm_raw_output['_total_tokens'] = usage.get('total_tokens', 0)
        
        return execution_report
    
    def _run_recon_on_results(
        self,
        sim_results: Dict[str, Any],
        security_analysis: Dict[str, Any],
        attack_config: Dict[str, Any],
        run_id: str,
    ) -> Any:
        """
        Esegue Recon sui risultati di una simulazione passata.
        
        Questo è il cuore del ciclo adattivo: il Recon Agent analizza
        l'impatto dell'attacco precedente e produce un report che
        alimenta il Planning Agent per la strategia successiva.
        """
        print(f"\n[DEBUG _run_recon_on_results] sim_results keys={list(sim_results.keys())} | qber_est={sim_results.get('qber_est')} | sifted_len={sim_results.get('sifted_len')} | attack_active={sim_results.get('attack_active')}")
        print(f"[DEBUG _run_recon_on_results] security_analysis keys={list(security_analysis.keys()) if security_analysis else 'None'}")
        print(f"[DEBUG _run_recon_on_results] attack_config keys={list(attack_config.keys())}")
        
        # Normalizza i risultati nel formato BB84ChannelSource
        config = SimulationConfig(
            raw_key_size=self.config.raw_key_size,
            num_iterations=self.config.num_iterations,
            **{k: v for k, v in attack_config.items() 
               if k in ('eve_attack_enabled', 'interception_rate', 'eve_pns_enabled',
                         'eve_pns_block_ratio', 'blinding_attack_active', 
                         'trojan_horse_prob', 'qber_tamper_active', 'qber_tamper_amount',
                         'depolarization_prob', 'amplitude_damping_gamma',
                         'use_amplitude_damping', 'use_phase_damping',
                         'use_weak_laser', 'mean_photon_num', 'source_frequency_hz',
                         'detector_efficiency', 'use_dead_time', 'dead_time_us',
                         'use_thermal_noise', 'thermal_ratio',
                         'use_channel_attenuation', 'attenuation_coeff', 'distance_km',
                         'fiber_loss_db_per_km', 'optical_isolator_efficiency',
                         'sharing_rate', 'track_state_purities', 'research_mode')}
        )
        
        channel_source = BB84ChannelSource.from_simulation_result(
            sim_result=sim_results,
            config=config,
            run_id=run_id,
            security_analysis=security_analysis,
        )
        
        llm_client = LMStudioClient(
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
            temperature=0.2,
            max_tokens=self.config.llm_max_tokens,
            max_reasoning_tokens=4092,
            request_timeout_seconds=500,
        )
        
        recon_agent = ReconAgent(
            llm_client=llm_client,
            channel_source=channel_source,
            repository=None,
            qber_abort_threshold=self.config.qber_abort_threshold,
            max_llm_retries=self.config.max_llm_retries,
        )
        
        recon_report = recon_agent.run()
        recon_report.llm_analysis['_llm_calls'] = recon_agent._llm_call_count
        usage = recon_agent.llm_client.last_usage
        if usage:
            recon_report.llm_analysis['_total_tokens'] = usage.get('total_tokens', 0)
        
        return recon_report
    
    def _count_llm_calls(
        self,
        recon_report: Any,
        planning_report: Any,
        execution_report: Any,
    ) -> Dict[str, int]:
        """Estrae il numero di chiamate LLM dai report."""
        return {
            'recon': recon_report.llm_analysis.get('_llm_calls', 0) if recon_report else 0,
            'planning': planning_report.llm_raw_output.get('_llm_calls', 0) if planning_report else 0,
            'execution': execution_report.llm_raw_output.get('_llm_calls', 0) if execution_report else 0,
        }
    
    def _get_true_qber(self, execution_report: Any) -> float:
        """Estrae il true QBER dai risultati della simulazione."""
        sim_results = execution_report.simulation_results
        true_qber = sim_results.get('true_qber_est', 0.0)
        if true_qber == 0.0 and 'true_qber_estimates' in sim_results:
            estimates = sim_results['true_qber_estimates']
            if estimates:
                non_nan = [q for q in estimates if q == q]  # filter NaN
                true_qber = sum(non_nan) / len(non_nan) if non_nan else 0.0
        return true_qber


# ============================================================================
# Funzione helper per esecuzione rapida
# ============================================================================

def run_adaptive_evaluation(
    trials_per_scenario: int = 5,
    seed: int = 42,
    scenario_names: Optional[List[str]] = None,
    output_dir: str = "adaptive_output",
    enable_adaptive_loop: bool = True,
    max_adaptation_rounds: int = 3,
    execution_temperature: float = 0.5,
    detector_efficiency_mean: float = 0.72,
    detector_efficiency_std: float = 0.05,
    dark_count_rate_mean: float = 0.0004,
    dark_count_rate_std: float = 0.00015,
    channel_loss_mean: float = 0.18,
    channel_loss_std: float = 0.04,
    mean_photon_num_mean: float = 0.1,
    mean_photon_num_std: float = 0.03,
    distance_km_mean: float = 10.0,
    distance_km_std: float = 3.0,
) -> List[TrialResult]:
    """Funzione helper per eseguire la valutazione esplorativa in una riga."""
    
    config = AdaptiveConfig(
        seed=seed,
        raw_key_size=5000,
        num_iterations=5,
        detector_efficiency_mean=detector_efficiency_mean,
        detector_efficiency_std=detector_efficiency_std,
        dark_count_rate_mean=dark_count_rate_mean,
        dark_count_rate_std=dark_count_rate_std,
        channel_loss_mean=channel_loss_mean,
        channel_loss_std=channel_loss_std,
        mean_photon_num_mean=mean_photon_num_mean,
        mean_photon_num_std=mean_photon_num_std,
        distance_km_mean=distance_km_mean,
        distance_km_std=distance_km_std,
        enable_adaptive_loop=enable_adaptive_loop,
        max_adaptation_rounds=max_adaptation_rounds,
        execution_temperature=execution_temperature,
        output_dir=output_dir,
        llm_max_tokens=32678,
        llm_max_reasoning_tokens=4092,
    )
    
    runner = AdaptiveTrialRunner(config)
    return runner.run_multiple_trials(
        scenario_names=scenario_names,
        trials_per_scenario=trials_per_scenario,
    )
