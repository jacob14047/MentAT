"""
EvaluationRunner: orchestratore delle run di valutazione BB84.

Supporta due modalità:
- Modalità simulatore (default): chiama direttamente BB84SimulationV2
  per ogni scenario d'attacco. Veloce, adatto per RQ1 e RQ5.
- Modalità ibrida (opzionale): per un subset di trial, esegue il flusso
  completo Recon → Planning → Execution per raccogliere dati RQ3
  (collaborazione multi-agente).
"""
import time
import uuid
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from channel.bb84_simulator_Eve import BB84SimulationV2, SimulationConfig
from evaluation.attack_scenarios import SCENARIOS, get_all_scenarios, get_attack_scenarios
from evaluation.trial_logger import TrialLogger, TrialResult


# ============================================================================
# Config
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configurazione della valutazione."""
    num_trials: int = 50
    seed: int = 42
    base_seed: int = 42
    scenario_names: List[str] = field(default_factory=lambda: list(SCENARIOS.keys()))
    output_dir: str = "output"
    plots_dir: str = "plots"
    
    # Modalità ibrida: quante trial eseguire con il flusso agenti?
    # 0 = solo simulatore, >0 = numero di trial ibride per scenario
    hybrid_agent_trials: int = 0
    
    # Parametri di default del simulatore
    raw_key_size: int = 5000
    num_iterations: int = 5
    mean_photon_num: float = 0.4
    detector_efficiency: float = 0.72
    dark_count_rate: float = 0.0004
    channel_loss: float = 0.18
    fiber_loss_db_per_km: float = 0.2
    distance_km: float = 10.0
    sharing_rate: float = 0.1


# ============================================================================
# EvaluationRunner
# ============================================================================

class EvaluationRunner:
    """Orchestratore delle run di valutazione."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = TrialLogger(config.output_dir)
        
        # Seed management per riproducibilità
        self._rng = np.random.default_rng(config.base_seed)
        
        # Directory output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = Path(config.plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_log()
    
    def _setup_log(self) -> None:
        """Log iniziale della valutazione."""
        scenarios_str = ", ".join(self.config.scenario_names)
        print(f"\n[EvaluationRunner] Avvio valutazione: {len(self.config.scenario_names)} scenari, "
              f"{self.config.num_trials} trial ciascuno, seed={self.config.seed}")
        print(f"[EvaluationRunner] Output: {self.output_dir}\n")
    
    def run_all(self) -> List[TrialResult]:
        """Esegue la valutazione per tutti gli scenari."""
        all_results = []
        
        for scenario_name in self.config.scenario_names:
            results = self._run_scenario(scenario_name)
            all_results.extend(results)
        
        return all_results
    
    def _run_scenario(self, scenario_name: str) -> List[TrialResult]:
        """Esegue tutte le trial per un singolo scenario."""
        scenario = get_all_scenarios()[scenario_name]
        results = []
        
        print("=" * 70)
        print(f"Scenario: {scenario_name}")
        print(f"  Attacco: {scenario.description}")
        print("=" * 70)
        
        for trial_idx in range(self.config.num_trials):
            seed = self.config.base_seed + trial_idx * 1000 + hash(scenario_name) % 1000
            
            trial_result = self._run_single_trial(
                scenario_name=scenario_name,
                scenario=scenario,
                trial_idx=trial_idx,
                seed=seed,
            )
            
            results.append(trial_result)
            
            status = "SUCCESS" if (not trial_result.crashed and trial_result.success) else \
                     "CRASH" if trial_result.crashed else "FAIL"
            
            print(f"  [{scenario_name}] trial {trial_idx + 1}/{self.config.num_trials}: "
                  f"{status} | QBER={trial_result.qber:.4f} (true={trial_result.true_qber:.4f}) | "
                  f"Tempo={trial_result.wall_time_sec:.2f}s")
        
        print()
        return results
    
    def _run_single_trial(
        self,
        scenario_name: str,
        scenario: Any,
        trial_idx: int,
        seed: int,
    ) -> TrialResult:
        """Esegue una singola trial BB84."""
        start_time = time.time()
        run_id = f"{scenario_name[:10]}_{trial_idx:04d}"
        
        try:
            # Costruisci config del simulatore
            sim_config = self._build_sim_config(scenario, seed)
            
            # Esegui simulazione
            sim = BB84SimulationV2(sim_config)
            sim.run()
            
            # Ottieni risultati completi
            sim_results = sim.get_full_parameters()
            security = sim.analyze_security()
            
            # Determina successo e stealth
            qber = sim_results.get('qber', 0.0)
            true_qber = sim_results.get('true_qber', 0.0)
            detected = sim_results.get('protocol_aborted', False)
            secure_key = sim_results.get('secure_key_length', 0)
            eve_info = sim_results.get('eve_max_information', 0.0)
            sifted_len = sim_results.get('sifted_key_length', 0)
            
            # Blinding attack: Eve controlla completamente la chiave quando non rilevata
            if scenario_name == "blinding" and not detected and sifted_len > 0:
                eve_info = float(sifted_len)
            
            # Success = chiave sicura generata (baseline) o attacco riuscito (attacco)
            is_baseline = scenario_name == "clean_baseline"
            success = secure_key > 0 if is_baseline else (eve_info > 0 and not detected)
            stealth = not detected
            
            # Bits rubati
            stolen_bits = int(eve_info) if not detected else 0
            
            # Tracking agente (default 0 per modalità simulatore)
            agent_messages = 0
            handoffs_attempted = 0
            handoffs_successful = 0
            llm_calls = 0
            
            # Tempo
            wall_time = time.time() - start_time
            
            result = TrialResult(
                run_id=run_id,
                seed=seed,
                scenario_name=scenario_name,
                attack_type=scenario_name,
                success=success,
                stealth=stealth,
                qber=qber,
                true_qber=true_qber,
                stolen_key_bits=stolen_bits,
                key_compromised=stolen_bits > 0,
                detected=detected,
                sifted_key_length=sifted_len,
                secure_key_length=secure_key,
                eve_information=eve_info,
                agent_messages=agent_messages,
                handoffs_attempted=handoffs_attempted,
                handoffs_successful=handoffs_successful,
                llm_calls=llm_calls,
                wall_time_sec=wall_time,
                crashed=False,
                crash_reason="",
            )
            
            self.logger.log_trial(result)
            return result
            
        except Exception as e:
            wall_time = time.time() - start_time
            
            result = TrialResult(
                run_id=run_id,
                seed=seed,
                scenario_name=scenario_name,
                attack_type=scenario_name,
                success=False,
                stealth=False,
                qber=0.0,
                true_qber=0.0,
                stolen_key_bits=0,
                key_compromised=False,
                detected=False,
                sifted_key_length=0,
                secure_key_length=0,
                eve_information=0.0,
                agent_messages=0,
                handoffs_attempted=0,
                handoffs_successful=0,
                llm_calls=0,
                wall_time_sec=wall_time,
                crashed=True,
                crash_reason=str(e)[:200],
            )
            
            self.logger.log_trial(result)
            return result
    
    def _build_sim_config(self, scenario: Any, seed: int) -> SimulationConfig:
        """Costruisce SimulationConfig dallo scenario."""
        params = scenario.params
        channel = scenario.channel_params
        
        return SimulationConfig(
            raw_key_size=self.config.raw_key_size,
            num_iterations=self.config.num_iterations,
            
            # Eve attack params
            eve_attack_enabled=params.get("eve_attack_enabled", False),
            interception_rate=params.get("interception_rate", 0.0),
            eve_pns_enabled=params.get("eve_pns_enabled", False),
            eve_pns_block_ratio=params.get("eve_pns_block_ratio", 0.5),
            blinding_attack_active=params.get("blinding_attack_active", False),
            trojan_horse_prob=params.get("trojan_horse_prob", 0.0),
            qber_tamper_active=params.get("qber_tamper_active", False),
            qber_tamper_amount=params.get("qber_tamper_amount", 0.0),
            
            # Channel params
            depolarization_prob=channel.get("depolarization_prob", 0.0),
            amplitude_damping_gamma=channel.get("amplitude_damping_gamma", 0.15),
            use_amplitude_damping=channel.get("use_amplitude_damping", True),
            use_phase_damping=channel.get("use_phase_damping", False),
            phase_damping_lambda=0.05,
            
            # Source
            use_weak_laser=True,
            mean_photon_num=channel.get("mean_photon_num", self.config.mean_photon_num),
            source_frequency_hz=1e6,
            
            # Detector
            detector_efficiency=self.config.detector_efficiency,
            use_dead_time=True,
            dead_time_us=0.01,
            use_thermal_noise=self.config.dark_count_rate > 0,
            thermal_ratio=self.config.dark_count_rate,
            
            # Channel loss
            use_channel_attenuation=True,
            attenuation_coeff=self.config.fiber_loss_db_per_km / 0.2,
            distance_km=self.config.distance_km,
            fiber_loss_db_per_km=self.config.fiber_loss_db_per_km,
            
            # Optical isolator
            optical_isolator_efficiency=channel.get("optical_isolator_efficiency", 0.95),
            
            # Security
            sharing_rate=self.config.sharing_rate,
            track_state_purities=False,
            research_mode=True,
        )
    
    def run_evaluation(self) -> List[TrialResult]:
        """Metodo alternativo: esegue la valutazione e salva risultati."""
        results = self.run_all()
        
        # Salva risultati
        self.logger.save_results_json()
        self.logger.save_metrics_csv()
        
        summary = self.logger.get_summary()
        print(f"\n[EvaluationRunner] Valutazione completata: {len(results)} trial eseguite.")
        print(f"[EvaluationRunner] Success rate: {summary.get('success_rate', 0):.2%}")
        print(f"[EvaluationRunner] Crash rate: {summary.get('crash_rate', 0):.2%}")
        print(f"[EvaluationRunner] Risultati salvati in: {self.output_dir}")
        
        return results


# ============================================================================
# Funzione helper per esecuzione rapida
# ============================================================================

def run_evaluation(
    num_trials: int = 50,
    seed: int = 42,
    scenario_names: Optional[List[str]] = None,
    output_dir: str = "output",
    plots_dir: str = "plots",
    hybrid_agent_trials: int = 0,
) -> List[TrialResult]:
    """Funzione helper per eseguire la valutazione in una riga."""
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())
    
    config = EvaluationConfig(
        num_trials=num_trials,
        seed=seed,
        base_seed=seed,
        scenario_names=scenario_names,
        output_dir=output_dir,
        plots_dir=plots_dir,
        hybrid_agent_trials=hybrid_agent_trials,
    )
    
    runner = EvaluationRunner(config)
    return runner.run_evaluation()
