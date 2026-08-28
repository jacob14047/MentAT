"""
TrialLogger: registra i risultati di una singola trial BB84.

Raccoglie dati per tutte e tre le RQ:
- RQ1: success, stealth, qber, stolen_key_bits
- RQ3: agent_messages, handoffs_attempted, handoffs_successful
- RQ5: wall_time_sec, crashed, crash_reason
"""
import json
import csv
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class TrialResult:
    """Risultato di una singola trial BB84."""
    # Identificazione
    run_id: str = ""
    seed: int = 0
    scenario_name: str = ""
    attack_type: str = ""
    
    # RQ1 – Attack Effectiveness
    success: bool = False
    stealth: bool = False
    qber: float = 0.0
    true_qber: float = 0.0
    stolen_key_bits: int = 0
    key_compromised: bool = False
    detected: bool = False
    sifted_key_length: int = 0
    secure_key_length: int = 0
    eve_information: float = 0.0
    
    # RQ3 – Multi-Agent Collaboration
    agent_messages: int = 0
    handoffs_attempted: int = 0
    handoffs_successful: int = 0
    llm_calls: int = 0
    
    # RQ5 – Robustness
    wall_time_sec: float = 0.0
    crashed: bool = False
    crash_reason: str = ""
    
    # Token usage
    total_tokens: int = 0
    tokens_per_agent: Dict[str, int] = field(default_factory=dict)
    
    # Metadata aggiuntivo
    extra_params: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrialLogger:
    """Logger che accumula risultati e salva in JSON/CSV."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[TrialResult] = []
        self.results_path = self.output_dir / "evaluation_results.json"
        self.metrics_path = self.output_dir / "evaluation_results.csv"
    
    def log_trial(self, result: TrialResult) -> None:
        """Registra un risultato di trial."""
        self.results.append(result)
    
    def log_trial_dict(self, data: Dict[str, Any]) -> TrialResult:
        """Registra un risultato da dict e lo restituisce."""
        result = TrialResult.from_dict(data)
        self.log_trial(result)
        return result
    
    def save_results_json(self) -> str:
        """Salva tutti i risultati in JSON."""
        data = [r.to_dict() for r in self.results]
        with open(self.results_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return str(self.results_path)
    
    def save_metrics_csv(self) -> str:
        """Salva i dati in CSV."""
        if not self.results:
            return str(self.metrics_path)
        
        # Campi principali
        fieldnames = [
            'run_id', 'seed', 'scenario_name', 'attack_type',
            'success', 'stealth', 'qber', 'true_qber',
            'stolen_key_bits', 'key_compromised', 'detected',
            'sifted_key_length', 'secure_key_length', 'eve_information',
            'agent_messages', 'handoffs_attempted', 'handoffs_successful', 'llm_calls',
            'wall_time_sec', 'crashed', 'crash_reason',
            'total_tokens', 'tokens_per_agent',
            'timestamp',
        ]
        
        with open(self.metrics_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for result in self.results:
                row = result.to_dict()
                # Serializza extra_params per CSV
                if 'extra_params' in row and isinstance(row['extra_params'], dict):
                    row['extra_params'] = json.dumps(row['extra_params'], default=str, ensure_ascii=False)
                writer.writerow(row)
        
        return str(self.metrics_path)
    
    def save_adaptive_metrics_csv(self) -> str:
        """Salva metriche specifiche per trial adattive in CSV separato."""
        if not self.results:
            return str(self.metrics_path)
        
        adaptive_results = [r for r in self.results if r.extra_params.get('adaptation_depth', 0) > 0]
        if not adaptive_results:
            return str(self.metrics_path)
        
        fieldnames = [
            'run_id', 'scenario_name',
            'success', 'stealth', 'qber', 'true_qber',
            'stolen_key_bits', 'detected',
            'sifted_key_length', 'secure_key_length',
            'llm_calls', 'agent_messages',
            'wall_time_sec',
            'adaptation_depth', 'attack_types_tried',
            'qber_trajectory', 'channel_config',
            'llm_calls_per_agent', 'is_adaptive',
        ]
        
        with open(self.metrics_path.replace('.csv', '_adaptive.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for result in adaptive_results:
                row = result.to_dict()
                # Serializza campi complessi
                for key in ['extra_params', 'qber_trajectory', 'channel_config', 'llm_calls_per_agent']:
                    if key in row and isinstance(row[key], (dict, list)):
                        row[key] = json.dumps(row[key], default=str, ensure_ascii=False)
                writer.writerow(row)
        
        return str(self.metrics_path).replace('.csv', '_adaptive.csv')
    
    def get_summary(self) -> Dict[str, Any]:
        """Restituisce un riepilogo statistico dei risultati."""
        if not self.results:
            return {}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if not r.crashed and r.success)
        crashed = sum(1 for r in self.results if r.crashed)
        
        # Calcola avg_qber separato per baseline e attacchi
        baseline_qber = [r.qber for r in self.results if not r.crashed and r.scenario_name == 'clean_baseline']
        attack_true_qber = [r.true_qber for r in self.results if not r.crashed and r.scenario_name != 'clean_baseline']
        
        avg_baseline_qber = sum(baseline_qber) / len(baseline_qber) if baseline_qber else 0.0
        avg_attack_true_qber = sum(attack_true_qber) / len(attack_true_qber) if attack_true_qber else 0.0
        
        return {
            'total_trials': total,
            'successful_trials': successful,
            'crashed_trials': crashed,
            'success_rate': successful / total if total > 0 else 0,
            'crash_rate': crashed / total if total > 0 else 0,
            'avg_qber_baseline': avg_baseline_qber,
            'avg_true_qber_attacks': avg_attack_true_qber,
            'avg_wall_time': sum(r.wall_time_sec for r in self.results) / total,
        }
    
    def get_results_by_scenario(self, scenario_name: str) -> List[TrialResult]:
        """Filtra risultati per scenario."""
        return [r for r in self.results if r.scenario_name == scenario_name]
    
    def get_valid_results(self) -> List[TrialResult]:
        """Restituisce solo risultati non crashati."""
        return [r for r in self.results if not r.crashed]
    
    def load_results_json(self) -> List[TrialResult]:
        """Carica risultati da file JSON."""
        if not self.results_path.exists():
            return []
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results = [TrialResult.from_dict(r) for r in data]
        return self.results
    
    def get_adaptive_summary(self) -> Dict[str, Any]:
        """Restituisce statistiche specifiche per trial adattive."""
        adaptive_results = [r for r in self.results if r.extra_params.get('adaptation_depth', 0) > 0]
        if not adaptive_results:
            return {}
        
        total = len(adaptive_results)
        successful = sum(1 for r in adaptive_results if not r.crashed and r.success)
        
        # Stats adaptation depth
        depths = [r.extra_params.get('adaptation_depth', 0) for r in adaptive_results]
        avg_depth = sum(depths) / total if total > 0 else 0
        
        # Stats QBER trajectory
        qbers = [r.qber for r in adaptive_results if not r.crashed]
        avg_qber = sum(qbers) / len(qbers) if qbers else 0.0
        
        # Attack diversity
        attack_sets = [set(r.extra_params.get('attack_types_tried', [])) for r in adaptive_results]
        avg_attacks = sum(len(s) for s in attack_sets) / total if total > 0 else 0
        
        # LLM calls
        llm_calls = [r.llm_calls for r in adaptive_results if not r.crashed]
        avg_llm_calls = sum(llm_calls) / len(llm_calls) if llm_calls else 0
        
        return {
            'total_adaptive_trials': total,
            'successful_adaptive_trials': successful,
            'adaptive_success_rate': successful / total if total > 0 else 0,
            'avg_adaptation_depth': avg_depth,
            'max_adaptation_depth': max(depths) if depths else 0,
            'min_adaptation_depth': min(depths) if depths else 0,
            'avg_qber': avg_qber,
            'avg_attack_types_per_trial': avg_attacks,
            'avg_llm_calls': avg_llm_calls,
        }
