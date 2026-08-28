"""
Analysis module: metriche e analisi per le domande di ricerca.

Funzioni principali:
- compute_rq1(): Attack Effectiveness metrics
- compute_rq3(): Multi-Agent Collaboration Quality metrics
- compute_rq5(): Robustness and Reproducibility metrics
- compute_metrics(): Metriche complete (combina RQ1 + RQ3 + RQ5)
- compute_scenario_comparison(): Confronto tra scenari
- compute_correlations(): Correlazioni chiave
- generate_report(): Report testuale per la tesi
- save_report(): Salva report in file
- load_results(): Carica risultati da JSON/CSV
"""
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from evaluation.trial_logger import TrialResult, TrialLogger


# ============================================================================
# RQ1 – Attack Effectiveness
# ============================================================================

def compute_rq1(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcola metriche per RQ1: Attack Effectiveness.
    
    Metriche per scenario:
    - success_rate: tasso di successo dell'attacco
    - stealth_rate: tasso di stealth (non rilevato)
    - qber_mean/std: QBER medio e deviazione standard
    - avg_stolen_bits: bits di chiave rubati in media
    - key_compromised_rate: tasso di compromissione della chiave
    - detected_rate: tasso di rilevamento (QBER > 0.11)
    """
    results = {}
    
    for scenario, group in df.groupby('scenario_name'):
        valid = group[group['crashed'] == False]
        total = len(group)
        is_baseline = scenario == 'clean_baseline'
        
        # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
        qber_col = 'true_qber' if not is_baseline else 'qber'
        valid = valid.copy()
        valid['_qber'] = valid[qber_col]
        
        results[scenario] = {
            'total_trials': int(total),
            'valid_trials': int(len(valid)),
            'success_rate': float(valid['success'].mean()) if len(valid) > 0 else 0.0,
            'stealth_rate': float(valid['stealth'].mean()) if len(valid) > 0 else 0.0,
            'qber_mean': float(valid['_qber'].mean()) if len(valid) > 0 else 0.0,
            'qber_std': float(valid['_qber'].std()) if len(valid) > 0 else 0.0,
            'qber_min': float(valid['_qber'].min()) if len(valid) > 0 else 0.0,
            'qber_max': float(valid['_qber'].max()) if len(valid) > 0 else 0.0,
            'qber_median': float(valid['_qber'].median()) if len(valid) > 0 else 0.0,
            'avg_stolen_bits': float(valid['stolen_key_bits'].mean()) if len(valid) > 0 else 0.0,
            'key_compromised_rate': float(valid['key_compromised'].mean()) if len(valid) > 0 else 0.0,
            'detected_rate': float(valid['detected'].mean()) if len(valid) > 0 else 0.0,
            'eve_info_mean': float(valid['eve_information'].mean()) if len(valid) > 0 else 0.0,
        }
    
    return results


# ============================================================================
# RQ3 – Multi-Agent Collaboration Quality
# ============================================================================

def compute_rq3(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcola metriche per RQ3: Multi-Agent Collaboration Quality.
    
    Metriche per scenario:
    - avg_agent_messages: numero medio di messaggi agent
    - handoff_rate: tasso di handoff riusciti
    - handoff_success_count: conteo handoff riusciti
    - avg_wall_time: tempo medio di esecuzione
    - time_std: deviazione standard del tempo
    """
    results = {}
    
    for scenario, group in df.groupby('scenario_name'):
        valid = group[group['crashed'] == False]
        
        results[scenario] = {
            'total_trials': int(len(group)),
            'valid_trials': int(len(valid)),
            'avg_agent_messages': float(valid['agent_messages'].mean()) if len(valid) > 0 else 0.0,
            'avg_handoffs_attempted': float(valid['handoffs_attempted'].mean()) if len(valid) > 0 else 0.0,
            'avg_handoffs_successful': float(valid['handoffs_successful'].mean()) if len(valid) > 0 else 0.0,
            'handoff_rate': float(
                valid['handoffs_successful'].mean() / max(valid['handoffs_attempted'].mean(), 0.001)
            ) if len(valid) > 0 else 0.0,
            'avg_wall_time': float(valid['wall_time_sec'].mean()) if len(valid) > 0 else 0.0,
            'time_std': float(valid['wall_time_sec'].std()) if len(valid) > 0 else 0.0,
            'time_cv': float(
                valid['wall_time_sec'].std() / max(valid['wall_time_sec'].mean(), 0.001)
            ) if len(valid) > 0 else 0.0,
            'avg_llm_calls': float(valid['llm_calls'].mean()) if len(valid) > 0 else 0.0,
        }
    
    return results


# ============================================================================
# RQ5 – Robustness and Reproducibility
# ============================================================================

def compute_rq5(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcola metriche per RQ5: Robustness and Reproducibility.
    
    Metriche per scenario:
    - crash_rate: tasso di crash
    - success_cv: coefficiente di variazione del successo
    - time_cv: coefficiente di variazione del tempo
    - qber_cv: coefficiente di variazione del QBER
    - avg_wall_time: tempo medio
    - time_std: deviazione standard del tempo
    """
    results = {}
    
    for scenario, group in df.groupby('scenario_name'):
        valid = group[group['crashed'] == False].copy()
        total = len(group)
        is_baseline = scenario == 'clean_baseline'
        
        # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
        qber_col = 'true_qber' if not is_baseline else 'qber'
        valid['_qber'] = valid[qber_col]
        
        results[scenario] = {
            'total_trials': int(total),
            'valid_trials': int(len(valid)),
            'crash_rate': float(group['crashed'].mean()) if total > 0 else 0.0,
            'success_cv': float(
                group['success'].std() / max(group['success'].mean(), 0.001)
            ) if total > 0 else 0.0,
            'time_cv': float(
                valid['wall_time_sec'].std() / max(valid['wall_time_sec'].mean(), 0.001)
            ) if len(valid) > 0 else 0.0,
            'qber_cv': float(
                valid['_qber'].std() / max(valid['_qber'].mean(), 0.001)
            ) if len(valid) > 0 else 0.0,
            'avg_wall_time': float(valid['wall_time_sec'].mean()) if len(valid) > 0 else 0.0,
            'time_std': float(valid['wall_time_sec'].std()) if len(valid) > 0 else 0.0,
            'success_mean': float(valid['success'].mean()) if len(valid) > 0 else 0.0,
            'success_std': float(valid['success'].std()) if len(valid) > 0 else 0.0,
        }
    
    return results


# ============================================================================
# Metriche complete
# ============================================================================

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Combina RQ1 + RQ3 + RQ5 in un unico dizionario."""
    return {
        'rq1_attack_effectiveness': compute_rq1(df),
        'rq3_multi_agent_collaboration': compute_rq3(df),
        'rq5_robustness_reproducibility': compute_rq5(df),
    }


# ============================================================================
# Confronto scenari
# ============================================================================

def compute_scenario_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Confronto diretto tra scenari per metriche chiave."""
    comparison = {}
    
    for scenario, group in df.groupby('scenario_name'):
        valid = group[group['crashed'] == False].copy()
        total = len(group)
        is_baseline = scenario == 'clean_baseline'
        
        # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
        qber_col = 'true_qber' if not is_baseline else 'qber'
        valid['_qber'] = valid[qber_col]
        
        comparison[scenario] = {
            'status': 'complete' if len(valid) > 0 else 'no_data',
            'total_trials': int(total),
            'valid_trials': int(len(valid)),
            'success_rate': float(valid['success'].mean()) if len(valid) > 0 else 0.0,
            'stealth_rate': float(valid['stealth'].mean()) if len(valid) > 0 else 0.0,
            'qber_mean': float(valid['_qber'].mean()) if len(valid) > 0 else 0.0,
            'qber_std': float(valid['_qber'].std()) if len(valid) > 0 else 0.0,
            'time_mean': float(valid['wall_time_sec'].mean()) if len(valid) > 0 else 0.0,
            'time_std': float(valid['wall_time_sec'].std()) if len(valid) > 0 else 0.0,
            'stealth_mean': float(valid['stealth'].mean()) if len(valid) > 0 else 0.0,
            'crash_rate': float(group['crashed'].mean()) if total > 0 else 0.0,
            'avg_stolen_bits': float(valid['stolen_key_bits'].mean()) if len(valid) > 0 else 0.0,
        }
    
    return comparison


# ============================================================================
# Correlazioni
# ============================================================================

def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """Calcola correlazioni chiave tra variabili."""
    valid = df[df['crashed'] == False].copy()
    
    if len(valid) < 3:
        return {}
    
    # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
    is_baseline = valid['scenario_name'].unique().tolist() == ['clean_baseline']
    qber_col = 'true_qber' if not is_baseline else 'qber'
    valid['_qber'] = valid[qber_col]
    
    correlations = {}
    
    # QBER vs successo
    if valid['_qber'].std() > 0 and valid['success'].std() > 0:
        correlations['qber_vs_success'] = float(valid['_qber'].corr(valid['success']))
    
    # Tempo vs successo
    if valid['wall_time_sec'].std() > 0 and valid['success'].std() > 0:
        correlations['time_vs_success'] = float(valid['wall_time_sec'].corr(valid['success']))
    
    # QBER vs bits rubati
    if valid['_qber'].std() > 0 and valid['stolen_key_bits'].std() > 0:
        correlations['qber_vs_stolen_bits'] = float(valid['_qber'].corr(valid['stolen_key_bits']))
    
    # Stealth vs successo
    if valid['stealth'].std() > 0 and valid['success'].std() > 0:
        correlations['stealth_vs_success'] = float(valid['stealth'].corr(valid['success']))
    
    # Bits rubati vs stealth
    if valid['stolen_key_bits'].std() > 0 and valid['stealth'].std() > 0:
        correlations['stolen_bits_vs_stealth'] = float(valid['stolen_key_bits'].corr(valid['stealth']))
    
    return correlations


# ============================================================================
# Report
# ============================================================================

def generate_report(metrics: Dict[str, Any], df: pd.DataFrame) -> str:
    """Genera un report testuale per la tesi."""
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORTO DI VALUTAZIONE BB84 AI RED TEAMING")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary generale
    total = len(df)
    successful = len(df[(df['crashed'] == False) & (df['success'] == True)])
    crashed = df['crashed'].sum()
    
    lines.append("SINTESI GENERALE")
    lines.append("-" * 40)
    lines.append(f"  Trial totali: {total}")
    lines.append(f"  Trial riuscite: {successful}")
    lines.append(f"  Trial fallite: {total - successful}")
    lines.append(f"  Crash: {int(crashed)}")
    lines.append(f"  Success rate globale: {successful / total:.2%}" if total > 0 else "  Success rate globale: N/A")
    lines.append("")
    
    # RQ1
    lines.append("RQ1 – ATTACK EFFECTIVENESS")
    lines.append("-" * 40)
    rq1 = metrics.get('rq1_attack_effectiveness', {})
    
    # Ordina per success rate
    sorted_rq1 = sorted(rq1.items(), key=lambda x: x[1].get('success_rate', 0), reverse=True)
    
    for scenario, data in sorted_rq1:
        is_baseline = scenario == 'clean_baseline'
        qber_label = "True QBER" if not is_baseline else "QBER"
        lines.append(f"  [{scenario}]")
        lines.append(f"    Success rate: {data.get('success_rate', 0):.2%}")
        lines.append(f"    Stealth rate: {data.get('stealth_rate', 0):.2%}")
        lines.append(f"    {qber_label} mean ± std: {data.get('qber_mean', 0):.4f} ± {data.get('qber_std', 0):.4f}")
        lines.append(f"    Avg stolen bits: {data.get('avg_stolen_bits', 0):.1f}")
        lines.append(f"    Detected rate: {data.get('detected_rate', 0):.2%}")
        lines.append("")
    
    # RQ3
    lines.append("RQ3 – MULTI-AGENT COLLABORATION QUALITY")
    lines.append("-" * 40)
    rq3 = metrics.get('rq3_multi_agent_collaboration', {})
    
    for scenario, data in rq3.items():
        lines.append(f"  [{scenario}]")
        lines.append(f"    Avg agent messages: {data.get('avg_agent_messages', 0):.1f}")
        lines.append(f"    Handoff rate: {data.get('handoff_rate', 0):.2%}")
        lines.append(f"    Avg wall time: {data.get('avg_wall_time', 0):.2f}s")
        lines.append("")
    
    # RQ5
    lines.append("RQ5 – ROBUSTNESS AND REPRODUCIBILITY")
    lines.append("-" * 40)
    rq5 = metrics.get('rq5_robustness_reproducibility', {})
    
    for scenario, data in rq5.items():
        lines.append(f"  [{scenario}]")
        lines.append(f"    Crash rate: {data.get('crash_rate', 0):.2%}")
        lines.append(f"    Time CV: {data.get('time_cv', 0):.4f}")
        lines.append(f"    QBER CV: {data.get('qber_cv', 0):.4f}")
        lines.append(f"    Avg wall time: {data.get('avg_wall_time', 0):.2f}s")
        lines.append("")
    
    # Correlazioni
    correlations = compute_correlations(df)
    if correlations:
        lines.append("CORRELAZIONI CHIAVE")
        lines.append("-" * 40)
        for name, value in correlations.items():
            strength = "strong" if abs(value) > 0.7 else ("moderate" if abs(value) > 0.4 else "weak")
            direction = "positive" if value > 0 else "negative"
            lines.append(f"  {name}: {value:.4f} ({strength} {direction})")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("FINE RAPPORTO")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_report(metrics: Dict[str, Any], df: pd.DataFrame, output_dir: str = "output") -> str:
    """Salva il report in un file .txt."""
    report = generate_report(metrics, df)
    
    report_path = Path(output_dir) / "evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return str(report_path)


# ============================================================================
# Caricamento risultati
# ============================================================================

def load_results(json_path: str = None, csv_path: str = None) -> pd.DataFrame:
    """Carica risultati da JSON o CSV."""
    if json_path and Path(json_path).exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    if csv_path and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    
    return pd.DataFrame()
