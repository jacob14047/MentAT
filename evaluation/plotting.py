"""
Plotting module: generazione di grafici per la tesi BB84 AI Red Teaming.

Funzioni disponibili:
- plot_success_rate(): Bar chart del successo per scenario
- plot_qber_distribution(): Box plot della distribuzione QBER
- plot_qber_by_scenario_box(): Box plot QBER per scenario
- plot_stealth_vs_success(): Scatter stealth vs successo
- plot_agent_collaboration(): Bar chart messaggi agent
- plot_handoff_analysis(): Bar chart handoff
- plot_confidence_analysis(): Scatter tempo vs successo
- plot_robustness_analysis(): Bar chart crash rate
- plot_time_distribution(): Error bar tempo ± std
- plot_attack_comparison(): Bar chart bits rubati
- plot_correlation_matrix(): Heatmap correlazioni
- plot_scenario_summary(): Radar chart scenario summary
- generate_all_plots(): Genera tutti i grafici in una volta
"""
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# matplotlib backend non-interattivo per server/Windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import seaborn as sns

# Palette colori
COLORS = {
    'intercept_resend': '#FF6B6B',
    'intercept_resend_stealth': '#FFA07A',
    'pns': '#4ECDC4',
    'blinding': '#95E1D3',
    'trojan_horse': '#F38183',
    'qber_tamper': '#AA96DA',
    'mixed': '#FCBAD3',
    'clean_baseline': '#A8D8B9',
}

DEFAULT_COLORS = sns.color_palette("husl", 8)


# ============================================================================
# Utility
# ============================================================================

def _save_plot(fig: plt.Figure, filename: str, plots_dir: str = "plots") -> str:
    """Salva un grafico come PDF e PNG."""
    path = Path(plots_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    pdf_path = path / filename.replace('.png', '.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(path / filename, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return str(pdf_path)


def _get_scenario_colors(scenarios: List[str]) -> Dict[str, str]:
    """Mappa colori per scenari."""
    colors = {}
    for i, scenario in enumerate(scenarios):
        colors[scenario] = COLORS.get(scenario, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    return colors


# ============================================================================
# RQ1 – Attack Effectiveness Plots
# ============================================================================

def plot_success_rate(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Bar chart: Success rate per scenario.
    
    Mostra quale attacco ha il tasso di successo più alto.
    Error bars = deviazione standard.
    """
    summary = df.groupby('scenario_name').agg(
        success_rate=('success', 'mean'),
        count=('success', 'count'),
        std=('success', 'std'),
    ).reset_index()
    
    summary = summary.sort_values('success_rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = [_get_scenario_colors(summary['scenario_name'].tolist()).get(s, '#4A90D9') 
              for s in summary['scenario_name']]
    
    bars = ax.barh(
        summary['scenario_name'][::-1],
        summary['success_rate'][::-1] * 100,
        xerr=summary['std'][::-1] * 100,
        capsize=5,
        color=colors[::-1],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel('Success Rate (%)', fontsize=12)
    ax.set_ylabel('Attack Scenario', fontsize=12)
    ax.set_title('RQ1 – Attack Effectiveness: Success Rate by Scenario', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
    
    # Aggiungi valori sulle barre
    for bar, val in zip(bars, summary['success_rate'][::-1]):
        ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    
    return _save_plot(fig, 'rq1_success_rate.pdf', plots_dir)


def plot_qber_distribution(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Box plot: Distribuzione QBER per scenario.
    
    Mostra la variabilità del QBER tra trial.
    """
    valid = df[df['crashed'] == False].copy()
    
    # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
    is_baseline_only = valid['scenario_name'].unique().tolist() == ['clean_baseline']
    qber_col = 'qber' if is_baseline_only else 'true_qber'
    valid['_qber'] = valid[qber_col]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = sorted(valid['scenario_name'].unique())
    data_by_scenario = [valid[valid['scenario_name'] == s]['_qber'].values for s in scenarios]
    
    bp = ax.boxplot(
        data_by_scenario,
        labels=[s[:15] + '...' if len(s) > 15 else s for s in scenarios],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(alpha=0.7, edgecolor='black'),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    
    qber_label = 'True QBER' if not is_baseline_only else 'QBER'
    ax.set_xlabel('Attack Scenario', fontsize=12)
    ax.set_ylabel(qber_label, fontsize=12)
    ax.set_title('RQ1 – Attack Effectiveness: QBER Distribution by Scenario', fontsize=14, fontweight='bold')
    ax.axhline(y=0.11, color='red', linestyle='--', linewidth=2, label='Abort Threshold (0.11)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _save_plot(fig, 'rq1_qber_distribution.pdf', plots_dir)


def plot_qber_by_scenario_box(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Box plot alternativo: QBER per scenario con jitter dei punti.
    """
    valid = df[df['crashed'] == False].copy()
    
    # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
    is_baseline_only = valid['scenario_name'].unique().tolist() == ['clean_baseline']
    qber_col = 'qber' if is_baseline_only else 'true_qber'
    valid['_qber'] = valid[qber_col]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = sorted(valid['scenario_name'].unique())
    
    for i, scenario in enumerate(scenarios):
        data = valid[valid['scenario_name'] == scenario]['_qber']
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(
            [i + 1 + jitter for _ in range(len(data))],
            data,
            alpha=0.4,
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            s=30,
            edgecolors='black',
            linewidth=0.3,
        )
    
    qber_label = 'True QBER' if not is_baseline_only else 'QBER'
    ax.set_xlabel('Attack Scenario', fontsize=12)
    ax.set_ylabel(qber_label, fontsize=12)
    ax.set_title('RQ1 – QBER per Scenario con Dati Individuali', fontsize=14, fontweight='bold')
    ax.axhline(y=0.11, color='red', linestyle='--', linewidth=2, label='Abort Threshold')
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels([s[:12] + '...' if len(s) > 12 else s for s in scenarios], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _save_plot(fig, 'rq1_qber_scatter.pdf', plots_dir)


def plot_stealth_vs_success(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Scatter plot: Stealth vs Success trade-off.
    
    Ogni punto è una trial. Mostra se attacchi più efficaci sono anche più rilevati.
    """
    valid = df[df['crashed'] == False].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scenarios = sorted(valid['scenario_name'].unique())
    
    for i, scenario in enumerate(scenarios):
        data = valid[valid['scenario_name'] == scenario]
        ax.scatter(
            data['stealth'],
            data['success'],
            alpha=0.5,
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            s=50,
            label=scenario[:20],
            edgecolors='black',
            linewidth=0.3,
        )
    
    ax.set_xlabel('Stealth Rate (Eve non rilevata)', fontsize=12)
    ax.set_ylabel('Success Rate (Attacco riuscito)', fontsize=12)
    ax.set_title('RQ1 – Stealth vs Success Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return _save_plot(fig, 'rq1_stealth_vs_success.pdf', plots_dir)


# ============================================================================
# RQ3 – Multi-Agent Collaboration Plots
# ============================================================================

def plot_agent_collaboration(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Bar chart: Numero medio di messaggi agent per scenario.
    
    Mostra la complessità della collaborazione multi-agente.
    """
    summary = df.groupby('scenario_name').agg(
        avg_messages=('agent_messages', 'mean'),
        avg_handoffs=('handoffs_successful', 'mean'),
        avg_llm=('llm_calls', 'mean'),
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(summary))
    width = 0.35
    
    bars1 = ax1.bar(
        x - width/2,
        summary['avg_messages'],
        width,
        label='Avg Agent Messages',
        color='#4A90D9',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width/2,
        summary['avg_llm'],
        width,
        label='Avg LLM Calls',
        color='#FF6B6B',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax1.set_xlabel('Attack Scenario', fontsize=12)
    ax1.set_ylabel('Avg Agent Messages', fontsize=12, color='#4A90D9')
    ax2.set_ylabel('Avg LLM Calls', fontsize=12, color='#FF6B6B')
    ax1.set_title('RQ3 – Multi-Agent Collaboration: Messages per Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in summary['scenario_name']], 
                        rotation=45, ha='right')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    
    return _save_plot(fig, 'rq3_agent_collaboration.pdf', plots_dir)


def plot_handoff_analysis(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Bar chart: Tasso di handoff riusciti per scenario.
    """
    summary = df.groupby('scenario_name').agg(
        handoffs_attempted=('handoffs_attempted', 'mean'),
        handoffs_successful=('handoffs_successful', 'mean'),
    ).reset_index()
    
    summary['handoff_rate'] = summary['handoffs_successful'] / summary['handoffs_attempted'].clip(lower=0.001)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(summary))
    width = 0.35
    
    bars1 = ax.bar(
        x - width/2,
        summary['handoffs_attempted'],
        width,
        label='Attempted',
        color='#FFA07A',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    bars2 = ax.bar(
        x + width/2,
        summary['handoffs_successful'],
        width,
        label='Successful',
        color='#4ECDC4',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel('Attack Scenario', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('RQ3 – Handoff Analysis: Attempted vs Successful', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in summary['scenario_name']], 
                        rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _save_plot(fig, 'rq3_handoff_analysis.pdf', plots_dir)


def plot_confidence_analysis(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Scatter: Tempo di esecuzione vs successo.
    
    Mostra se trial più lunghe hanno successo maggiore/minore.
    """
    valid = df[df['crashed'] == False].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scenarios = sorted(valid['scenario_name'].unique())
    
    for i, scenario in enumerate(scenarios):
        data = valid[valid['scenario_name'] == scenario]
        ax.scatter(
            data['wall_time_sec'],
            data['success'],
            alpha=0.5,
            color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            s=50,
            label=scenario[:20],
            edgecolors='black',
            linewidth=0.3,
        )
    
    ax.set_xlabel('Wall Time (seconds)', fontsize=12)
    ax.set_ylabel('Success (0/1)', fontsize=12)
    ax.set_title('RQ3 – Execution Time vs Success', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return _save_plot(fig, 'rq3_confidence_analysis.pdf', plots_dir)


# ============================================================================
# RQ5 – Robustness & Reproducibility Plots
# ============================================================================

def plot_robustness_analysis(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Bar chart: Crash rate per scenario.
    
    Mostra quale scenario destabilizza di più il sistema.
    """
    summary = df.groupby('scenario_name').agg(
        crash_rate=('crashed', 'mean'),
        total=('crashed', 'count'),
    ).reset_index()
    
    summary = summary.sort_values('crash_rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B' if r > 0.1 else '#4ECDC4' for r in summary['crash_rate']]
    
    bars = ax.barh(
        summary['scenario_name'][::-1],
        summary['crash_rate'][::-1] * 100,
        color=colors[::-1],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel('Crash Rate (%)', fontsize=12)
    ax.set_ylabel('Attack Scenario', fontsize=12)
    ax.set_title('RQ5 – Robustness: Crash Rate by Scenario', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    for bar, val in zip(bars, summary['crash_rate'][::-1]):
        ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    
    return _save_plot(fig, 'rq5_robustness.pdf', plots_dir)


def plot_time_distribution(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Error bar chart: Tempo medio ± std per scenario.
    
    Mostra la variabilità temporale (riproducibilità).
    """
    summary = df.groupby('scenario_name').agg(
        time_mean=('wall_time_sec', 'mean'),
        time_std=('wall_time_sec', 'std'),
    ).reset_index()
    
    summary = summary.sort_values('time_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(summary))
    width = 0.4
    
    bars = ax.bar(
        x,
        summary['time_mean'],
        yerr=summary['time_std'],
        capsize=5,
        color='#AA96DA',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel('Attack Scenario', fontsize=12)
    ax.set_ylabel('Avg Wall Time (seconds)', fontsize=12)
    ax.set_title('RQ5 – Reproducibility: Execution Time Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in summary['scenario_name']], 
                        rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _save_plot(fig, 'rq5_time_distribution.pdf', plots_dir)


# ============================================================================
# Cross-cutting plots
# ============================================================================

def plot_attack_comparison(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Bar chart: Bits di chiave rubati per scenario.
    """
    valid = df[df['crashed'] == False].copy()
    
    # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
    is_baseline_only = valid['scenario_name'].unique().tolist() == ['clean_baseline']
    qber_col = 'qber' if is_baseline_only else 'true_qber'
    
    summary = valid.groupby('scenario_name').agg(
        avg_stolen=('stolen_key_bits', 'mean'),
        qber_mean=(qber_col, 'mean'),
    ).reset_index()
    
    summary = summary.sort_values('avg_stolen', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.barh(
        summary['scenario_name'][::-1],
        summary['avg_stolen'][::-1],
        color='#F38183',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
    )
    
    ax.set_xlabel('Avg Stolen Key Bits', fontsize=12)
    ax.set_ylabel('Attack Scenario', fontsize=12)
    ax.set_title('RQ1 – Key Compromise: Stolen Bits by Scenario', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, summary['avg_stolen'][::-1]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
    
    fig.tight_layout()
    return _save_plot(fig, 'rq1_attack_comparison.pdf', plots_dir)


def plot_correlation_matrix(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Heatmap: Correlazioni tra variabili.
    """
    valid = df[df['crashed'] == False].copy()
    
    # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
    is_baseline_only = valid['scenario_name'].unique().tolist() == ['clean_baseline']
    qber_col = 'qber' if is_baseline_only else 'true_qber'
    
    numeric_cols = ['success', 'stealth', qber_col, 'stolen_key_bits', 
                    'wall_time_sec', 'agent_messages', 'handoffs_successful',
                    'detected', 'crashed']
    available = [c for c in numeric_cols if c in valid.columns]
    
    corr = valid[available].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    
    ax.set_title('RQ – Correlation Matrix', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    return _save_plot(fig, 'cross_correlation_matrix.pdf', plots_dir)


def plot_scenario_summary(df: pd.DataFrame, plots_dir: str = "plots") -> str:
    """
    Radar chart: Summary composito per scenario.
    """
    scenarios = sorted(df['scenario_name'].unique())
    
    if len(scenarios) < 2:
        return "plot_scenario_summary.pdf"
    
    # Normalizza metriche per ogni scenario
    metrics = {}
    for scenario in scenarios:
        data = df[df['scenario_name'] == scenario].copy()
        valid = data[data['crashed'] == False]
        is_baseline = scenario == 'clean_baseline'
        
        # Per scenari di attacco usa true_qber (QBER reale introdotto da Eve)
        qber_col = 'true_qber' if not is_baseline else 'qber'
        valid['_qber'] = valid[qber_col]
        
        metrics[scenario] = {
            'success_rate': valid['success'].mean() if len(valid) > 0 else 0,
            'stealth_rate': valid['stealth'].mean() if len(valid) > 0 else 0,
            'qber_norm': valid['_qber'].mean() / 0.11 if len(valid) > 0 else 0,  # normalizzato
            'efficiency': 1.0 / (valid['wall_time_sec'].mean() + 0.001) if len(valid) > 0 else 0,
        }
    
    # Normalizza a [0, 1]
    keys = list(metrics[scenarios[0]].keys())
    for key in keys:
        vals = [metrics[s][key] for s in scenarios]
        min_v, max_v = min(vals), max(vals)
        if max_v > min_v:
            for s in scenarios:
                metrics[s][key] = (metrics[s][key] - min_v) / (max_v - min_v)
        else:
            for s in scenarios:
                metrics[s][key] = 0.5
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, scenario in enumerate(scenarios):
        values = [metrics[scenario][k] for k in keys]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=scenario[:20], 
                color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.fill(angles, values, alpha=0.15, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace('_', '\n') for k in keys], fontsize=10)
    ax.set_title('RQ – Scenario Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    fig.tight_layout()
    return _save_plot(fig, 'cross_scenario_summary.pdf', plots_dir)


# ============================================================================
# Genera tutti i grafici
# ============================================================================

def generate_all_plots(df: pd.DataFrame, plots_dir: str = "plots") -> Dict[str, str]:
    """Genera tutti i grafici in una volta."""
    if df.empty:
        return {}
    
    plots = {}
    
    # RQ1
    plots['rq1_success_rate'] = plot_success_rate(df, plots_dir)
    plots['rq1_qber_distribution'] = plot_qber_distribution(df, plots_dir)
    plots['rq1_qber_scatter'] = plot_qber_by_scenario_box(df, plots_dir)
    plots['rq1_stealth_vs_success'] = plot_stealth_vs_success(df, plots_dir)
    plots['rq1_attack_comparison'] = plot_attack_comparison(df, plots_dir)
    
    # RQ3
    plots['rq3_agent_collaboration'] = plot_agent_collaboration(df, plots_dir)
    plots['rq3_handoff_analysis'] = plot_handoff_analysis(df, plots_dir)
    plots['rq3_confidence_analysis'] = plot_confidence_analysis(df, plots_dir)
    
    # RQ5
    plots['rq5_robustness'] = plot_robustness_analysis(df, plots_dir)
    plots['rq5_time_distribution'] = plot_time_distribution(df, plots_dir)
    
    # Cross-cutting
    plots['cross_correlation_matrix'] = plot_correlation_matrix(df, plots_dir)
    plots['cross_scenario_summary'] = plot_scenario_summary(df, plots_dir)
    
    return plots
