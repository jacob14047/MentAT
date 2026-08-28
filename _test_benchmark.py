"""Test script: simula l'esecuzione del benchmark notebook."""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# Setup path
PROJECT_ROOT = Path(r'D:\Unsloth\TESI_FINALE')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'evaluation') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'evaluation'))

from evaluation.analysis import (
    compute_rq1, compute_rq3, compute_rq5,
    compute_metrics, compute_scenario_comparison, compute_correlations,
    generate_report, load_results
)
from evaluation.plotting import (
    plot_success_rate, plot_qber_distribution,
    plot_agent_collaboration, plot_handoff_analysis,
    plot_robustness_analysis, plot_time_distribution
)
from evaluation.evaluation_runner import EvaluationRunner, EvaluationConfig
from evaluation.attack_scenarios import SCENARIOS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

print("=" * 60)
print("TEST BENCHMARK NOTEBOOK")
print("=" * 60)

# --- Sezione 1: Caricamento ---
print("\n[1] Caricamento dati...")
EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "output"
EVAL_RESULTS_JSON = EVAL_OUTPUT_DIR / "evaluation_results.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(EVAL_RESULTS_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(f"  Caricati {len(df)} trial da {EVAL_RESULTS_JSON}")
print(f"  Scenari: {df['scenario_name'].nunique()}")

# --- Sezione 2: Pulizia ---
print("\n[2] Pulizia e validazione...")
df = df.copy()
df['success'] = df['success'].astype(bool)
df['crashed'] = df['crashed'].astype(bool)
df['stealth'] = df.get('stealth', pd.Series(False, index=df.index)).astype(bool)
df['detected'] = df.get('detected', pd.Series(False, index=df.index)).astype(bool)

# Rimuovi duplicati
if 'run_id' in df.columns:
    n_dupes = df.duplicated(subset=['run_id'], keep='first').sum()
    df = df.drop_duplicates(subset=['run_id'], keep='first')
    if n_dupes > 0:
        print(f"  Rimossi {n_dupes} duplicati")

# Rimuovi QBER fuori [0,1]
if 'qber' in df.columns:
    qber_valid = df['qber'].between(0.0, 1.0)
    n_bad = (~qber_valid).sum()
    if n_bad > 0:
        print(f"  Rimossi {n_bad} QBER fuori [0,1]")
        df = df[qber_valid]

# Rimuovi wall_time <= 0
if 'wall_time_sec' in df.columns:
    time_valid = df['wall_time_sec'] > 0
    n_bad = (~time_valid).sum()
    if n_bad > 0:
        print(f"  Rimossi {n_bad} wall_time <= 0")
        df = df[time_valid]

# Colonne derivate
df['_qber'] = df.apply(
    lambda r: r.get('true_qber', r.get('qber', 0.0))
    if r['scenario_name'] != 'clean_baseline'
    else r.get('qber', 0.0), axis=1
)
df['_is_attack'] = df['scenario_name'] != 'clean_baseline'

print(f"  {len(df)} trial valide")

# Statistiche
print("\n  Statistiche:")
stats = df.groupby('scenario_name').agg(
    count=('run_id', 'count'),
    success_rate=('success', 'mean'),
    qber_mean=('_qber', 'mean'),
    wall_time_mean=('wall_time_sec', 'mean'),
    crash_rate=('crashed', 'mean'),
).round(4)
print(stats.to_string())

# --- Sezione 4: Metriche ---
print("\n[4] Calcolo metriche...")
rq1_metrics = compute_rq1(df)
rq3_metrics = compute_rq3(df)
rq5_metrics = compute_rq5(df)
full_metrics = compute_metrics(df)
scenario_comparison = compute_scenario_comparison(df)
correlations = compute_correlations(df)

print("\n  RQ1 (per scenario):")
for s, m in rq1_metrics.items():
    print(f"    {s}: success={m['success_rate']:.1%}, qber={m['qber_mean']:.4f}, stealth={m['stealth_rate']:.1%}")

print("\n  RQ3 (per scenario):")
for s, m in rq3_metrics.items():
    print(f"    {s}: msgs={m['avg_agent_messages']:.1f}, handoff={m['handoff_rate']:.1%}, time={m['avg_wall_time']:.2f}s")

print("\n  RQ5 (per scenario):")
for s, m in rq5_metrics.items():
    print(f"    {s}: crash={m['crash_rate']:.1%}, time_cv={m['time_cv']:.4f}")

if correlations:
    print("\n  Correlazioni:")
    for name, value in correlations.items():
        print(f"    {name}: {value:.4f}")

# --- Sezione 5: Figure ---
print("\n[5] Generazione figure...")
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

# Figure 1: Success Rate
fig, ax = plt.subplots(figsize=(12, 7))
stats1 = df.groupby('scenario_name').agg(
    success_rate=('success', 'mean'),
    count=('run_id', 'count'),
).reset_index().sort_values('success_rate', ascending=True)

scenario_colors = {
    'intercept_resend': '#FF6B6B', 'intercept_resend_stealth': '#FFA07A',
    'pns': '#4ECDC4', 'blinding': '#95E1D3', 'trojan_horse': '#F38183',
    'qber_tamper': '#AA96DA', 'mixed': '#FCBAD3', 'clean_baseline': '#A8D8B9',
}
colors = [scenario_colors.get(s, '#4A90D9') for s in stats1['scenario_name']]
stats1['stderr'] = np.sqrt(stats1['success_rate'] * (1 - stats1['success_rate']) / stats1['count'])

bars = ax.barh(stats1['scenario_name'], stats1['success_rate'] * 100,
               xerr=stats1['stderr'] * 100, capsize=4, color=colors,
               alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, stats1['success_rate']):
    ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
            f'{val:.1%}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Success Rate (%)')
ax.set_ylabel('Attack Scenario')
ax.set_title('RQ1 - Attack Effectiveness: Success Rate by Scenario', fontweight='bold')
ax.set_xlim(0, 105)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='x')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'rq1_success_rate.pdf', dpi=300)
fig.savefig(FIGURES_DIR / 'rq1_success_rate.png', dpi=150)
plt.close(fig)
print("  [OK] rq1_success_rate.pdf")

# Figure 2: QBER Distribution
fig, ax = plt.subplots(figsize=(14, 8))
valid = df[df['crashed'] == False].copy()
scenarios = sorted(valid['scenario_name'].unique())
color_palette = sns.color_palette("husl", len(scenarios))

bp = ax.boxplot(
    [valid[valid['scenario_name'] == s]['_qber'].values for s in scenarios],
    labels=[s[:15] + '...' if len(s) > 15 else s for s in scenarios],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(alpha=0.7, edgecolor='black'),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(color_palette[i % len(color_palette)])

for i, scenario in enumerate(scenarios):
    data = valid[valid['scenario_name'] == scenario]['_qber']
    jitter = np.random.normal(0, 0.03, len(data))
    ax.scatter(i + 1 + jitter, data, alpha=0.3,
               color=color_palette[i % len(color_palette)], s=20)

ax.set_xlabel('Attack Scenario')
ax.set_ylabel('True QBER')
ax.set_title('RQ1 - Attack Effectiveness: QBER Distribution by Scenario', fontweight='bold')
ax.axhline(y=0.11, color='red', linestyle='--', linewidth=2, label='Abort Threshold (0.11)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'rq1_qber_distribution.pdf', dpi=300)
fig.savefig(FIGURES_DIR / 'rq1_qber_distribution.png', dpi=150)
plt.close(fig)
print("  [OK] rq1_qber_distribution.pdf")

# Figure 3: Messages & Handoff
fig, ax = plt.subplots(figsize=(12, 7))
stats3 = df.groupby('scenario_name').agg(
    avg_messages=('agent_messages', 'mean'),
    handoffs_attempted=('handoffs_attempted', 'mean'),
    handoffs_successful=('handoffs_successful', 'mean'),
).reset_index().sort_values('avg_messages', ascending=True)

x = np.arange(len(stats3))
width = 0.25
ax.bar(x - width, stats3['avg_messages'], width, label='Avg Agent Messages',
       color='#4A90D9', alpha=0.85, edgecolor='black', linewidth=0.5)
ax.bar(x, stats3['handoffs_attempted'], width, label='Handoffs Attempted',
       color='#FFA07A', alpha=0.85, edgecolor='black', linewidth=0.5)
ax.bar(x + width, stats3['handoffs_successful'], width, label='Handoffs Successful',
       color='#4ECDC4', alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Attack Scenario')
ax.set_ylabel('Count')
ax.set_title('RQ3 - Multi-Agent Collaboration: Messages & Handoffs', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in stats3['scenario_name']],
                  rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'rq3_messages_handoff.pdf', dpi=300)
fig.savefig(FIGURES_DIR / 'rq3_messages_handoff.png', dpi=150)
plt.close(fig)
print("  [OK] rq3_messages_handoff.pdf")

# Figure 4: Execution Time
fig, ax = plt.subplots(figsize=(12, 7))
stats4 = df.groupby('scenario_name').agg(
    time_mean=('wall_time_sec', 'mean'),
    time_std=('wall_time_sec', 'std'),
    count=('run_id', 'count'),
).reset_index().sort_values('time_mean', ascending=True)

x = np.arange(len(stats4))
width = 0.4
colors4 = [scenario_colors.get(s, '#4A90D9') for s in stats4['scenario_name']]

bars = ax.bar(x, stats4['time_mean'], yerr=stats4['time_std'], capsize=4,
              color=colors4, alpha=0.85, edgecolor='black', linewidth=0.5)
for i, (bar, val) in enumerate(zip(bars, stats4['time_mean'])):
    ax.text(bar.get_x() + bar.get_width() / 2, val + stats4.iloc[i]['time_std'] + 0.001,
            f'{val:.3f}s', ha='center', fontsize=9)

ax.set_xlabel('Attack Scenario')
ax.set_ylabel('Avg Wall Time (seconds)')
ax.set_title('RQ5 - Robustness: Execution Time Distribution', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in stats4['scenario_name']],
                  rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'rq5_time_distribution.pdf', dpi=300)
fig.savefig(FIGURES_DIR / 'rq5_time_distribution.png', dpi=150)
plt.close(fig)
print("  [OK] rq5_time_distribution.pdf")

# --- Sezione 6: Benchmark Summary ---
print("\n[6] Export benchmark_summary.json...")

def make_serializable(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj

benchmark_summary = {
    "metadata": {
        "title": "BB84 AI Red Teaming Benchmark Summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_trials": int(len(df)),
        "total_scenarios": int(df['scenario_name'].nunique()),
        "scenarios": sorted(df['scenario_name'].unique().tolist()),
        "data_source": str(EVAL_RESULTS_JSON),
        "benchmark_version": "1.0",
    },
    "rq1_attack_effectiveness": rq1_metrics,
    "rq3_multi_agent_collaboration": rq3_metrics,
    "rq5_robustness_reproducibility": rq5_metrics,
    "scenario_comparison": scenario_comparison,
    "correlations": correlations,
    "data_summary": {
        "global_success_rate": float(df['success'].mean()),
        "global_crash_rate": float(df['crashed'].mean()),
        "global_qber_mean": float(df['_qber'].mean()),
        "global_qber_std": float(df['_qber'].std()),
        "global_wall_time_mean": float(df['wall_time_sec'].mean()),
        "global_wall_time_std": float(df['wall_time_sec'].std()),
        "per_scenario": {
            scenario: {
                "n_trials": int(len(group)),
                "success_rate": float(group['success'].mean()),
                "crash_rate": float(group['crashed'].mean()),
                "qber_mean": float(group['_qber'].mean()),
                "qber_std": float(group['_qber'].std()),
                "wall_time_mean": float(group['wall_time_sec'].mean()),
                "wall_time_std": float(group['wall_time_sec'].std()),
            }
            for scenario, group in df.groupby('scenario_name')
        },
    },
}

benchmark_summary = make_serializable(benchmark_summary)
summary_path = PROJECT_ROOT / "benchmark_summary.json"
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(benchmark_summary, f, indent=2, ensure_ascii=False, default=str)

print(f"  [OK] benchmark_summary.json salvato in {summary_path}")
print(f"  Trials: {benchmark_summary['metadata']['total_trials']}")
print(f"  Scenari: {benchmark_summary['metadata']['total_scenarios']}")
print(f"  Global success rate: {benchmark_summary['data_summary']['global_success_rate']:.1%}")
print(f"  Global crash rate: {benchmark_summary['data_summary']['global_crash_rate']:.1%}")
print(f"  Global QBER mean: {benchmark_summary['data_summary']['global_qber_mean']:.4f}")
print(f"  Global wall time mean: {benchmark_summary['data_summary']['global_wall_time_mean']:.3f}s")

# --- Riepilogo ---
print("\n" + "=" * 60)
print("RIEPILOGO OUTPUT")
print("=" * 60)
outputs = [
    ("benchmark_summary.json", PROJECT_ROOT / "benchmark_summary.json"),
    ("figures/rq1_success_rate.pdf", FIGURES_DIR / "rq1_success_rate.pdf"),
    ("figures/rq1_qber_distribution.pdf", FIGURES_DIR / "rq1_qber_distribution.pdf"),
    ("figures/rq3_messages_handoff.pdf", FIGURES_DIR / "rq3_messages_handoff.pdf"),
    ("figures/rq5_time_distribution.pdf", FIGURES_DIR / "rq5_time_distribution.pdf"),
]
for name, path in outputs:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    status = f"{size:,} bytes" if exists else "MISSING"
    print(f"  [{status:>12s}] {name}")
print("=" * 60)
print("[DONE] Benchmark test completato con successo!")
