import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurazione stile
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150


def load_results(file_path):
    """
    Carica i risultati da un file .json.
    Restituisce un DataFrame pandas.
    """
    data = []
    with open(file_path, "r") as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                raise ValueError("Formato non supportato")
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    df = pd.json_normalize(data)
    return df


def confidence_interval(data, confidence=0.95):
    """Calcola l'intervallo di confidenza al (confidence)*100% con metodo Wilson."""
    n = len(data)
    if n == 0:
        return (0.0, 1.0)
    p = data.mean()
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return (max(0, center - margin), min(1, center + margin))


# ============================================================================
# Caricamento dati
# ============================================================================

df = load_results("adaptive_output/evaluation_results.json")

# Rimossa riga bug: df.get() non funziona su DataFrame.
# La colonna key_compromised è già presente nei dati (campo top-level di TrialResult).

# Stampa riepilogo
print("Numero di run caricate:", len(df))
print("Colonne disponibili:\n", df.columns.tolist())

# Pulizia dati: rimuovi trial crashate per le metriche RQ3
df_valid = df[~df["crashed"]].copy()
print(f"\nTrial valide (non crashate): {len(df_valid)} / {len(df)}")

# Crea directory figures/
ensure_figures_dir = lambda: os.makedirs("figures", exist_ok=True)
ensure_figures_dir()

# Prepara colonne derivate utili
# FIX BUG: evita divisione per zero con np.where
df_valid["handoff_success_rate"] = np.where(
    df_valid["handoffs_attempted"] > 0,
    df_valid["handoffs_successful"] / df_valid["handoffs_attempted"],
    np.nan
)

# Parsa tokens_per_agent da stringa JSON a dict
# Nota: pd.json_normalize può aver appiattito extra_params in colonne come "extra_params.tokens_per_agent"
def parse_tokens_from_row(row):
    tp = row.get("tokens_per_agent")
    if tp is not None:
        if isinstance(tp, str):
            try:
                return json.loads(tp)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(tp, dict):
            return tp
    # Fallback: cerca colonna appiattita
    flat_tp = row.get("extra_params.tokens_per_agent")
    if isinstance(flat_tp, str):
        try:
            return json.loads(flat_tp)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}

df_valid["_tokens_per_agent"] = df_valid.apply(parse_tokens_from_row, axis=1)
df_valid["_recon_tokens"] = df_valid["_tokens_per_agent"].apply(lambda x: x.get("recon", 0))
df_valid["_planning_tokens"] = df_valid["_tokens_per_agent"].apply(lambda x: x.get("planning", 0))
df_valid["_execution_tokens"] = df_valid["_tokens_per_agent"].apply(lambda x: x.get("execution", 0))

# Estrai adaptation_depth (fallback su colonna appiattita se extra_params non esiste come colonna)
if "extra_params" in df_valid.columns:
    df_valid["_adaptation_depth"] = df_valid["extra_params"].apply(
        lambda x: x.get("adaptation_depth", 0) if isinstance(x, dict) else 0
    )
elif "extra_params.adaptation_depth" in df_valid.columns:
    df_valid["_adaptation_depth"] = pd.to_numeric(df_valid["extra_params.adaptation_depth"], errors="coerce").fillna(0).astype(int)
else:
    df_valid["_adaptation_depth"] = 0

# ============================================================================
# RQ3 – Qualità della Collaborazione fra Agenti
# ============================================================================

# --- 1. Distribuzione del numero di messaggi scambiati (bins adattivo) ---
print("\n=== Grafico 1: Distribuzione messaggi ===")
plt.figure(figsize=(9, 5))
msg_data = df_valid["agent_messages"].dropna()
n_bins = max(10, int(np.sqrt(len(msg_data))))
sns.histplot(msg_data, bins=n_bins, kde=True, color="mediumpurple", edgecolor="black", alpha=0.7)
plt.title(f"Distribuzione del numero di messaggi scambiati (n={len(msg_data)}, bins={n_bins})")
plt.xlabel("Numero di messaggi")
plt.ylabel("Frequenza")
plt.tight_layout()
plt.savefig("figures/rq3_messages_hist.pdf")
plt.show()

# --- 2. Tasso di successo degli handoff per tipo di attacco (con CI 95%) ---
print("=== Grafico 2: Handoff Success Rate (con CI) ===")
plt.figure(figsize=(9, 5))

handoff_stats = []
for attack, group in df_valid.groupby("attack_type"):
    valid_group = group.dropna(subset=["handoff_success_rate"])
    n = len(valid_group)
    if n == 0:
        continue
    rate = valid_group["handoff_success_rate"].mean()
    handoff_stats.append({
        "attack_type": attack,
        "handoff_rate": rate,
        "n_trials": n,
    })

handoff_df = pd.DataFrame(handoff_stats)
palette = sns.color_palette("viridis", len(handoff_df))

for i, row in handoff_df.iterrows():
    plt.bar(row["attack_type"], row["handoff_rate"],
            color=palette[i % len(palette)], edgecolor="black", alpha=0.85)
    plt.text(
        row["attack_type"], row["handoff_rate"] + 0.02,
        f"{row['handoff_rate']:.2%}\n(n={row['n_trials']})",
        ha="center", va="bottom", fontsize=8,
    )

plt.title("Tasso medio di successo degli handoff per tipo di attacco")
plt.xlabel("Tipo di attacco")
plt.ylabel("Handoff Success Rate")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("figures/rq3_handoff_success_rate.pdf")
plt.show()

# --- 3. Confronto tra handoff tentati e riusciti (media per tipo di attacco) ---
print("=== Grafico 3: Handoff Comparison ===")
plt.figure(figsize=(10, 6))
handoff_avg = df_valid.groupby("attack_type")[["handoffs_attempted", "handoffs_successful"]].mean().reset_index()
handoff_avg = handoff_avg.melt(id_vars="attack_type", var_name="Tipo", value_name="Media")
sns.barplot(data=handoff_avg, x="attack_type", y="Media", hue="Tipo", palette="Set2")
plt.title("Media di handoff tentati e riusciti per tipo di attacco")
plt.xlabel("Tipo di attacco")
plt.ylabel("Numero medio di handoff")
plt.legend(title="")
plt.tight_layout()
plt.savefig("figures/rq3_handoff_comparison.pdf")
plt.show()

# --- 4. Consumo di token per agente (da stringa JSON parsata) ---
print("=== Grafico 4: Token per Agente (parsato) ===")
plt.figure(figsize=(8, 5))
agent_tokens = {
    "recon": df_valid["_recon_tokens"].mean(),
    "planning": df_valid["_planning_tokens"].mean(),
    "execution": df_valid["_execution_tokens"].mean(),
}
df_token_agent = pd.DataFrame(list(agent_tokens.items()), columns=["Agente", "Token medi"])
sns.barplot(data=df_token_agent, x="Agente", y="Token medi", palette="mako")
plt.title("Consumo medio di token per agente")
plt.xlabel("Agente")
plt.ylabel("Numero medio di token")
plt.tight_layout()
plt.savefig("figures/rq3_token_per_agent.pdf")
plt.show()

# --- 5. Distribuzione del numero di token totali per run ---
print("=== Grafico 5: Token Totali ===")
if "total_tokens" in df_valid.columns:
    plt.figure(figsize=(9, 5))
    token_data = df_valid["total_tokens"].dropna()
    n_bins = max(10, int(np.sqrt(len(token_data))))
    sns.histplot(token_data, bins=n_bins, kde=True, color="coral", edgecolor="black", alpha=0.7)
    plt.title(f"Distribuzione del consumo totale di token per run (n={len(token_data)}, bins={n_bins})")
    plt.xlabel("Token totali")
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.savefig("figures/rq3_total_tokens_hist.pdf")
    plt.show()

# --- 6. Relazione tra messaggi scambiati e numero di handoff (con jitter) ---
print("=== Grafico 6: Messages vs Handoffs (con jitter) ===")
plt.figure(figsize=(9, 5))

jitter_x = np.random.uniform(-0.3, 0.3, len(df_valid))
jitter_y = np.random.uniform(-0.3, 0.3, len(df_valid))

sns.scatterplot(
    data=df_valid.assign(
        messages_jitter=df_valid["agent_messages"] + jitter_x,
        handoffs_jitter=df_valid["handoffs_attempted"] + jitter_y,
    ),
    x="messages_jitter",
    y="handoffs_jitter",
    hue="attack_type",
    style="success",
    s=60,
    alpha=0.7,
)
plt.title("Relazione tra messaggi scambiati e handoff tentati (jitter applicato)")
plt.xlabel("Numero di messaggi")
plt.ylabel("Handoff tentati")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig("figures/rq3_messages_vs_handoffs.pdf")
plt.show()

# ============================================================================
# GRAFICI AGGIUNTIVI: QBER trajectory + Correlazione agent_messages × success
# ============================================================================

# --- 7. QBER trajectory attraverso i round adattivi (media ± CI) ---
print("\n=== Grafico 7: QBER Trajectory ===")
qber_trajectory_data = []
for _, row in df_valid.iterrows():
    # Cerca extra_params come dict, o come stringa JSON, o colonna appiattita
    traj = row.get("extra_params")
    if traj is None:
        flat_tp = row.get("extra_params.qber_trajectory")
        if flat_tp is not None:
            if isinstance(flat_tp, str):
                try:
                    traj = json.loads(flat_tp)
                except (json.JSONDecodeError, TypeError):
                    traj = {}
            elif isinstance(flat_tp, dict):
                traj = flat_tp
    elif isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except (json.JSONDecodeError, TypeError):
            traj = {}
    if isinstance(traj, dict):
        qber_traj = traj.get("qber_trajectory", [])
        if qber_traj and len(qber_traj) > 1:
            qber_trajectory_data.append({
                "round": len(qber_traj) - 1,
                "qber": qber_traj[-1],
                "attack_type": row.get("attack_type", "unknown"),
            })

df_traj = pd.DataFrame(qber_trajectory_data)

if len(df_traj) > 0:
    # Raggruppa per round e calcola media ± CI
    traj_stats = []
    for round_idx, group in df_traj.groupby("round"):
        n = len(group)
        mean_qber = group["qber"].mean()
        std_qber = group["qber"].std()
        # CI 95%
        se = std_qber / np.sqrt(n) if n > 0 else 0
        ci = 1.96 * se
        traj_stats.append({
            "round": round_idx,
            "mean_qber": mean_qber,
            "std_qber": std_qber,
            "ci": ci,
            "n_trials": n,
        })

    df_traj_stats = pd.DataFrame(traj_stats)

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        df_traj_stats["round"],
        df_traj_stats["mean_qber"],
        yerr=df_traj_stats["ci"],
        fmt='o-',
        color='darkblue',
        capsize=4,
        label=f"Media QBER (n={df_traj_stats['n_trials'].sum()})"
    )
    plt.fill_between(
        df_traj_stats["round"],
        df_traj_stats["mean_qber"] - df_traj_stats["ci"],
        df_traj_stats["mean_qber"] + df_traj_stats["ci"],
        alpha=0.15,
        color='darkblue',
        label='CI 95%'
    )
    plt.axhline(y=0.11, color='red', linestyle='--', linewidth=1.5, label="Soglia QBER (0.11)")
    plt.title("QBER trajectory attraverso i round adattivi")
    plt.xlabel("Round adattivo")
    plt.ylabel("QBER medio")
    plt.xticks(df_traj_stats["round"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/rq3_qber_trajectory.pdf")
    plt.show()

    # Stats riassuntive
    print("\n  [QBER Trajectory - Statistiche]")
    for _, row in df_traj_stats.iterrows():
        print(f"    Round {row['round']}: QBER={row['mean_qber']:.4f}±{row['ci']:.4f} (n={row['n_trials']})")
else:
    print("  Nessun dato QBER trajectory disponibile (nessuna trial con >1 round).")

# --- 8. Correlazione agent_messages × successo dell'attacco ---
print("\n=== Grafico 8: Correlazione Messages × Success ===")
if "agent_messages" in df_valid.columns and df_valid["agent_messages"].dropna().shape[0] > 2:
    plt.figure(figsize=(9, 5))
    
    # Scatter con jitter
    jitter = np.random.uniform(-0.3, 0.3, len(df_valid))
    x_vals = df_valid["agent_messages"] + jitter
    
    sns.scatterplot(
        data=df_valid.assign(messages_jitter=x_vals),
        x="messages_jitter",
        y="success",
        hue="attack_type",
        palette="deep",
        alpha=0.7,
        s=60,
    )
    
    # Regressione lineare
    valid_data = df_valid.dropna(subset=["agent_messages"])
    if len(valid_data) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data["agent_messages"],
            valid_data["success"]
        )
        x_line = np.array([valid_data["agent_messages"].min(), valid_data["agent_messages"].max()])
        plt.plot(
            x_line,
            intercept + slope * x_line,
            'r--',
            linewidth=1.5,
            label=f"Regione (r={r_value:.4f}, p={p_value:.4g})"
        )
    
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.title("Correlazione: messaggi scambiati vs successo dell'attacco")
    plt.xlabel("Numero di messaggi (con jitter)")
    plt.ylabel("Success (0/1)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/rq3_messages_vs_success.pdf")
    plt.show()
    
    # Stats riassuntive
    print("\n  [Correlazione Messages × Success - Statistiche]")
    print(f"    Pearson r = {r_value:.4f}, p-value = {p_value:.4g}")
    print(f"    Regressione: success = {intercept:.4f} + {slope:.6f} × messages")
    
    # Breakdown per valore di agent_messages
    print("\n    Breakdown per agent_messages:")
    for msg_val in sorted(df_valid["agent_messages"].dropna().unique()):
        group = df_valid[df_valid["agent_messages"] == msg_val]
        print(f"      messages={msg_val}: success_rate={group['success'].mean():.2%} (n={len(group)})")
else:
    print("  Dati insufficienti per la correlazione.")

# ============================================================================
# Summary
# ============================================================================
print("\n=== Riepilogo RQ3 ===")
total = len(df_valid)
print(f"  Trial valide:           {total}")
print(f"  Agent messages medio:   {df_valid['agent_messages'].mean():.1f}")
print(f"  Handoff rate medio:     {df_valid['handoff_success_rate'].mean():.2%}")
print(f"  Adaption depth medio:   {df_valid['_adaptation_depth'].mean():.1f}")
print(f"  Token totali medio:     {df_valid['total_tokens'].mean():.0f}")

print("\nGrafici RQ3 salvati nella cartella 'figures/'")
