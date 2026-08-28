import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Configurazione stile grafici
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

# ============================================================================
# Utility
# ============================================================================

def load_results(file_path):
    """
    Carica i risultati da un file .json (lista di run).
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


def ensure_figures_dir():
    """Crea la directory figures/ se non esiste."""
    os.makedirs("figures", exist_ok=True)


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
print("\nScenari presenti:", df["scenario_name"].unique().tolist())
print("Trial per scenario:\n", df.groupby("scenario_name").size())

# Pulizia dati: rimuovi trial crashate per le metriche di effectiveness
df_valid = df[~df["crashed"]].copy()
print(f"\nTrial valide (non crashate): {len(df_valid)} / {len(df)}")

# ============================================================================
# RQ1 – Attack Effectiveness
# ============================================================================

# --- 1. Tasso di successo per tipo di attacco (con CI 95%) ---
print("\n=== Grafico 1: Success Rate per tipo di attacco ===")
plt.figure(figsize=(9, 5))

success_stats = []
for attack, group in df_valid.groupby("attack_type"):
    n = len(group)
    rate = group["success"].mean()
    lo, hi = confidence_interval(group["success"])
    success_stats.append({
        "attack_type": attack,
        "success_rate": rate,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_trials": n,
    })

success_df = pd.DataFrame(success_stats)
palette = sns.color_palette("viridis", len(success_df))

for i, row in success_df.iterrows():
    plt.bar(
        row["attack_type"],
        row["success_rate"],
        yerr=[[0], [row["ci_upper"] - row["success_rate"]]],
        capsize=4,
        color=palette[i % len(palette)],
        edgecolor="black",
        alpha=0.85,
    )
    plt.text(
        row["attack_type"],
        row["success_rate"] + (row["ci_upper"] - row["success_rate"]) + 0.02,
        f"{row['success_rate']:.1%}\n(n={row['n_trials']})",
        ha="center", va="bottom", fontsize=8,
    )

plt.title("Tasso di successo per tipo di attacco (CI 95%)")
plt.xlabel("Tipo di attacco")
plt.ylabel("Success Rate")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/rq1_success_rate_by_attack_type.pdf")
plt.show()

# --- 2. Tasso di compromissione chiave (con CI 95%) ---
print("=== Grafico 2: Key Compromise Rate per tipo di attacco ===")
plt.figure(figsize=(9, 5))

compromise_stats = []
for attack, group in df_valid.groupby("attack_type"):
    n = len(group)
    rate = group["key_compromised"].mean()
    lo, hi = confidence_interval(group["key_compromised"])
    compromise_stats.append({
        "attack_type": attack,
        "compromise_rate": rate,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_trials": n,
    })

compromise_df = pd.DataFrame(compromise_stats)
palette = sns.color_palette("rocket_r", len(compromise_df))

for i, row in compromise_df.iterrows():
    plt.bar(
        row["attack_type"],
        row["compromise_rate"],
        yerr=[[0], [row["ci_upper"] - row["compromise_rate"]]],
        capsize=4,
        color=palette[i % len(palette)],
        edgecolor="black",
        alpha=0.85,
    )
    plt.text(
        row["attack_type"],
        row["compromise_rate"] + (row["ci_upper"] - row["compromise_rate"]) + 0.02,
        f"{row['compromise_rate']:.1%}\n(n={row['n_trials']})",
        ha="center", va="bottom", fontsize=8,
    )

plt.title("Tasso di compromissione della chiave (CI 95%)")
plt.xlabel("Tipo di attacco")
plt.ylabel("Key Compromised Rate")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/rq1_key_compromise_rate.pdf")
plt.show()

# --- 3. Distribuzione del QBER osservato (bins adattivo) ---
print("=== Grafico 3: Istogramma QBER ===")
plt.figure(figsize=(9, 5))
qber_data = df_valid["qber"].dropna()
n_bins = max(10, int(np.sqrt(len(qber_data))))
sns.histplot(qber_data, bins=n_bins, kde=True, stat="density",
             color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(x=0.11, color="red", linestyle="--", linewidth=1.5,
            label="Soglia QBER (0.11)")
plt.axvline(x=qber_data.mean(), color="blue", linestyle=":", linewidth=1.5,
            label=f"Media QBER ({qber_data.mean():.4f})")
plt.title(f"Distribuzione del QBER osservato (n={len(qber_data)}, bins={n_bins})")
plt.xlabel("QBER")
plt.ylabel("Densità")
plt.legend()
plt.tight_layout()
plt.savefig("figures/rq1_qber_histogram.pdf")
plt.show()

# --- 4. Confronto QBER osservato vs reale (con jitter) ---
print("=== Grafico 4: Scatter QBER ===")
plt.figure(figsize=(9, 5))

# Jitter per evitare sovrapposizione
jitter_x = np.random.uniform(-0.01, 0.01, len(df_valid))
jitter_y = np.random.uniform(-0.01, 0.01, len(df_valid))

sns.scatterplot(
    data=df_valid.assign(
        qber_jitter=df_valid["qber"] + jitter_x,
        true_qber_jitter=df_valid["true_qber"] + jitter_y,
    ),
    x="qber_jitter",
    y="true_qber_jitter",
    hue="attack_type",
    style="detected",
    palette="deep",
    alpha=0.7,
    s=60,
)
plt.plot([0, 0.3], [0, 0.3], 'k--', linewidth=1, label="y = x")
plt.title("QBER osservato vs QBER reale (jitter applicato)")
plt.xlabel("QBER osservato (Alice)")
plt.ylabel("QBER reale (Eve)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig("figures/rq1_qber_scatter.pdf", bbox_inches="tight")
plt.show()

# --- 5. Tasso di stealth per tipo di attacco (con CI 95%) ---
print("=== Grafico 5: Stealth Rate ===")
df_valid["stealth"] = ~df_valid["detected"]

stealth_stats = []
for attack, group in df_valid.groupby("attack_type"):
    n = len(group)
    rate = group["stealth"].mean()
    lo, hi = confidence_interval(group["stealth"])
    stealth_stats.append({
        "attack_type": attack,
        "stealth_rate": rate,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_trials": n,
    })

stealth_df = pd.DataFrame(stealth_stats)
palette = sns.color_palette("mako_r", len(stealth_df))

for i, row in stealth_df.iterrows():
    plt.bar(
        row["attack_type"],
        row["stealth_rate"],
        yerr=[[0], [row["ci_upper"] - row["stealth_rate"]]],
        capsize=4,
        color=palette[i % len(palette)],
        edgecolor="black",
        alpha=0.85,
    )
    plt.text(
        row["attack_type"],
        row["stealth_rate"] + (row["ci_upper"] - row["stealth_rate"]) + 0.02,
        f"{row['stealth_rate']:.1%}\n(n={row['n_trials']})",
        ha="center", va="bottom", fontsize=8,
    )

plt.title("Tasso di stealth (non rilevato) per tipo di attacco (CI 95%)")
plt.xlabel("Tipo di attacco")
plt.ylabel("Stealth Rate")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/rq1_stealth_rate.pdf")
plt.show()

# --- 6. Box plot del QBER per tipo di attacco ---
print("=== Grafico 6: Box plot QBER ===")
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_valid, x="attack_type", y="qber", palette="Set2",
            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red",
                                        "markeredgecolor": "black"})
plt.title("Distribuzione del QBER per tipo di attacco (D = media)")
plt.xlabel("Tipo di attacco")
plt.ylabel("QBER")
plt.tight_layout()
plt.savefig("figures/rq1_qber_boxplot.pdf")
plt.show()

# ============================================================================
# GRAFICI AGGIUNTIVI: Token Usage e Wall Time
# ============================================================================

# --- 7. Correlazione tra token totali e successo dell'attacco ---
print("\n=== Grafico 7: Token Usage vs Successo ===")
if "total_tokens" in df_valid.columns:
    plt.figure(figsize=(9, 5))
    
    df_tokens = df_valid[df_valid["total_tokens"] > 0].copy()
    
    if len(df_tokens) > 0:
        # Scatter con jitter
        jitter = np.random.uniform(-0.5, 0.5, len(df_tokens))
        x_vals = df_tokens["total_tokens"] + jitter
        
        sns.scatterplot(
            data=df_tokens.assign(x_adj=x_vals),
            x="x_adj",
            y="success",
            hue="attack_type",
            palette="deep",
            alpha=0.7,
            s=60,
        )
        
        # Linea di regressione logistica
        if len(df_tokens) > 2:
            x_clean = df_tokens["total_tokens"].values.reshape(-1, 1)
            y_clean = df_tokens["success"].values
            try:
                logreg = stats.linregress(
                    df_tokens["total_tokens"],
                    df_tokens["success"]
                )
                plt.plot(
                    [x_clean.min(), x_clean.max()],
                    [logreg.intercept, logreg.intercept + logreg.slope * x_clean.max()],
                    'r--', linewidth=1.5,
                    label=f"Regione (slope={logreg.slope:.6f}, r={logreg.rvalue:.4f})"
                )
            except Exception:
                pass
        
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        plt.title("Token totali vs Successo dell'attacco")
        plt.xlabel("Total Tokens")
        plt.ylabel("Success (0/1)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("figures/rq1_tokens_vs_success.pdf")
        plt.show()
        
        # Stats riassuntive
        print("\n  [Token Usage - Statistiche]")
        print(f"  Trial con token tracciati: {len(df_tokens)}")
        for attack, group in df_tokens.groupby("attack_type"):
            print(f"    {attack}: tokens={group['total_tokens'].mean():.0f}±{group['total_tokens'].std():.0f}, "
                  f"success_rate={group['success'].mean():.2%}")
    else:
        print("  Nessun dato token disponibile.")
else:
    print("  Colonna 'total_tokens' non presente nei dati.")

# --- 8. Wall time per tipo di attacco ---
print("\n=== Grafico 8: Wall Time per tipo di attacco ===")
if "wall_time_sec" in df_valid.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_valid, x="attack_type", y="wall_time_sec", palette="PuRd",
                showmeans=True, meanprops={"marker": "D", "markerfacecolor": "orange",
                                            "markeredgecolor": "black"})
    
    # Aggiungi valori medi in testa
    for attack, group in df_valid.groupby("attack_type"):
        mean_time = group["wall_time_sec"].mean()
        plt.text(
            attack, mean_time + group["wall_time_sec"].std() * 0.3,
            f"{mean_time:.1f}s",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    
    plt.title("Wall time per tipo di attacco (D = media)")
    plt.xlabel("Tipo di attacco")
    plt.ylabel("Wall Time (secondi)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("figures/rq1_wall_time_by_attack.pdf")
    plt.show()
    
    # Stats riassuntive
    print("\n  [Wall Time - Statistiche]")
    for attack, group in df_valid.groupby("attack_type"):
        print(f"    {attack}: {group['wall_time_sec'].mean():.1f}s±{group['wall_time_sec'].std():.1f}s "
              f"(min={group['wall_time_sec'].min():.1f}, max={group['wall_time_sec'].max():.1f})")
else:
    print("  Colonna 'wall_time_sec' non presente nei dati.")

# --- 9. Tokens per agente (stacked bar) ---
print("\n=== Grafico 9: Tokens per Agente ===")
if "tokens_per_agent" in df_valid.columns:
    # Estrai i conteggi per agente
    agent_tokens = {"recon": [], "planning": [], "execution": []}
    for _, row in df_valid.iterrows():
        tp = row["tokens_per_agent"]
        if isinstance(tp, str):
            tp = json.loads(tp)
        for agent in agent_tokens:
            agent_tokens[agent].append(tp.get(agent, 0))
    
    df_tokens_agent = pd.DataFrame(agent_tokens)
    df_tokens_agent["attack_type"] = df_valid["attack_type"].values
    
    # Media per agente e scenario
    agent_means = df_tokens_agent.groupby("attack_type").mean()
    
    plt.figure(figsize=(10, 6))
    agents = ["recon", "planning", "execution"]
    x = np.arange(len(agent_means))
    width = 0.25
    
    for i, agent in enumerate(agents):
        plt.bar(x + i * width, agent_means[agent], width,
                label=agent.capitalize(),
                color=sns.color_palette("deep")[i])
    
    plt.xlabel("Tipo di attacco")
    plt.ylabel("Token medi per agente")
    plt.title("Token medi per agente (stacked)")
    plt.xticks(x + width, agent_means.index, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/rq1_tokens_per_agent.pdf")
    plt.show()
else:
    print("  Colonna 'tokens_per_agent' non presente nei dati.")

# ============================================================================
# Summary
# ============================================================================
print("\n=== Riepilogo ===")
total = len(df)
valid = len(df_valid)
crashed = total - valid
success_count = df_valid["success"].sum()
stealth_count = df_valid["stealth"].sum() if "stealth" in df_valid.columns else 0

print(f"  Trial totali:           {total}")
print(f"  Trial valide:           {valid} ({valid/total:.1%})")
print(f"  Trial crashate:         {crashed} ({crashed/total:.1%})")
print(f"  Success rate medio:     {df_valid['success'].mean():.2%}")
print(f"  Stealth rate medio:     {df_valid['stealth'].mean():.2%}")
print(f"  QBER medio:             {df_valid['qber'].mean():.4f}")
print(f"  Wall time medio:        {df_valid['wall_time_sec'].mean():.1f}s")

if "total_tokens" in df_valid.columns:
    tokens_valid = df_valid[df_valid["total_tokens"] > 0]
    if len(tokens_valid) > 0:
        print(f"  Total tokens medio:     {tokens_valid['total_tokens'].mean():.0f}")

print("\nGrafici RQ1 salvati nella cartella 'figures/'")
