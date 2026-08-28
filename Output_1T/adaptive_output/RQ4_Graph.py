import json
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
    Carica i risultati da un file .json o .jsonl.
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

# Stampa riepilogo
print("Numero di run caricate:", len(df))
print("Colonne disponibili:\n", df.columns.tolist())

# Pulizia dati: rimuovi trial crashate per le metriche di efficienza
df_valid = df[~df["crashed"]].copy()
print(f"\nTrial valide (non crashate): {len(df_valid)} / {len(df)}")

# Parsa tokens_per_agent da stringa JSON a dict
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

# Estrai adaptation_depth
if "extra_params" in df_valid.columns:
    df_valid["_adaptation_depth"] = df_valid["extra_params"].apply(
        lambda x: x.get("adaptation_depth", 0) if isinstance(x, dict) else 0
    )
elif "extra_params.adaptation_depth" in df_valid.columns:
    df_valid["_adaptation_depth"] = pd.to_numeric(df_valid["extra_params.adaptation_depth"], errors="coerce").fillna(0).astype(int)
else:
    df_valid["_adaptation_depth"] = 0

# ============================================================================
# RQ4 – Efficienza Operativa
# ============================================================================

# --- 1. Distribuzione dei tempi di esecuzione (bins adattivo) ---
print("\n=== Grafico 1: Distribuzione tempi ===")
plt.figure(figsize=(9, 5))
time_data = df_valid["wall_time_sec"].dropna()
n_bins = max(10, int(np.sqrt(len(time_data))))
sns.histplot(time_data, bins=n_bins, kde=True, color="teal", edgecolor="black", alpha=0.7)
plt.axvline(x=time_data.mean(), color="red", linestyle="--", linewidth=1.5,
            label=f"Media ({time_data.mean():.1f}s)")
plt.title(f"Distribuzione dei tempi di esecuzione (n={len(time_data)}, bins={n_bins})")
plt.xlabel("Tempo (secondi)")
plt.ylabel("Frequenza")
plt.legend()
plt.tight_layout()
plt.savefig("figures/rq4_time_hist.pdf")
plt.show()

# --- 2. Box plot dei tempi per tipo di attacco (con media) ---
print("=== Grafico 2: Box plot tempi ===")
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_valid, x="attack_type", y="wall_time_sec", palette="Set3",
            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red",
                                        "markeredgecolor": "black"})
# Aggiungi valori medi in testa
for attack, group in df_valid.groupby("attack_type"):
    mean_time = group["wall_time_sec"].mean()
    plt.text(
        attack, mean_time + group["wall_time_sec"].std() * 0.3,
        f"{mean_time:.1f}s",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )
plt.title("Tempo di esecuzione per tipo di attacco (D = media)")
plt.xlabel("Tipo di attacco")
plt.ylabel("Tempo (secondi)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("figures/rq4_time_boxplot.pdf")
plt.show()

# --- 3. Relazione tra tempo di esecuzione e token totali (con jitter) ---
print("=== Grafico 3: Tempo vs Token ===")
plt.figure(figsize=(9, 5))

jitter_x = np.random.uniform(-5, 5, len(df_valid))
jitter_y = np.random.uniform(-500, 500, len(df_valid))

sns.scatterplot(
    data=df_valid.assign(
        time_jitter=df_valid["wall_time_sec"] + jitter_x,
        tokens_jitter=df_valid["total_tokens"] + jitter_y,
    ),
    x="time_jitter",
    y="tokens_jitter",
    hue="attack_type",
    style="crashed",
    s=60,
    alpha=0.7,
)

# Regressione lineare
valid_time = df_valid.dropna(subset=["wall_time_sec", "total_tokens"])
if len(valid_time) > 2:
    slope, intercept, r_value, p_value, _ = stats.linregress(
        valid_time["wall_time_sec"],
        valid_time["total_tokens"]
    )
    x_line = np.array([valid_time["wall_time_sec"].min(), valid_time["wall_time_sec"].max()])
    plt.plot(
        x_line,
        intercept + slope * x_line,
        'r--',
        linewidth=1.5,
        label=f"Regione (r={r_value:.4f}, p={p_value:.4g})"
    )

plt.title("Tempo di esecuzione vs Token totali (jitter applicato)")
plt.xlabel("Tempo (secondi)")
plt.ylabel("Token totali")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig("figures/rq4_time_vs_tokens.pdf")
plt.show()

# Stats riassuntive
if len(valid_time) > 2:
    print(f"\n  [Tempo vs Token - Correlazione]")
    print(f"    Pearson r = {r_value:.4f}, p-value = {p_value:.4g}")
    print(f"    Regressione: tokens = {intercept:.0f} + {slope:.2f} × tempo(s)")

# --- 4. Tasso di crash per tipo di attacco ---
print("=== Grafico 4: Crash Rate per tipo di attacco ===")
plt.figure(figsize=(9, 5))

crash_stats = []
for attack, group in df.groupby("attack_type"):
    n = len(group)
    crashed = group["crashed"].sum()
    rate = crashed / n if n > 0 else 0
    lo, hi = confidence_interval(group["crashed"]) if n > 0 else (0, 1)
    crash_stats.append({
        "attack_type": attack,
        "crash_rate": rate,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_trials": n,
        "n_crashed": int(crashed),
    })

crash_df = pd.DataFrame(crash_stats)
palette = sns.color_palette("crest_r", len(crash_df))

for i, row in crash_df.iterrows():
    plt.bar(
        row["attack_type"],
        row["crash_rate"] * 100,
        yerr=(row["ci_upper"] - row["crash_rate"]) * 100,
        capsize=4,
        color=palette[i % len(palette)],
        edgecolor="black",
        alpha=0.85,
    )
    plt.text(
        row["attack_type"],
        row["crash_rate"] * 100 + (row["ci_upper"] - row["crash_rate"]) * 100 + 1,
        f"{row['crash_rate']:.1%}\n({row['n_crashed']}/{row['n_trials']})",
        ha="center", va="bottom", fontsize=8,
    )

plt.title("Tasso di crash per tipo di attacco (CI 95%)")
plt.xlabel("Tipo di attacco")
plt.ylabel("Crash Rate (%)")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig("figures/rq4_crash_rate.pdf")
plt.show()

# --- 5. Consumo di token per agente (da stringa JSON parsata) ---
print("=== Grafico 5: Token per Agente ===")
agent_tokens = {
    "recon": df_valid["_recon_tokens"].mean(),
    "planning": df_valid["_planning_tokens"].mean(),
    "execution": df_valid["_execution_tokens"].mean(),
}
df_token_agent = pd.DataFrame(list(agent_tokens.items()), columns=["Agente", "Token medi"])
plt.figure(figsize=(8, 5))
sns.barplot(data=df_token_agent, x="Agente", y="Token medi", palette="viridis")
plt.title("Consumo medio di token per agente")
plt.xlabel("Agente")
plt.ylabel("Numero medio di token")
plt.tight_layout()
plt.savefig("figures/rq4_token_per_agent.pdf")
plt.show()

# --- 6. Statistiche descrittive essenziali ---
print("\n--- Statistiche RQ4 ---")
time_mean = df_valid["wall_time_sec"].mean() if df_valid["wall_time_sec"].notna().any() else 0
time_std = df_valid["wall_time_sec"].std() if df_valid["wall_time_sec"].notna().any() else 0
time_cv = time_std / time_mean if time_mean > 0 else 0
token_mean = df_valid["total_tokens"].mean() if df_valid["total_tokens"].notna().any() else 0
token_std = df_valid["total_tokens"].std() if df_valid["total_tokens"].notna().any() else 0
total_crashed = df["crashed"].sum()
crash_rate = total_crashed / len(df) if len(df) > 0 else 0

print(f"  Tempo medio:          {time_mean:.2f} s")
print(f"  Deviazione standard:  {time_std:.2f} s")
print(f"  CV tempo:             {time_cv:.2f}")
print(f"  Token medi per run:   {token_mean:.0f}")
print(f"  Deviazione standard:  {token_std:.0f}")
print(f"  Tasso di crash:       {crash_rate:.1%}")

# --- 7. CV del tempo per tipo di attacco (con clipping per divisione per zero) ---
print("=== Grafico 7: CV Tempo per tipo di attacco ===")
cv_by_attack = df.groupby("attack_type")["wall_time_sec"].apply(
    lambda x: x.std() / max(x.mean(), np.finfo(float).eps)
).reset_index()
cv_by_attack.columns = ["attack_type", "CV_tempo"]

plt.figure(figsize=(9, 5))
sns.barplot(data=cv_by_attack, x="attack_type", y="CV_tempo", palette="rocket",
            edgecolor="black")
plt.title("Coefficiente di variazione del tempo per tipo di attacco")
plt.xlabel("Tipo di attacco")
plt.ylabel("CV tempo")
plt.tight_layout()
plt.savefig("figures/rq4_cv_time_by_attack.pdf")
plt.show()

# ============================================================================
# GRAFICI AGGIUNTIVI: RQ4 Efficienza
# ============================================================================

# --- 8. Correlazione wall_time × successo dell'attacco ---
print("\n=== Grafico 8: Wall Time vs Success ===")
if "wall_time_sec" in df_valid.columns and df_valid["wall_time_sec"].dropna().shape[0] > 2:
    plt.figure(figsize=(9, 5))
    
    jitter = np.random.uniform(-3, 3, len(df_valid))
    x_vals = df_valid["wall_time_sec"] + jitter
    
    sns.scatterplot(
        data=df_valid.assign(time_jitter=x_vals),
        x="time_jitter",
        y="success",
        hue="attack_type",
        palette="deep",
        alpha=0.7,
        s=60,
    )
    
    valid_data = df_valid.dropna(subset=["wall_time_sec"])
    if len(valid_data) > 2:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            valid_data["wall_time_sec"],
            valid_data["success"]
        )
        x_line = np.array([valid_data["wall_time_sec"].min(), valid_data["wall_time_sec"].max()])
        plt.plot(
            x_line,
            intercept + slope * x_line,
            'r--',
            linewidth=1.5,
            label=f"Regione (r={r_value:.4f}, p={p_value:.4g})"
        )
    
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.title("Correlazione: wall time vs successo dell'attacco")
    plt.xlabel("Tempo (secondi, con jitter)")
    plt.ylabel("Success (0/1)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/rq4_time_vs_success.pdf")
    plt.show()
    
    print(f"\n  [Wall Time × Success - Correlazione]")
    print(f"    Pearson r = {r_value:.4f}, p-value = {p_value:.4g}")
    print(f"    Regressione: success = {intercept:.4f} + {slope:.6f} × tempo(s)")
else:
    print("  Dati insufficienti per la correlazione.")

# --- 9. Correlazione agent_messages × wall_time ---
print("\n=== Grafico 9: Messages × Wall Time ===")
if "agent_messages" in df_valid.columns and "wall_time_sec" in df_valid.columns:
    plt.figure(figsize=(9, 5))
    
    jitter = np.random.uniform(-0.3, 0.3, len(df_valid))
    x_vals = df_valid["agent_messages"] + jitter
    
    sns.scatterplot(
        data=df_valid.assign(messages_jitter=x_vals),
        x="messages_jitter",
        y="wall_time_sec",
        hue="attack_type",
        palette="deep",
        alpha=0.7,
        s=60,
    )
    
    valid_data = df_valid.dropna(subset=["agent_messages", "wall_time_sec"])
    if len(valid_data) > 2:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            valid_data["agent_messages"],
            valid_data["wall_time_sec"]
        )
        x_line = np.array([valid_data["agent_messages"].min(), valid_data["agent_messages"].max()])
        plt.plot(
            x_line,
            intercept + slope * x_line,
            'r--',
            linewidth=1.5,
            label=f"Regione (r={r_value:.4f}, p={p_value:.4g})"
        )
    
    plt.title("Correlazione: messaggi scambiati vs wall time")
    plt.xlabel("Numero di messaggi (con jitter)")
    plt.ylabel("Tempo (secondi)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/rq3_messages_vs_time.pdf")
    plt.show()
    
    print(f"\n  [Messages × Time - Correlazione]")
    print(f"    Pearson r = {r_value:.4f}, p-value = {p_value:.4g}")
    print(f"    Regressione: tempo = {intercept:.1f} + {slope:.1f} × messages")
else:
    print("  Dati insufficienti per la correlazione.")

# --- 10. Distribuzione adaptation depth (costo dell'adattamento) ---
print("\n=== Grafico 10: Adaption Depth Distribution ===")
plt.figure(figsize=(9, 5))
depth_data = df_valid["_adaptation_depth"]
unique_depths = sorted(depth_data.unique())
depth_counts = [depth_data[depth_data == d].shape[0] for d in unique_depths]
depth_rates = [c / len(depth_data) for c in depth_counts]

plt.bar(unique_depths, depth_rates, width=0.6, color="steelblue", edgecolor="black", alpha=0.85)
for d, count, rate in zip(unique_depths, depth_counts, depth_rates):
    plt.text(d, count + 1, f"{count}\n({rate:.1%})", ha="center", va="bottom", fontsize=8)

plt.title("Distribuzione dell'adaptation depth (costo dell'adattamento)")
plt.xlabel("Adaptation Depth (round aggiuntivi)")
plt.ylabel("Frequenza relativa")
plt.xticks(unique_depths)
plt.tight_layout()
plt.savefig("figures/rq4_adaptation_depth.pdf")
plt.show()

# Stats riassuntive
print(f"\n  [Adaptation Depth - Statistiche]")
print(f"    Media: {depth_data.mean():.1f}")
print(f"    Max: {depth_data.max()}")
print(f"    Min: {depth_data.min()}")
for d in unique_depths:
    count = depth_data[depth_data == d].shape[0]
    rate = count / len(depth_data)
    print(f"      depth={d}: {count} trial ({rate:.1%})")

# --- 11. Token efficiency (success rate per token speso) ---
print("\n=== Grafico 11: Token Efficiency ===")
df_efficiency = df_valid.dropna(subset=["total_tokens", "success"]).copy()
if len(df_efficiency) > 0:
    df_efficiency["_token_efficiency"] = df_efficiency["success"] / df_efficiency["total_tokens"]
    
    plt.figure(figsize=(9, 5))
    
    eff_stats = []
    for attack, group in df_efficiency.groupby("attack_type"):
        n = len(group)
        mean_eff = group["_token_efficiency"].mean()
        std_eff = group["_token_efficiency"].std()
        eff_stats.append({
            "attack_type": attack,
            "efficiency": mean_eff,
            "std": std_eff,
            "n_trials": n,
        })
    
    eff_df = pd.DataFrame(eff_stats)
    eff_df = eff_df.sort_values("efficiency", ascending=True)
    palette = sns.color_palette("viridis", len(eff_df))
    
    for i, row in eff_df.iterrows():
        plt.barh(row["attack_type"], row["efficiency"],
                 xerr=[[row["std"]], [row["std"]]],
                 capsize=3,
                 color=palette[i % len(palette)],
                 edgecolor="black",
                 alpha=0.85)
        plt.text(
            row["efficiency"] + row["std"] + 1e-7,
            row["attack_type"],
            f"{row['efficiency']:.2e}\n(n={row['n_trials']})",
            ha="left", va="center", fontsize=8,
        )
    
    plt.title("Token Efficiency (success rate per token speso)")
    plt.xlabel("Token Efficiency (success / tokens)")
    plt.ylabel("Tipo di attacco")
    plt.tight_layout()
    plt.savefig("figures/rq4_token_efficiency.pdf")
    plt.show()
    
    # Stats riassuntive
    print(f"\n  [Token Efficiency - Top 3]")
    for _, row in eff_df.head(3).iterrows():
        print(f"    {row['attack_type']}: {row['efficiency']:.2e} (±{row['std']:.2e})")
else:
    print("  Dati insufficienti per il calcolo dell'efficienza.")

# ============================================================================
# Summary
# ============================================================================
print("\n=== Riepilogo RQ4 ===")
print(f"  Trial totali:           {len(df)}")
print(f"  Trial valide:           {len(df_valid)} ({len(df_valid)/len(df):.1%})")
print(f"  Trial crashate:         {int(total_crashed)} ({crash_rate:.1%})")
print(f"  Tempo medio:            {time_mean:.1f}s ± {time_std:.1f}s")
print(f"  Token medi per run:     {token_mean:.0f} ± {token_std:.0f}")
print(f"  CV tempo (globale):     {time_cv:.2f}")

print("\nGrafici RQ4 salvati nella cartella 'figures/'")
