from evaluation.adaptive_trial_runner import run_adaptive_evaluation

print("=" * 70)
print("ADAPTIVE EVALUATION - 10 TRIAL PER SCENARIO")
print("=" * 70)

results = run_adaptive_evaluation(
    trials_per_scenario=3,
    seed=42,
    scenario_names=['intercept_resend', 'pns', 'blinding', 'trojan_horse', 'qber_tamper'],
    output_dir='adaptive_output',
    enable_adaptive_loop=True,
    max_adaptation_rounds=3,
    execution_temperature=0.5,
)

print(f"\n{'=' * 70}")
print(f"Completato: {len(results)} trial totali eseguite")
print(f"Output: adaptive_output/")
print(f"{'=' * 70}")
