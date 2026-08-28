"""
Evaluation module for BB84 AI Red Teaming framework.

Modulari:
- attack_scenarios.py: Scenari d'attacco configurabili
- trial_logger.py: Registrazione risultati delle trial
- evaluation_runner.py: Orchestratore delle run (simulatore diretto + flusso agenti)
- analysis.py: Metriche per RQ1, RQ3, RQ5
- plotting.py: Generazione grafici per la tesi
"""

from pathlib import Path

# Creare directory output/plots se non esistono
OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR = Path(__file__).parent / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
