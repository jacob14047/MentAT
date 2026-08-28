#!/usr/bin/env python3
"""
Entry point del BB84 Recon Framework.

Esempi d'uso:

    # Con parametri di esempio incorporati (utile per testare la pipeline
    # senza avere ancora agganciato bb84_simulator_Eve.py)
    python cli.py --source example

    # Con un file JSON di risultati prodotto dal tuo simulatore
    python cli.py --source json --input risultati_bb84.json

    # Con un dizionario Python passato come stringa JSON inline
    python cli.py --source dict --input '{"qber": 0.04, "n_qubits_sent": 5000}'
"""
from __future__ import annotations

import argparse
import json
import sys

from agents.execution_agent import ExecutionAgent
from agents.planning_agent import PlanningAgent
from agents.recon_agent import ReconAgent
from channel.bb84_channel_source import BB84ChannelSource
from config.settings import get_settings
from db.sqlite_repository import SQLiteExecutionRepository, SQLitePlanningRepository, SQLiteReconRepository
from llm.lmstudio_client import LMStudioClient
from utils.logger import get_logger

logger = get_logger("cli")

# Parametri di esempio: utili per provare subito la pipeline end-to-end
# (LM Studio + prompt + salvataggio DB) prima di collegare il vero simulatore.
EXAMPLE_PARAMETERS = {
    "n_qubits_sent": 10000,
    "sifted_key_length": 4870,
    "raw_key_length": 5000,
    "qber": 0.038,
    "basis_match_rate": 0.5,
    "channel_loss": 0.18,
    "detector_efficiency": 0.72,
    "dark_count_rate": 0.0004,
    "eve_present": True,
    "eve_strategy": "intercept-resend",
    "eve_interception_rate": 0.25,
    "eve_detection_probability": 0.31,
    "error_correction_leakage": 812,
    "privacy_amplification_ratio": 0.61,
    "final_key_rate": 0.29,
    "protocol_aborted": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BB84 Recon Agent CLI")
    parser.add_argument(
        "--source",
        choices=["example", "json", "dict"],
        default="example",
        help="Origine dei parametri del canale BB84.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path al file JSON (--source json) oppure stringa JSON inline (--source dict).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identificativo della run (default: generato automaticamente).",
    )
    return parser.parse_args()


def build_channel_source(args: argparse.Namespace) -> BB84ChannelSource:
    if args.source == "example":
        return BB84ChannelSource.from_dict(EXAMPLE_PARAMETERS, run_id=args.run_id)

    if args.source == "json":
        if not args.input:
            sys.exit("Errore: --source json richiede --input <path_al_file.json>")
        return BB84ChannelSource.from_json_file(args.input, run_id=args.run_id)

    if args.source == "dict":
        if not args.input:
            sys.exit("Errore: --source dict richiede --input '<json inline>'")
        return BB84ChannelSource.from_dict(json.loads(args.input), run_id=args.run_id)

    raise ValueError(f"Sorgente non supportata: {args.source}")


def main() -> None:
    args = parse_args()
    settings = get_settings()

    channel_source = build_channel_source(args)
    llm_client = LMStudioClient.from_settings(settings.llm)
    recon_repository = SQLiteReconRepository(settings.database.absolute_path())
    planning_repository = SQLitePlanningRepository(settings.database.absolute_path())
    execution_repository = SQLiteExecutionRepository(settings.database.absolute_path())

    print("\n" + "=" * 70)
    print("FASE 1: RECON AGENT")
    print("=" * 70)

    recon_agent = ReconAgent(
        llm_client=llm_client,
        channel_source=channel_source,
        repository=recon_repository,
        qber_abort_threshold=settings.bb84.qber_abort_threshold,
        max_llm_retries=settings.agent.max_llm_retries,
    )
    recon_report = recon_agent.run()

    print("\n" + "=" * 70)
    print("RECON REPORT")
    print("=" * 70)
    print(recon_report.to_json())
    print("=" * 70)

    print("\n" + "=" * 70)
    print("FASE 2: PLANNING AGENT")
    print("=" * 70)

    planning_agent = PlanningAgent(
        recon_repository=recon_repository,
        planning_repository=planning_repository,
        recon_report=recon_report,
        max_llm_retries=settings.planning.max_llm_retries,
        llm_base_url=settings.llm.base_url,
        llm_model_name=settings.llm.model_name,
        llm_temperature=settings.llm.temperature,
        llm_max_tokens=settings.llm.max_tokens,
    )
    planning_report = planning_agent.run()

    print("\n" + "=" * 70)
    print("PLANNING REPORT")
    print("=" * 70)
    print(planning_report.to_json())
    print("=" * 70)

    print("\n" + "=" * 70)
    print("FASE 3: EXECUTION AGENT")
    print("=" * 70)

    execution_agent = ExecutionAgent(
        planning_repository=planning_repository,
        execution_repository=execution_repository,
        planning_report=planning_report,
        max_llm_retries=settings.execution.max_llm_retries,
        llm_base_url=settings.llm.base_url,
        llm_model_name=settings.llm.model_name,
        llm_temperature=settings.llm.temperature,
        llm_max_tokens=settings.llm.max_tokens,
    )
    execution_report = execution_agent.run()

    print("\n" + "=" * 70)
    print("EXECUTION REPORT")
    print("=" * 70)
    print(execution_report.to_json())
    print("=" * 70)
    print(f"\nSalvato in: {settings.database.absolute_path()}")
    print(f"  - tabella recon_reports (run_id={recon_report.run_id})")
    print(f"  - tabella planning_reports (run_id={planning_report.run_id})")
    print(f"  - tabella execution_reports (run_id={execution_report.run_id})")


if __name__ == "__main__":
    main()
