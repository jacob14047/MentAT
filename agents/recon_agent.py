"""
Recon Agent.

Obiettivo (come da specifica): raccogliere il maggior numero possibile di
parametri osservabili dal canale BB84, esattamente come farebbe Eve, per poi
stimare la sua liberta' di azione e fornire il contesto al futuro Planning
Agent.

Flusso:
  1. Legge i parametri grezzi dalla `BaseChannelSource` (channel/).
  2. Li passa all'LLM (via LM Studio) insieme al prompt di ruolo "Eve/recon".
  3. Valida/parsa l'analisi JSON restituita.
  4. Costruisce un `ReconReport` e lo salva nel repository (db/).
"""
from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseAgent
from channel.base_channel_source import BaseChannelSource
from db.base_repository import BaseReconRepository
from db.models import ReconReport
from llm.base_client import BaseLLMClient
from prompts.recon_prompt import ReconPromptTemplate


class ReconAgent(BaseAgent):
    def __init__(
        self,
        llm_client: BaseLLMClient,
        channel_source: BaseChannelSource,
        repository: Optional[BaseReconRepository] = None,
        qber_abort_threshold: float = 0.11,
        max_llm_retries: int = 2,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            prompt_template=ReconPromptTemplate(),
            max_llm_retries=max_llm_retries,
        )
        self.channel_source = channel_source
        self.repository = repository
        self.qber_abort_threshold = qber_abort_threshold

    def run(self) -> ReconReport:
        self.logger.info("Raccolta parametri grezzi dal canale BB84...")
        raw_parameters = self.channel_source.get_raw_parameters()
        metadata = self.channel_source.get_metadata()

        self.logger.info(
            "Parametri raccolti (%d campi). Avvio analisi LLM come Eve...",
            len(raw_parameters),
        )
        print(f"\n[DEBUG ReconAgent] raw_parameters ({len(raw_parameters)} campi): {list(raw_parameters.keys())}")
        print(f"[DEBUG ReconAgent] qber={raw_parameters.get('qber')} | true_qber={raw_parameters.get('true_qber')} | sifted_key_length={raw_parameters.get('sifted_key_length')} | n_qubits_sent={raw_parameters.get('n_qubits_sent')}")
        print(f"[DEBUG ReconAgent] eve_present={raw_parameters.get('eve_present')} | eve_strategy={raw_parameters.get('eve_strategy')} | eve_interception_rate={raw_parameters.get('eve_interception_rate')}")
        print(f"[DEBUG ReconAgent] detector_efficiency={raw_parameters.get('detector_efficiency')} | dark_count_rate={raw_parameters.get('dark_count_rate')} | channel_loss={raw_parameters.get('channel_loss')}")
        print(f"[DEBUG ReconAgent] mean_photon_num={raw_parameters.get('mean_photon_num')} | distance_km={raw_parameters.get('distance_km')} | amplitude_damping_gamma={raw_parameters.get('amplitude_damping_gamma')}")
        print(f"[DEBUG ReconAgent] avg_state_purity={raw_parameters.get('avg_state_purity')} | secure_key_length={raw_parameters.get('secure_key_length')} | eve_max_information={raw_parameters.get('eve_max_information')}")
        context = {
            "raw_parameters": raw_parameters,
            "metadata": metadata,
            "qber_abort_threshold": self.qber_abort_threshold,
        }
        analysis = self._call_llm(context)

        report = ReconReport(
            run_id=metadata.get("run_id", "unknown_run"),
            protocol=metadata.get("protocol", "BB84"),
            raw_parameters=raw_parameters,
            llm_analysis=analysis,
        )

        if self.repository is not None:
            self.logger.info("Salvataggio del report nel database (run_id=%s)...", report.run_id)
            self.repository.save(report)

        self.logger.info(
            "Recon completata. freedom_of_action_score=%s, confidence=%s",
            report.freedom_of_action_score,
            report.confidence,
        )
        return report
