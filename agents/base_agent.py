"""
Classe base generica per un agente LLM.

Non sa nulla di BB84: sa solo come combinare un `BaseLLMClient` e un
`BasePromptTemplate` per ottenere un'analisi strutturata (JSON), con un
numero limitato di retry se il modello locale produce un output non
parsabile. Ogni agente specifico (Recon, in futuro Planning/Executor)
eredita da qui e implementa `run(...)`.

Tracking per valutazione RQ3:
- llm_call_count: numero di chiamate LLM
- handoff_log: registro degli handoff tra agenti
- timing: timing delle operazioni
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from llm.base_client import BaseLLMClient
from prompts.base_prompt import BasePromptTemplate
from utils.json_utils import JSONExtractionError
from utils.logger import get_logger


class BaseAgent(ABC):
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_template: BasePromptTemplate,
        max_llm_retries: int = 2,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.max_llm_retries = max_llm_retries
        self.logger = get_logger(self.__class__.__name__)
        
        # Tracking per valutazione RQ3
        self._llm_call_count: int = 0
        self._handoff_log: List[Dict[str, Any]] = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0
    
    def _call_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Costruisce i messaggi dal prompt template e chiama l'LLM, con retry."""
        messages = self.prompt_template.render(context)

        last_error: Exception | None = None
        for attempt in range(1, self.max_llm_retries + 2):
            try:
                self.logger.info("Chiamata all'LLM (tentativo %d)...", attempt)
                result = self.llm_client.chat_json(messages)
                self._llm_call_count += 1
                return result
            except (JSONExtractionError, ConnectionError, ValueError) as exc:
                last_error = exc
                self.logger.warning("Tentativo %d fallito: %s", attempt, exc)

        raise RuntimeError(
            f"L'LLM non ha prodotto un JSON valido dopo "
            f"{self.max_llm_retries + 1} tentativi."
        ) from last_error
    
    def record_handoff(self, target_agent: str, context_type: str) -> None:
        """Registra un handoff tra agenti."""
        self._handoff_log.append({
            'from_agent': self.__class__.__name__,
            'to_agent': target_agent,
            'context_type': context_type,
            'timestamp': time.time(),
        })
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Restituisce il riepilogo del tracking per valutazione RQ3."""
        elapsed = (self._end_time - self._start_time) if (self._start_time and self._end_time) else 0.0
        return {
            'agent_name': self.__class__.__name__,
            'llm_calls': self._llm_call_count,
            'handoff_count': len(self._handoff_log),
            'handoff_log': self._handoff_log,
            'execution_time_sec': elapsed,
        }
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Restituisce il conteggio token dell'ultima risposta LLM per questo agente."""
        usage = self.llm_client.last_usage
        if usage is None:
            return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
        }
    
    def reset_tracking(self) -> None:
        """Resetta il tracking per una nuova esecuzione."""
        self._llm_call_count = 0
        self._handoff_log = []
        self._start_time = 0.0
        self._end_time = 0.0
    
    def start_timing(self) -> None:
        """Avvia il timer."""
        self._start_time = time.time()
    
    def stop_timing(self) -> None:
        """Ferma il timer."""
        self._end_time = time.time()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
