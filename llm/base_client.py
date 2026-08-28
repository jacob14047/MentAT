"""
Interfaccia generica per un client LLM.

Qualunque backend (LM Studio, Ollama, OpenAI, ecc.) deve implementare questa
interfaccia, cosi' gli agenti non dipendono mai da un provider specifico.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import logging

from utils.json_utils import extract_json_object

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class BaseLLMClient(ABC):
    """Contratto generico: manda messaggi in stile chat, ricevi testo."""

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        """Invia una conversazione al modello e restituisce il testo di risposta."""
        raise NotImplementedError

    @property
    @abstractmethod
    def last_usage(self) -> Optional[Dict[str, Any]]:
        """Restituisce i token usage dell'ultima chiamata (se disponibile)."""
        raise NotImplementedError

    def chat_json(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        """
        Variante di `chat` che si aspetta un oggetto JSON come risposta.

        Comportamento comune a tutti i provider: chiama `chat`, poi estrae e
        valida il JSON in modo tollerante (i modelli locali a volte aggiungono
        testo o code fence attorno al JSON).
        """
        raw_text = self.chat(messages, **kwargs)
        logger.info("Risposta grezza dall'LLM (len=%d): %s", len(raw_text), raw_text[:500])
        try:
            return extract_json_object(raw_text)
        except Exception:
            logger.error("JSON extraction fallita. Risposta completa:\n%s", raw_text)
            raise
