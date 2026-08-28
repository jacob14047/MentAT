"""
Classe base generica per un "template di prompt".

Separare i prompt dal codice degli agenti permette di:
- versionare/modificare i prompt senza toccare la logica Python,
- riusare lo stesso agente con prompt diversi (es. A/B testing),
- avere, come richiesto, un file dedicato per ogni prompt specifico.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

Message = Dict[str, str]


class BasePromptTemplate(ABC):
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Istruzioni di ruolo/sistema per il modello."""
        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Costruisce il messaggio utente a partire dal contesto (es. parametri del canale)."""
        raise NotImplementedError

    def render(self, context: Dict[str, Any]) -> List[Message]:
        """Produce la lista di messaggi in stile chat pronta per l'LLM."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(context)},
        ]
