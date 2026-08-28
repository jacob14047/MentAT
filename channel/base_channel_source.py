"""
Interfaccia astratta per le sorgenti di parametri del canale BB84.

Tutte le implementazioni (simulatore, file JSON, dati di esempio) devono
estendere questa classe in modo che `ReconAgent` possa interagire con esse
in modo uniforme.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseChannelSource(ABC):
    """Contratto per qualsiasi sorgente di parametri del canale BB84."""

    @abstractmethod
    def get_raw_parameters(self) -> Dict[str, Any]:
        """Restituisce i parametri grezzi osservati dal canale (QBER, efficienza, ecc.)."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Restituisce i metadati della run (protocollo, run_id, etc.)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], run_id: str | None = None) -> "BaseChannelSource":
        """Crea un'istanza da un dizionario di parametri."""

    @classmethod
    @abstractmethod
    def from_json_file(cls, path: str, run_id: str | None = None) -> "BaseChannelSource":
        """Crea un'istanza da un file JSON."""
