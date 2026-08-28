"""
Interfacce generiche per la persistenza dei report (ReconReport, PlanningReport).

Astrarre il database dietro interfacce permette di cambiare backend
(SQLite -> Postgres -> MongoDB, ecc.) senza toccare gli agenti.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from db.models import ExecutionReport, PlanningReport, ReconReport


class BaseReconRepository(ABC):
    @abstractmethod
    def save(self, report: ReconReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[ReconReport]:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> List[ReconReport]:
        raise NotImplementedError

    @abstractmethod
    def get_latest(self) -> Optional[ReconReport]:
        raise NotImplementedError


class BasePlanningRepository(ABC):
    @abstractmethod
    def save(self, report: PlanningReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[PlanningReport]:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> List[PlanningReport]:
        raise NotImplementedError

    @abstractmethod
    def get_latest(self) -> Optional[PlanningReport]:
        raise NotImplementedError


class BaseExecutionRepository(ABC):
    @abstractmethod
    def save(self, report: ExecutionReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> Optional[ExecutionReport]:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> List[ExecutionReport]:
        raise NotImplementedError

    @abstractmethod
    def get_latest(self) -> Optional[ExecutionReport]:
        raise NotImplementedError
