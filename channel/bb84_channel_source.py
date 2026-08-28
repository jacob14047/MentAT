"""
Adattatore che collega `BB84SimulationV2` (bb84_simulator_Eve.py)
a `BaseChannelSource`, in modo che `ReconAgent` possa usare il simulatore
senza conoscere i dettagli interni del canale quantistico.
"""
from __future__ import annotations

import json

import numpy as np
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from channel.base_channel_source import BaseChannelSource
from channel.bb84_simulator_Eve import (
    BB84SimulationV2,
    SimulationConfig,
    SecurityAnalyzer,
)


class BB84ChannelSource(BaseChannelSource):
    """
    Adattatore che esegue `BB84SimulationV2` e normalizza i risultati
    nel formato flat dict atteso da `ReconAgent`.

    Utilizzo con il simulatore:

        >>> config = SimulationConfig(
        ...     raw_key_size=5000,
        ...     num_iterations=10,
        ...     interception_rate=0.25,
        ...     eve_attack_enabled=True,
        ...     use_amplitude_damping=True,
        ...     amplitude_damping_gamma=0.15,
        ...     detector_efficiency=0.72,
        ... )
        >>> source = BB84ChannelSource.from_simulation(config, run_id="run_001")
        >>> params = source.get_raw_parameters()
        >>> meta = source.get_metadata()

    Utilizzo con dati esterni (JSON / dict):

        >>> source = BB84ChannelSource.from_dict({"qber": 0.04, ...})
        >>> source = BB84ChannelSource.from_json_file("risultati.json")
    """

    def __init__(
        self,
        raw_parameters: Dict[str, Any],
        metadata: Dict[str, Any],
        simulation_result: Optional[Dict[str, Any]] = None,
        security_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._raw_parameters = raw_parameters
        self._metadata = metadata
        self._simulation_result = simulation_result
        self._security_analysis = security_analysis

    # ------------------------------------------------------------------
    # BaseChannelSource contract
    # ------------------------------------------------------------------

    def get_raw_parameters(self) -> Dict[str, Any]:
        return dict(self._raw_parameters)

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Factory: dati esterni (dict / JSON)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> "BB84ChannelSource":
        """
        Crea un channel source da un dizionario di parametri.

        Se `data` contiene già tutti i campi del formato EXAMPLE_PARAMETERS
        (cli.py), li usa così com'è.
        Altrimenti normalizza un sottoinsieme minimale.
        """
        # Se i dati sembrano provenire dal simulatore, li normalizziamo
        if "qber_est" in data or "sifted_len" in data:
            config = SimulationConfig(
                raw_key_size=data.get("raw_key_size", 4000),
                num_iterations=data.get("num_iterations", 1),
                interception_rate=data.get("interception_rate", 0.0),
                eve_attack_enabled=data.get("eve_attack_enabled", False),
                use_amplitude_damping=data.get("use_amplitude_damping", True),
                amplitude_damping_gamma=data.get("amplitude_damping_gamma", 0.15),
                detector_efficiency=data.get("detector_efficiency", 1.0),
            )
            sim = BB84SimulationV2(config)
            sim_result = sim.run_single_iteration_v2(0)
            return cls.from_simulation_result(sim_result, config, run_id=run_id)

        # Dati già nel formato atteso da EXAMPLE_PARAMETERS
        run_id = run_id or data.get("run_id", "manual_dict")
        metadata = {
            "run_id": run_id,
            "protocol": "BB84",
            "source": "manual_dict",
        }
        raw = {
            "n_qubits_sent": data.get("n_qubits_sent", 10000),
            "sifted_key_length": data.get("sifted_key_length", 5000),
            "raw_key_length": data.get("raw_key_length", 10000),
            "qber": data.get("qber", 0.0),
            "basis_match_rate": data.get("basis_match_rate", 0.5),
            "channel_loss": data.get("channel_loss", 0.0),
            "detector_efficiency": data.get("detector_efficiency", 1.0),
            "dark_count_rate": data.get("dark_count_rate", 0.0),
            "eve_present": data.get("eve_present", False),
            "eve_strategy": data.get("eve_strategy", "none"),
            "eve_interception_rate": data.get("eve_interception_rate", 0.0),
            "eve_detection_probability": data.get("eve_detection_probability", 0.0),
            "error_correction_leakage": data.get("error_correction_leakage", 0),
            "privacy_amplification_ratio": data.get("privacy_amplification_ratio", 1.0),
            "final_key_rate": data.get("final_key_rate", 1.0),
            "protocol_aborted": data.get("protocol_aborted", False),
        }
        return cls(raw, metadata)

    @classmethod
    def from_json_file(
        cls,
        path: str,
        run_id: Optional[str] = None,
    ) -> "BB84ChannelSource":
        """Carica parametri da un file JSON e li passa a `from_dict`."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File JSON non trovato: {resolved.absolute()}")
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, run_id=run_id)

    # ------------------------------------------------------------------
    # Factory: simulatore bb84_simulator_Eve.py
    # ------------------------------------------------------------------

    @classmethod
    def from_simulation(
        cls,
        config: SimulationConfig,
        run_id: Optional[str] = None,
    ) -> "BB84ChannelSource":
        """
        Esegue `BB84SimulationV2` con il dato `config` e restituisce
        un channel source pronto per il Recon Agent.

        Usa get_full_parameters() del simulatore che restituisce TUTTI
        i parametri osservabili: QBER, presenza di Eve, dark counts,
        efficienza detector, strategie di attacco, ecc.
        """
        sim = BB84SimulationV2(config)

        # Esegue tutte le iterazioni
        print(f"[DEBUG from_simulation] PRIMA sim.run() | config.eve_attack_enabled={config.eve_attack_enabled} | config.interception_rate={config.interception_rate}")
        sim.run()
        print(f"[DEBUG from_simulation] DOPO PRIMA sim.run() | sifted_lengths={sim.results['sifted_key_lengths']} | qber_estimates={sim.results['qber_estimates']}")

        # Ottieni TUTTI i parametri dal simulatore
        full_params = sim.get_full_parameters()

        # Security analysis e aggregate come metadata
        security = sim.analyze_security()
        print(f"[DEBUG from_simulation] PRIMA SECONDA sim.run() | aggregate_simulation will be overwritten")
        aggregate = sim.run()
        print(f"[DEBUG from_simulation] DOPO SECONDA sim.run() | sifted_lengths={sim.results['sifted_key_lengths']} | qber_estimates={sim.results['qber_estimates']}")

        run_id = run_id or f"sim_{config.raw_key_size}"
        metadata = {
            "run_id": run_id,
            "protocol": "BB84",
            "source": "bb84_simulator_Eve",
            "config": cls._config_to_dict(config),
            "security_analysis": cls._to_native(security),
            "aggregate_simulation": cls._to_native(aggregate),
        }

        return cls(
            raw_parameters=full_params,
            metadata=metadata,
            simulation_result={"sifted_len": full_params["sifted_key_length"], "qber_est": full_params["qber"]},
            security_analysis=security,
        )

    @classmethod
    def from_simulation_result(
        cls,
        sim_result: Dict[str, Any],
        config: SimulationConfig,
        run_id: Optional[str] = None,
        security_analysis: Optional[Dict[str, Any]] = None,
        aggregate: Optional[Dict[str, Any]] = None,
    ) -> "BB84ChannelSource":
        """
        Normalizza il risultato grezzo di `run_single_iteration_v2`
        nel formato flat dict atteso da `ReconAgent`.
        """
        qber = sim_result.get("qber_est", 0.0)
        sifted_len = sim_result.get("sifted_len", 0)
        raw_key_size = config.raw_key_size
        eve_present = sim_result.get("attack_active", False)
        interception_rate = (
            sim_result.get("eve_params", {}).get("interception_rate", 0.0)
            if sim_result.get("eve_params")
            else 0.0
        )

        # Stima della channel loss basata sul rapporto sifted/raw
        basis_match = sifted_len / raw_key_size if raw_key_size > 0 else 0.0
        channel_loss = 1.0 - basis_match * 2  # 50% sifting efficiency baseline

        # Error correction leakage: stima basata su QBER * sifted key
        error_correction_leakage = int(sifted_len * qber) if qber > 0 else 0

        # Privacy amplification: riduce la chiave in base all'informazione di Eve
        pa_ratio = max(0.0, 1.0 - qber * 2) if qber > 0 else 1.0

        # Final key rate
        final_key_rate = (sifted_len * pa_ratio) / raw_key_size if raw_key_size > 0 else 0.0

        protocol_aborted = bool(qber > 0.11) if not isinstance(qber, float) or qber == qber else False

        true_qber_estimates = sim_result.get("true_qber_estimates", [])
        if true_qber_estimates:
            non_nan = [q for q in true_qber_estimates if q == q]
            true_qber = sum(non_nan) / len(non_nan) if non_nan else 0.0
        else:
            true_qber = sim_result.get("true_qber_est", 0.0)

        raw_parameters = {
            "n_qubits_sent": raw_key_size,
            "sifted_key_length": sifted_len,
            "raw_key_length": raw_key_size,
            "qber": round(qber, 6) if isinstance(qber, float) and qber == qber else 0.0,
            "true_qber": round(true_qber, 6),
            "basis_match_rate": round(basis_match, 6),
            "channel_loss": round(max(0.0, min(1.0, channel_loss)), 6),
            "detector_efficiency": config.detector_efficiency,
            "dark_count_rate": config.thermal_ratio if config.use_thermal_noise else 0.0,
            "eve_present": eve_present,
            "eve_strategy": cls._detect_strategy(sim_result),
            "eve_interception_rate": interception_rate,
            "eve_detection_probability": round(qber, 6) if eve_present else 0.0,
            "error_correction_leakage": error_correction_leakage,
            "privacy_amplification_ratio": round(pa_ratio, 6),
            "final_key_rate": round(final_key_rate, 6),
            "protocol_aborted": protocol_aborted,
            "mean_photon_num": config.mean_photon_num,
            "amplitude_damping_gamma": config.amplitude_damping_gamma if config.use_amplitude_damping else 0.0,
            "distance_km": config.distance_km if config.use_channel_attenuation else 0.0,
        }

        # Aggiungi campi avanzati sempre per consistenza con get_full_parameters()
        raw_parameters["avg_state_purity"] = round(sim_result["avg_purity"], 6) if sim_result.get("avg_purity") is not None else None
        raw_parameters["trojan_leaks"] = sim_result.get("trojan_leaks", 0)
        raw_parameters["blinded_bits"] = sim_result.get("blinded_bits", 0)

        # Popola i campi di security analysis da security_analysis (se fornito)
        if security_analysis:
            raw_parameters["secure_key_length"] = security_analysis.get("secure_key_length", 0)
            raw_parameters["eve_max_information"] = security_analysis.get("eve_max_information", 0)
            raw_parameters["false_positive_rate"] = security_analysis.get("false_positive_rate", 0.0)
            raw_parameters["false_negative_rate"] = security_analysis.get("false_negative_rate", 0.0)
            raw_parameters["detection_power"] = security_analysis.get("detection_power", 0.0)
            raw_parameters["shor_preskill_bound"] = security_analysis.get("shor_preskill_bound", 0.0)

        print(f"[DEBUG from_simulation_result] {len(raw_parameters)} campi | qber={raw_parameters['qber']} | sifted={raw_parameters['sifted_key_length']} | true_qber={raw_parameters['true_qber']} | avg_purity={raw_parameters['avg_state_purity']} | trojan_leaks={raw_parameters['trojan_leaks']} | blinded_bits={raw_parameters['blinded_bits']}")
        print(f"[DEBUG from_simulation_result] security_analysis provided={security_analysis is not None} | secure_key_length={raw_parameters.get('secure_key_length')} | eve_max_info={raw_parameters.get('eve_max_information')}")

        # Converti numpy types a Python nativo per JSON serialization
        safe_security = cls._to_native(security_analysis) if security_analysis else None
        safe_aggregate = cls._to_native(aggregate) if aggregate else None

        metadata = {
            "run_id": run_id or f"sim_{config.raw_key_size}",
            "protocol": "BB84",
            "source": "bb84_simulator_Eve",
            "config": cls._config_to_dict(config),
        }

        if safe_security:
            metadata["security_analysis"] = safe_security
        if safe_aggregate:
            metadata["aggregate_simulation"] = safe_aggregate

        return cls(
            raw_parameters=raw_parameters,
            metadata=metadata,
            simulation_result=sim_result,
            security_analysis=security_analysis,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_strategy(sim_result: Dict[str, Any]) -> str:
        """Estrae la strategia di attacco di Eve dai risultati della simulazione."""
        eve_params = sim_result.get("eve_params") or {}
        if eve_params.get("blinding_attack_active"):
            return "blinding"
        if eve_params.get("pns_enabled"):
            return "pns"
        if eve_params.get("trojan_horse_prob", 0) > 0:
            return "trojan_horse"
        if eve_params.get("interception_rate", 0) > 0:
            return "intercept-resend"
        if sim_result.get("qber_tamper_active"):
            return "qber_tamper"
        return "none"

    @staticmethod
    def _to_native(obj):
        """Converte tipi numpy a Python nativo per JSON serialization."""
        if obj is None:
            return None
        # Controlla numpy types PRIMA di Python types (np.bool_ e' subclass di bool)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Python nativo
        if type(obj) is bool:
            return obj
        if isinstance(obj, (int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {k: BB84ChannelSource._to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BB84ChannelSource._to_native(v) for v in obj]
        return obj

    @staticmethod
    def _config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
        """Converte `SimulationConfig` in dict serializzabile."""
        if is_dataclass(config):
            from dataclasses import fields as dc_fields

            return {
                f.name: getattr(config, f.name)
                for f in dc_fields(config)
            }
        return dict(config)
