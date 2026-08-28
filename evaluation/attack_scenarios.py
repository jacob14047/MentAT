"""
Scenari d'attacco predefiniti per la valutazione BB84.

Ogni scenario definisce:
- description: Descrizione testuale per report
- params: Parametri del simulatore (eve attack config)
- channel_params: Parametri del canale (damping, depolarization, ecc.)
- is_stealth: True se l'attacco cerca di nascondersi
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class AttackScenario:
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    channel_params: Dict[str, Any] = field(default_factory=dict)
    is_stealth: bool = False
    category: str = "general"


# ============================================================================
# Scenari d'attacco
# ============================================================================

SCENARIOS: Dict[str, AttackScenario] = {
    "intercept_resend": AttackScenario(
        description="Attacco Intercept-Resend base: Eve misura e rispedisce qubit",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.25,
            "eve_pns_enabled": False,
            "blinding_attack_active": False,
            "trojan_horse_prob": 0.0,
            "qber_tamper_active": False,
        },
        channel_params={
            "depolarization_prob": 0.0,
            "amplitude_damping_gamma": 0.15,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
        },
        is_stealth=False,
        category="basic",
    ),
    "intercept_resend_stealth": AttackScenario(
        description="Attacco Intercept-Resend ottimizzato: interception_rate basso per minimizzare QBER",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.15,
            "eve_pns_enabled": False,
            "blinding_attack_active": False,
            "trojan_horse_prob": 0.0,
            "qber_tamper_active": False,
        },
        channel_params={
            "depolarization_prob": 0.0,
            "amplitude_damping_gamma": 0.10,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
        },
        is_stealth=True,
        category="basic",
    ),
    "pns": AttackScenario(
        description="PNS attack (Photon Number Splitting): Eve blocca pulsazioni a singolo fotone",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.0,
            "eve_pns_enabled": True,
            "eve_pns_block_ratio": 0.5,
            "blinding_attack_active": False,
            "trojan_horse_prob": 0.0,
            "qber_tamper_active": False,
        },
        channel_params={
            "depolarization_prob": 0.0,
            "amplitude_damping_gamma": 0.15,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
            "mean_photon_num": 0.4,
        },
        is_stealth=True,
        category="photon",
    ),
    "blinding": AttackScenario(
        description="Blinding attack + QBER tampering: Eve controlla completamente il canale",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.3,
            "eve_pns_enabled": False,
            "blinding_attack_active": True,
            "trojan_horse_prob": 0.0,
            "qber_tamper_active": True,
            "qber_tamper_amount": 0.0,
        },
        channel_params={
            "depolarization_prob": 0.0,
            "amplitude_damping_gamma": 0.15,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
        },
        is_stealth=True,
        category="detector",
    ),
    "trojan_horse": AttackScenario(
        description="Trojan Horse attack: Eve inietta luce per rivelare la base di Alice",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.2,
            "eve_pns_enabled": False,
            "blinding_attack_active": False,
            "trojan_horse_prob": 0.3,
            "qber_tamper_active": False,
        },
        channel_params={
            "depolarization_prob": 0.0,
            "amplitude_damping_gamma": 0.15,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
            "optical_isolator_efficiency": 0.95,
        },
        is_stealth=True,
        category="optical",
    ),
    "qber_tamper": AttackScenario(
        description="QBER tampering: Eve altera la stima QBER per evitare il rilevamento",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.15,
            "eve_pns_enabled": False,
            "blinding_attack_active": False,
            "trojan_horse_prob": 0.0,
            "qber_tamper_active": True,
            "qber_tamper_amount": 0.0,
        },
        channel_params={
            "depolarization_prob": 0.0,
            "amplitude_damping_gamma": 0.10,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
        },
        is_stealth=True,
        category="detection",
    ),
    "mixed": AttackScenario(
        description="Attacco randomizzato: combina multiple strategie con probabilità per iterazione",
        params={
            "eve_attack_enabled": True,
            "interception_rate": 0.2,
            "eve_pns_enabled": True,
            "eve_pns_block_ratio": 0.4,
            "blinding_attack_active": True,
            "trojan_horse_prob": 0.2,
            "qber_tamper_active": True,
            "qber_tamper_amount": 0.0,
        },
        channel_params={
            "depolarization_prob": 0.01,
            "amplitude_damping_gamma": 0.15,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
        },
        is_stealth=True,
        category="mixed",
    ),
    "clean_baseline": AttackScenario(
        description="Baseline: nessun attacco Eve, solo rumore di canale e detector",
        params={
            "eve_attack_enabled": False,
            "interception_rate": 0.0,
            "eve_pns_enabled": False,
            "blinding_attack_active": False,
            "trojan_horse_prob": 0.0,
            "qber_tamper_active": False,
        },
        channel_params={
            "depolarization_prob": 0.01,
            "amplitude_damping_gamma": 0.15,
            "use_amplitude_damping": True,
            "use_phase_damping": False,
            "use_thermal_noise": True,
            "thermal_ratio": 0.0004,
        },
        is_stealth=False,
        category="baseline",
    ),
}


def get_all_scenarios() -> Dict[str, AttackScenario]:
    """Restituisce una copia di tutti gli scenari."""
    return SCENARIOS.copy()


def get_scenarios_by_category(category: str) -> Dict[str, AttackScenario]:
    """Filtra scenari per categoria."""
    return {k: v for k, v in SCENARIOS.items() if v.category == category}


def get_attack_scenarios() -> Dict[str, AttackScenario]:
    """Restituisce solo scenari con attacco attivo (esclude baseline)."""
    return {k: v for k, v in SCENARIOS.items() if k != "clean_baseline"}


def get_baseline_scenarios() -> Dict[str, AttackScenario]:
    """Restituisce solo scenari baseline."""
    return {k: v for k, v in SCENARIOS.items() if k == "clean_baseline"}
