import numpy as np
import secrets
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
import time



# ============================================================================
# Utility functions
# ============================================================================

def random_bit() -> int:
    """Cryptographically secure random bit."""
    return secrets.randbits(1)


def random_bits(n: int) -> np.ndarray:
    """Array of n random bits (using numpy for efficiency)."""
    return np.random.randint(0, 2, size=n, dtype=np.uint8)


def random_bases(n: int) -> np.ndarray:
    """0 = rectilinear (Z), 1 = diagonal (X)."""
    return np.random.randint(0, 2, size=n, dtype=np.uint8)


# ============================================================================
# Enums and data structures
# ============================================================================

class Basis(Enum):
    Z = 0  # rectilinear (horizontal/vertical)
    X = 1  # diagonal (45°/135°)


# ============================================================================
# Security analysis module
# ============================================================================

class VisualizationSuite:
    """Grafici per canale BB84 pulito (senza Eve)."""

    def plot_qber_vs_noise(self, iterations: int = 30):
        """QBER al variare della depolarizzazione."""
        depols = np.linspace(0, 0.15, 20)
        qbers = []

        for p in depols:
            config = SimulationConfig(
                raw_key_size=4000,
                num_iterations=1,
                depolarization_prob=p,
                use_amplitude_damping=True,
                amplitude_damping_gamma=0.1,
                use_phase_damping=True,
                phase_damping_lambda=0.05,
                track_state_purities=False
            )
            sim = BB84SimulationV2(config)
            result = sim.run_single_iteration_v2(0)
            qbers.append(result['qber_est'] if result['qber_est'] == result['qber_est'] else 0)

        plt.figure(figsize=(10, 6))
        plt.plot(depols * 100, np.array(qbers) * 100, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Depolarization Probability (%)', fontsize=12)
        plt.ylabel('QBER (%)', fontsize=12)
        plt.title('QBER vs Depolarization (clean channel, no Eve)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('qber_vs_noise.pdf', dpi=300)
        plt.show()
        print("[OK] Salvato: qber_vs_noise.pdf")

    def plot_purity_vs_gamma(self):
        """Purità media degli stati al variare di amplitude damping γ."""
        gammas = np.linspace(0, 0.5, 20)
        purities = []

        for g in gammas:
            config = SimulationConfig(
                raw_key_size=1000,
                num_iterations=1,
                use_amplitude_damping=True,
                amplitude_damping_gamma=g,
                track_state_purities=True
            )
            sim = BB84SimulationV2(config)
            result = sim.run_single_iteration_v2(0)
            purities.append(result['avg_purity'] if result['avg_purity'] else 1.0)

        plt.figure(figsize=(10, 6))
        plt.plot(gammas, purities, 's-', linewidth=2, markersize=8, color='purple')
        plt.xlabel('Amplitude Damping γ', fontsize=12)
        plt.ylabel('Average State Purity', fontsize=12)
        plt.title('State Purity Degradation vs Channel Loss', fontsize=14)
        plt.ylim(0.5, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('purity_vs_gamma.pdf', dpi=300)
        plt.show()
        print("[OK] Salvato: purity_vs_gamma.pdf")

    def plot_key_distribution(self, n_iterations: int = 100):
        """Distribuzione della lunghezza della sifted key su N iterazioni."""
        config = SimulationConfig(
            raw_key_size=4000,
            num_iterations=n_iterations,
            use_amplitude_damping=True,
            amplitude_damping_gamma=0.1,
            use_phase_damping=True,
            phase_damping_lambda=0.05,
            depolarization_prob=0.01,
            track_state_purities=False
        )
        sim = BB84SimulationV2(config)
        sim.run_single_iteration_v2(0)  # Run all iterations to collect key lengths

        key_lengths = sim.results['sifted_key_lengths']
        mean_len = np.mean(key_lengths)

        plt.figure(figsize=(10, 6))
        plt.hist(key_lengths, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(mean_len, color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_len:.0f} bits')
        plt.xlabel('Sifted Key Length (bits)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Sifted Key Length Distribution ({n_iterations} iterations)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('key_distribution.pdf', dpi=300)
        plt.show()
        print(f"[OK] Salvato: key_distribution.pdf  |  Mean: {mean_len:.0f} bits")

    def plot_qber_vs_distance(self):
        """QBER al variare della distanza (tramite amplitude damping realistico)."""
        distances_km = np.linspace(0, 50, 20)
        # 0.2 dB/km → trasmittanza = 10^(-0.2d/10), gamma = 1 - trasmittanza
        gammas = 1 - 10 ** (-0.2 * distances_km / 10)
        qbers = []

        for g in gammas:
            config = SimulationConfig(
                raw_key_size=2000,
                num_iterations=1,
                use_amplitude_damping=True,
                amplitude_damping_gamma=float(g),
                depolarization_prob=0.01,
                track_state_purities=False
            )
            sim = BB84SimulationV2(config)
            result = sim.run_single_iteration_v2(0)
            qbers.append(result['qber_est'] if result['qber_est'] == result['qber_est'] else 0)

        plt.figure(figsize=(10, 6))
        plt.plot(distances_km, np.array(qbers) * 100, 'o-', linewidth=2,
                 markersize=8, color='darkorange')
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('QBER (%)', fontsize=12)
        plt.title('QBER vs Fiber Distance (0.2 dB/km, no Eve)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('qber_vs_distance.pdf', dpi=300)
        plt.show()
        print("✅ Salvato: qber_vs_distance.pdf")

class SecurityAnalyzer:
    """
    Analisi rigorosa della sicurezza usando risultati provati.
    """
    
    def __init__(self, n_raw: int, qber: float, sharing_rate: float):
        """
        n_raw: numero di qubit grezzi
        qber: Quantum Bit Error Rate stimato
        sharing_rate: frazione di chiave rivelata per test
        """
        self.N = n_raw
        self.e = qber  # ε in literature
        self.f = sharing_rate
    
    @staticmethod
    def shor_preskill_bound(e_qber: float, threshold: float = 0.11) -> float:
        """
        Bound di Shor-Preskill (2000):
        
        Se QBER > h(threshold) dove h(x) = -x log₂(x) - (1-x) log₂(1-x)
        allora Eve non può avere ottenuto più di una certa informazione.
        
        Returns: Upper bound su Eve's information (bits)
        """
        import math
        
        def binary_entropy(x):
            if x <= 0 or x >= 1:
                return 0
            return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
        
        # Se e_qber è basso, Eve ha info limitata
        h_e = binary_entropy(e_qber)
        
        # Information Eve could have gained (theoretical bound)
        # I_eve ≤ n_sifted * h(QBER)
        return h_e  # Per bit
    
    def calculate_secure_key_length(self, n_sifted: int = None) -> Tuple[int, float]:
        """
        Calcola lunghezza della chiave finale dopo:
        1. Error correction
        2. Privacy amplification
        
        Basato su: Bennett, Brassard, Crépeau, Maurer (1995)
        
        l ≥ d - 2 log₂(1/ε_sec) - 2
        dove:
          d = number of sifted bits
          ε_sec = security parameter (e.g., 2^-128)
          
        Args:
            n_sifted: numero effettivo di bit siftati. Se None, usa self.N * 0.5.
        """
        if n_sifted is None:
            n_sifted = int(self.N * 0.5)  # 50% sifting efficiency
        
        if n_sifted <= 0:
            return (0, 0.0)
        
        n_tested = int(n_sifted * self.f)  # Bits tested for QBER
        n_private = n_sifted - n_tested  # Bits for final key
        
        # Privacy amplification overhead
        eps_sec = 1e-30  # 2^-128 security level
        privacy_amp_cost = 2 * np.log2(1 / eps_sec) + 2
        
        secure_length = max(0, n_private - privacy_amp_cost)
        
        # Information Eve could have
        eve_info = self.shor_preskill_bound(self.e)
        if not np.isfinite(eve_info):
            eve_info = 0.0
        eve_bits = int(n_tested * eve_info)
        
        return int(secure_length), eve_bits
    
    def test_hypothesis(self) -> Dict[str, float]:
        """
        Test statistico: Qual è il tasso di falso positivo/negativo?
        
        Usa Chernoff bound per quantificare:
        - P(reject|no eve) = False positive rate
        - P(accept|eve present) = False negative rate
        """
        n_tested = int(self.N * 0.5 * self.f)
        
        # H0: No Eve (p = 0.05, just noise)
        p0 = 0.05
        
        # H1: Eve (p = 0.12, noise + Eve)
        p1 = 0.12
        
        threshold = 0.11
        threshold_count = threshold * n_tested
        
        # Chernoff bound for false negatives
        # P(QBER < threshold | Eve) ≤ exp(-D(p1||threshold))
        def kullback_leibler(p, q):
            return p * np.log2(p/q) + (1-p) * np.log2((1-p)/(1-q))
        
        kl_eve = kullback_leibler(p1, threshold)
        false_neg_bound = 2 ** (-n_tested * kl_eve)  # Prob Eve not detected
        
        # False positives
        kl_clean = kullback_leibler(p0, threshold)
        false_pos_bound = 2 ** (-n_tested * kl_clean)
        
        return {
            'false_positive_rate': float(false_pos_bound),
            'false_negative_rate': float(false_neg_bound),
            'security_gap': threshold - p0,  # Margin from noise
            'detection_power': 1 - false_neg_bound
        }


@dataclass
class BlochVector:
    """
    Rappresenta uno stato di qubit sulla Bloch sphere.
    
    Ogni stato puro di un qubit può essere scritto come:
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    Sulla Bloch sphere:
    x = sin(θ)cos(φ)
    y = sin(θ)sin(φ)
    z = cos(θ)
    
    Punti speciali:
    - (0,0,1):   |0⟩ (North pole)
    - (0,0,-1):  |1⟩ (South pole)
    - (1,0,0):   |+⟩ = (|0⟩+|1⟩)/√2
    - (-1,0,0):  |-⟩ = (|0⟩-|1⟩)/√2
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0  # Default: |0⟩
    
    def __post_init__(self):
        pass
        

    def to_array(self) -> np.ndarray:
        """Ritorna il vettore come array numpy."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_basis_and_bit(cls, basis: int, bit: int) -> 'BlochVector':
        """
        Crea uno stato di Bloch da una base e un bit.
        
        Parametri:
        -----------
        basis : int
            0 = Z basis (rectilinear: |0⟩, |1⟩)
            1 = X basis (diagonal: |+⟩, |-⟩)
        
        bit : int
            0 o 1
        
        Returns:
        --------
        BlochVector instance
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩ in Z basis
        >>> print(state)
        BlochVector(x=0.0, y=0.0, z=-1.0)
        """
        if basis == 0:  # Z basis
            # |0⟩ -> (0, 0, 1),  |1⟩ -> (0, 0, -1)
            return cls(0, 0, 1 if bit == 0 else -1)
        else:  # X basis (basis == 1)
            # |+⟩ -> (1, 0, 0),  |-⟩ -> (-1, 0, 0)
            return cls(1 if bit == 0 else -1, 0, 0)

    @classmethod
    def z_basis_0(cls) -> 'BlochVector':
        return cls(0.0, 0.0, 1.0)

    @classmethod
    def z_basis_1(cls) -> 'BlochVector':
        return cls(0.0, 0.0, -1.0)

    @classmethod
    def x_basis_plus(cls) -> 'BlochVector':
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def x_basis_minus(cls) -> 'BlochVector':
        return cls(-1.0, 0.0, 0.0)

    def measure(self, measurement_basis: int) -> Tuple[int, float]:
        """
        Misura lo stato quantico in una base specificata.
        
        Usa la regola di Born: P(outcome|basis) = |⟨outcome|ψ⟩|²
        
        Sulla Bloch sphere: P(outcome) = (1 + ⟨ψ|M_outcome|ψ⟩) / 2
                           dove M_outcome è il proiettore sull'outcome
        
        Parametri:
        ----------
        measurement_basis : int
            0 = Misura in Z basis
            1 = Misura in X basis
        
        Returns:
        --------
        measured_bit : int (0 o 1)
        probability : float (probabilità dell'outcome ottenuto)
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩
        >>> bit, prob = state.measure(0)  # Misura in Z basis
        >>> print(f"Measured: {bit}, Probability: {prob:.2f}")
        Measured: 1, Probability: 1.00
        
        >>> bit, prob = state.measure(1)  # Misura in X basis (base sbagliata!)
        >>> print(f"Measured: {bit}, Probability: {prob:.2f}")
        Measured: 0 or 1, Probability: 0.50  # Random!
        """
        if measurement_basis == 0:  # Z basis
            # Proiettore su |0⟩: (0, 0, 1)
            # Proiettore su |1⟩: (0, 0, -1)
            measurement_vector = np.array([0, 0, 1])
        else:  # X basis (measurement_basis == 1)
            # Proiettore su |+⟩: (1, 0, 0)
            # Proiettore su |-⟩: (-1, 0, 0)
            measurement_vector = np.array([1, 0, 0])
        
        # Born rule: p(0) = (1 + ⟨measurement_vector · bloch⟩) / 2
        state_vector = self.to_array()
        overlap = np.dot(state_vector, measurement_vector)
        
        # Probabilità di misurare |0⟩ nella base specificata
        p_zero = 0.5 * (1.0 + overlap)
        
        # Campiona da Bernoulli(p_zero)
        measured_bit = 0 if np.random.random() < p_zero else 1
        
        # Probabilità dell'outcome ottenuto
        probability = p_zero if measured_bit == 0 else (1.0 - p_zero)
        
        # COLLASSO della funzione d'onda
        # Dopo la misura, lo stato collassa al eigenstate misurato
        if measured_bit == 0:
            # Collassa a |0⟩ nella base di misura
            self.x = measurement_vector[0]
            self.y = measurement_vector[1]
            self.z = measurement_vector[2]
        else:
            # Collassa a |1⟩ nella base di misura
            self.x = -measurement_vector[0]
            self.y = -measurement_vector[1]
            self.z = -measurement_vector[2]
        
        return measured_bit, probability
    
    def apply_depolarization(self, p: float) -> None:
        """
        Applica rumore di depolarizzazione.
        
        Canale di depolarizzazione: ρ' = (1-p)ρ + p·I/2
        
        Sulla Bloch sphere, questo equivale a rimescolare lo stato
        con il centro della sfera (stato misto I/2):
        bloch_vector' = (1-p) * bloch_vector
        
        Parametri:
        ----------
        p : float
            Probabilità di depolarizzazione (0 ≤ p ≤ 1)
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 0)  # |0⟩
        >>> state.apply_depolarization(0.1)  # 10% depolarization
        >>> print(state)  # Lo stato si avvicina al centro
        BlochVector(x=0.0, y=0.0, z=0.9)
        """
        scale_factor = 1.0 - p
        self.x *= scale_factor
        self.y *= scale_factor
        self.z *= scale_factor
    
    def apply_amplitude_damping(self, gamma: float) -> None:
        """
        Applica amplitude damping (perdita di energia).
        
        Questo è il modello più realistico per canali quantistici reali:
        - Fotoni persi nell'atmosfera o nella fibra ottica
        - Energia che si dissipa
        
        Kraus operators:
        K₀ = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        K₁ = √γ|0⟩⟨1|
        
        Sulla Bloch sphere (per stati puri):
        x' = (1-γ)x
        y' = (1-γ)y
        z' = 2γ(1/2 + z/2) - 1 = γ(1+z) - 1
        
        Parametri:
        ----------
        gamma : float
            Decay rate (0 ≤ γ ≤ 1)
            Per fibra ottica a 10 km e 0.2 dB/km: γ ≈ 0.15-0.20
        """
        # Applica gli effetti
        self.x *= np.sqrt(1 - gamma)
        self.y *= np.sqrt(1 - gamma)
        # z si muove verso -1 (stato |1⟩)
        self.z = (1 - gamma) * self.z + gamma  # si muove verso +1 (|0⟩, ground state)
    
    def apply_phase_damping(self, lambda_param: float) -> None:
        """
        Applica phase damping (dephasing).
        
        Questo modella la perdita di coerenza quantistica:
        - Interazione con l'ambiente
        - Fluttuazioni di fase
        - Decoherence
        
        Kraus operators:
        K₀ = |0⟩⟨0| + √(1-λ)|1⟩⟨1|
        K₁ = √λ|1⟩⟨1|
        
        Sulla Bloch sphere:
        x' = (1-λ)x
        y' = (1-λ)y
        z' = z  (no change in z)
        
        Parametri:
        ----------
        lambda_param : float
            Dephasing rate
        """
        self.x *= (1 - lambda_param)
        self.y *= (1 - lambda_param)
        # z non cambia
    
    def distance_from_ideal(self, ideal_basis: int, ideal_bit: int) -> float:
        """
        Calcola la distanza di traccia (trace distance) dallo stato ideale.
        
        Misura quanto lo stato attuale è "lontano" da uno stato puro ideale.
        
        D = 1/2 ||ρ - σ||₁
        
        Su Bloch sphere: D = 1/2 ||bloch_vector - ideal_vector||
        
        Returns:
        --------
        distance : float (0 ≤ distance ≤ 1)
            0 = stato identico all'ideale
            1 = stato completamente ortogonale
        """
        ideal_state = BlochVector.from_basis_and_bit(ideal_basis, ideal_bit)
        diff = self.to_array() - ideal_state.to_array()
        return 0.5 * np.linalg.norm(diff)
    
    def purity(self) -> float:
        """
        Calcola la purità dello stato: Tr(ρ²).
        
        Su Bloch sphere: P = (1 + ||bloch||²) / 2
        
        Returns:
        --------
        purity : float (0 ≤ P ≤ 1)
            1 = stato puro
            1/2 = stato completamente misto
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 0)  # |0⟩ puro
        >>> print(f"Purity: {state.purity():.2f}")  # ~1.0
        Purity: 1.00
        
        >>> state.apply_depolarization(0.5)
        >>> print(f"Purity: {state.purity():.2f}")  # ~0.75
        Purity: 0.75
        """
        r_squared = self.x**2 + self.y**2 + self.z**2
        return (1.0 + r_squared) / 2.0
    
    def plot_bloch_sphere(self, title: str = "Bloch Sphere") -> None:
        """
        Visualizza lo stato sulla Bloch sphere (se matplotlib disponibile).
        
        Example:
        --------
        >>> state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩
        >>> state.plot_bloch_sphere(title="State |1⟩")
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sfera unitaria (sfondo)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        # Vettore dello stato
        ax.quiver(0, 0, 0, self.x, self.y, self.z, color='red', arrow_length_ratio=0.1, linewidth=2)
        
        # Assi
        ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', alpha=0.3, linewidth=1)
        ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', alpha=0.3, linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', alpha=0.3, linewidth=1)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X (|+⟩)')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (|0⟩)')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig


@dataclass
class SimulationConfig:
    """Main configuration for the BB84 simulation."""
    
    raw_key_size: int = 4000          # Number of photons sent (N)
    num_iterations: int  = 10         # Number of key distribution rounds
    depolarization_prob: float = 0.0  # p: total depolarization probability
    sharing_rate: float = 0.2         # f: fraction of sifted key shared for QBER estimation
    use_amplitude_damping: bool = True
    amplitude_damping_gamma: float = 0.15
    use_phase_damping: bool = False
    phase_damping_lambda: float = 0.05
    track_state_purities: bool = True
    use_thermal_noise: bool = False
    thermal_ratio: float = 0.02          # Dark counts, thermal photons
    fiber_loss_db_per_km: float = 0.2    # Standard ITU: 0.2 dB/km @ 1550nm
    

    
    use_weak_laser: bool = False      # Weak coherent pulse source
    mean_photon_num: float = 0.1      # μ: average photons per pulse
    source_frequency_hz: float = 1e6  # f: pulse generation rate (Hz)
    use_channel_attenuation: bool = False
    attenuation_coeff: float = 0.2    # dB/km
    distance_km: float = 0.0
    detector_efficiency: float = 1.0  # η_D
    use_dead_time: bool = True
    dead_time_us: float = 0.01      # τ in microseconds
    
    interception_rate: float = 0.0      # ε: frazione di fotoni intercettati (0..1)
    eve_attack_enabled: bool = False   # Abilita/disabilita Eve globalmente
    eve_pns_enabled: bool = False       # Photon Number Splitting attack
    eve_pns_block_ratio: float = 0.5    # Frazione di pulse a singolo fotone da bloccare (PNS)

    # Attacchi avanzati Eve
    blinding_attack_active: bool = False   # Blinding: Eve forza bob_bits = eve_bit
    trojan_horse_prob: float = 0.0         # Probabilità per qubit di bypassare l'isolatore ottico
    optical_isolator_efficiency: float = 0.95  # Difesa Alice: attenuazione del Trojan Horse
    qber_tamper_active: bool = False       # Eve altera bob_share / qber_est prima che Alice legga

    # Advanced
    research_mode: bool = True       # Enable remaining key module (shares usable key for analysis)
    
    # ========== Opzionale: random attacks ==========
    use_random_attacks: bool = False    # Randomizza quali iterazioni sono attaccate
    attack_rate: float = 0.5            # Probabilità che una iterazione sia attaccata

    # ========== QBER tampering amount ==========
    qber_tamper_amount: float = 0.0     # Valore a cui Eve zera/altera QBER (0.0 = completamente nascosto)

    def validate(self):
        assert 0 <= self.amplitude_damping_gamma <= 1
        assert 0 <= self.phase_damping_lambda <= 1
        assert 0 <= self.interception_rate <= 1
        assert 0 <= self.attack_rate <= 1
        assert 0 <= self.trojan_horse_prob <= 1
        assert 0 <= self.optical_isolator_efficiency <= 1
        assert 0 <= self.depolarization_prob <= 1
        assert 0 < self.sharing_rate <= 1
        if self.use_weak_laser:
            assert 0 < self.mean_photon_num < 1
            assert self.source_frequency_hz > 0
        if self.use_channel_attenuation:
            assert self.attenuation_coeff >= 0
            assert self.distance_km >= 0
        assert 0 <= self.detector_efficiency <= 1


# ============================================================================
# Physical modules: Source, Channel, Detector
# ============================================================================

class WeakCoherentSource:
    """Simulates a weak coherent pulse source with Poisson statistics."""
    def __init__(self, mean_photon_num: float, frequency_hz: float):
        self.mu = mean_photon_num
        self.frequency = frequency_hz
        
    def generate_photons(self, required_photons: int) -> Tuple[int, float]:
        """
        Simulate generation of a pulse train until required_photons are produced.
        Returns (total_photons_generated, time_seconds).
        For each pulse, number of photons ~ Poisson(mu). Count only pulses with >=1 photon.
        """
        pulses_needed = 0
        photons_produced = 0
        while photons_produced < required_photons:
            n_photons = np.random.poisson(self.mu)
            if n_photons > 0:
                photons_produced += 1  # each pulse contributes at most one qubit (as per BB84)
            pulses_needed += 1
        time_sec = pulses_needed / self.frequency
        return photons_produced, time_sec


class Detector:
    """Models a realistic photon detector with efficiency and dead time."""
    def __init__(self, efficiency: float, dead_time_us: float = 0.0):
        self.efficiency = efficiency
        self.dead_time_sec = dead_time_us * 1e-6
        
    def detect(self, photon_arrival_times: List[float]) -> Tuple[List[bool], float]:
        """
        Simulate detection events.
        photon_arrival_times: list of absolute times when photons hit detector.
        Returns (detection_success_list, total_dead_time_penalty_sec).
        """
        detections = []
        dead_time_accum = 0.0
        last_detection_time = -np.inf
        for t in photon_arrival_times:
            if t - last_detection_time < self.dead_time_sec:
                # detector dead, no detection
                detections.append(False)
                continue
            if np.random.random() < self.efficiency:
                detections.append(True)
                last_detection_time = t
                dead_time_accum += self.dead_time_sec
            else:
                detections.append(False)
        return detections, dead_time_accum


class QuantumChannel_ImprovedPhysics:
    """
    Versione migliorata di QuantumChannel con modello fisico realistico.
    """
    
    def __init__(self,
                 amplitude_damping_gamma: float = 0.0,
                 phase_damping_lambda: float = 0.0,
                 attenuation_coeff: float = 0.0,
                 distance_km: float = 0.0,
                 depolarization_prob: float = 0.0):
        """
        Parametri fisicamente significativi:
        
        amplitude_damping_gamma: perdita di energia (0-1)
            - Per fibra ottica 10 km @ 0.2 dB/km: γ ≈ 0.15
        
        phase_damping_lambda: dephasing/decoherence (0-1)
            - Per fiber: λ ≈ 0.05
        
        depolarization_prob: depolarizzazione generica (0-1)
            - Per canale "sporco": p ≈ 0.05-0.10
        """
        self.gamma = amplitude_damping_gamma
        self.lambda_param = phase_damping_lambda
        self.p_depol = depolarization_prob
        if distance_km > 0:
            self.transmission_prob = 10 ** (-attenuation_coeff * distance_km / 10)
        else:
            self.transmission_prob = 1.0
    
    def transmit_qubit(self, state: BlochVector) -> Optional[BlochVector]:
        """
        Trasmette un singolo qubit attraverso il canale rumoroso.
        
        Parameters:
        -----------
        state : BlochVector
            Lo stato quantico preparato da Alice
        
        Returns:
        --------
        received_state : BlochVector
            Lo stato degradato dal canale
        """
        # Applica effetti del canale in sequenza
        # (ordine può importare fisicamente, ma qui trascuriamo)
        
        if np.random.random() > self.transmission_prob:
            return None  # fotone non arriva

        if self.gamma > 0:
            state.apply_amplitude_damping(self.gamma)
        
        if self.lambda_param > 0:
            state.apply_phase_damping(self.lambda_param)
        
        if self.p_depol > 0:
            state.apply_depolarization(self.p_depol)
        
        return state
 

class Alice_WithBloch:
    """
    Versione migliorata di Alice usando Bloch sphere.
    """
    
    def __init__(self, config):
        self.config = config
    
    def prepare_qubits_with_bloch(self, num_photons: int):
        """
        Prepara qubit usando Bloch sphere (fisicamente accurato).
        
        Returns:
        --------
        states : list of BlochVector
            Stati quantici preparati da Alice
        bases : np.ndarray
            Basi usate per preparare i qubit
        bits : np.ndarray
            Bit codificati
        """
        bits = np.random.randint(0, 2, size=num_photons, dtype=np.uint8)
        bases = np.random.randint(0, 2, size=num_photons, dtype=np.uint8)
        
        # NUOVO: Crea BlochVector per ogni qubit
        states = [
            BlochVector.from_basis_and_bit(bases[i], bits[i])
            for i in range(num_photons)
        ]
        
        return states, bases, bits
    

class Bob_WithBloch:
    """
    Bob con rivelatore realistico: efficiency + dead time + Born rule.
    """
    
    def __init__(self, config):
        self.config = config
        self.detector = Detector(
            efficiency=config.detector_efficiency,
            dead_time_us=config.dead_time_us if config.use_dead_time else 0.0
        )
    
    def measure_qubit(self, received_state: BlochVector, bob_basis: int, arrival_time: float = 0.0):
        """
        Misura un singolo qubit con rivelatore realistico.
        
        Prima verifica se il detector rileva il fotone (efficiency + dead time),
        poi applica Born rule sulla Bloch sphere.
        
        Returns:
        --------
        measured_bit : int (0 o 1, oppure random se non rilevato)
        detected : bool
        probability : float
        """
        detections, _ = self.detector.detect([arrival_time])
        
        if not detections[0]:
            # Fotone non rilevato: Bob assegna bit random (dark count / loss)
            return random_bit(), False, 0.5
        
        measured_bit, probability = received_state.measure(bob_basis)
        return measured_bit, True, probability
    

class BB84SimulationV2:
    """
    Versione 2 di BB84 con modello fisico Bloch sphere.

    Differenze da versione 1:
    - Usa BlochVector per stati quantici
    - Modelli di canale fisicamente accurati
    - Misurazioni con Born rule esatto
    - Tracking della purità degli stati
    """

    def __init__(self, config):
        self.config = config
        self.alice = Alice_WithBloch(config)
        self.bob = Bob_WithBloch(config)

        # Sorgente: weak coherent pulse oppure ideale
        if config.use_weak_laser:
            self.source = WeakCoherentSource(config.mean_photon_num, config.source_frequency_hz)
        else:
            self.source = None


        self._current_eve_params = {
            "interception_rate": config.interception_rate,
            "pns_enabled": config.eve_pns_enabled,
            "pns_block_ratio": config.eve_pns_block_ratio,
            "blinding_attack_active": config.blinding_attack_active,
            "trojan_horse_prob": config.trojan_horse_prob,
            "qber_tamper_active": config.qber_tamper_active,
            "qber_tamper_amount": config.qber_tamper_amount,
        }
        self._attack_active_this_iteration = self._any_eve_attack_enabled()

        # Canale con parametri fisici realistici
        self.channel = QuantumChannel_ImprovedPhysics(
            amplitude_damping_gamma=config.amplitude_damping_gamma if config.use_amplitude_damping else 0.0,
            phase_damping_lambda=config.phase_damping_lambda if config.use_phase_damping else 0.0,
            depolarization_prob=config.depolarization_prob,
            attenuation_coeff=config.attenuation_coeff if config.use_channel_attenuation else 0.0,
            distance_km=config.distance_km if config.use_channel_attenuation else 0.0,
        )
        # Storage risultati
        self.results = {
            'state_purities': [],
            'qber_estimates': [],
            'true_qber_estimates': [],
            'sifted_key_lengths': [],
            'elapsed_times': [],
        }

        # Tracking Eve attraverso le iterazioni
        self._eve_tracking = {
            'attack_iterations': [],
            'total_trojan_leaks': 0,
            'total_blinded_bits': 0,
            'strategies_used': set(),
            'interception_rates': [],
        }


        # ========== NUOVI METODI PER CONTROLLARE EVE ==========
    def set_eve_params(self, interception_rate: float = None, 
                       pns_enabled: bool = None,
                       pns_block_ratio: float = None,
                       blinding_attack_active: bool = None,
                       trojan_horse_prob: float = None,
                       qber_tamper_active: bool = None,
                       qber_tamper_amount: float = None):
        """
        Modifica i parametri di Eve dinamicamente.
        Questo è il metodo principale che userai dal main.
        """
        if interception_rate is not None:
            self._current_eve_params["interception_rate"] = max(0.0, min(1.0, interception_rate))
        if pns_enabled is not None:
            self._current_eve_params["pns_enabled"] = pns_enabled
        if pns_block_ratio is not None:
            self._current_eve_params["pns_block_ratio"] = max(0.0, min(1.0, pns_block_ratio))
        if blinding_attack_active is not None:
            self._current_eve_params["blinding_attack_active"] = blinding_attack_active
        if trojan_horse_prob is not None:
            self._current_eve_params["trojan_horse_prob"] = max(0.0, min(1.0, trojan_horse_prob))
        if qber_tamper_active is not None:
            self._current_eve_params["qber_tamper_active"] = qber_tamper_active
        if qber_tamper_amount is not None:
            self._current_eve_params["qber_tamper_amount"] = max(0.0, min(1.0, qber_tamper_amount))
        
        self._attack_active_this_iteration = self._any_eve_attack_enabled()
    
    def _any_eve_attack_enabled(self) -> bool:
        """True se almeno un vettore di attacco Eve è attivo."""
        p = self._current_eve_params
        return (
            self.config.eve_attack_enabled or
            p["interception_rate"] > 0 or
            p["pns_enabled"] or
            p["blinding_attack_active"] or
            p["trojan_horse_prob"] > 0 or
            p["qber_tamper_active"]
        )
    
    def disable_eve(self):
        """Disabilita completamente Eve (attacco disattivo)"""
        self.set_eve_params(interception_rate=0.0, pns_enabled=False)
    
    def enable_eve(self, interception_rate: float = 0.3):
        """Abilita Eve con interception rate specificato"""
        self.set_eve_params(interception_rate=interception_rate, pns_enabled=False)
    
    def get_eve_params(self) -> dict:
        """Restituisce i parametri correnti di Eve"""
        return self._current_eve_params.copy()
    
    def is_attack_active(self) -> bool:
        """Restituisce True se Eve è attiva nell'iterazione corrente"""
        return self._attack_active_this_iteration

    

    # ------------------------------------------------------------------
    def run(self) -> dict:
        for i in range(self.config.num_iterations):
            self.run_single_iteration_v2(i)
        return {
            'avg_qber': np.nanmean(self.results['qber_estimates']),
            'avg_sifted_len': np.mean(self.results['sifted_key_lengths']),
            'avg_purity': np.nanmean(self.results['state_purities']),
        }

    def run_single_iteration_v2(self, iteration_idx: int) -> dict:

        # Decide se questa iterazione è attaccata
        if self.config.use_random_attacks and self.config.eve_attack_enabled:
            self._attack_active_this_iteration = np.random.random() < self.config.attack_rate
        else:
            self._attack_active_this_iteration = self._any_eve_attack_enabled()

        self.source = WeakCoherentSource(
            self.config.mean_photon_num, self.config.source_frequency_hz
        ) if self.config.use_weak_laser else None

        # 0. Numero di fotoni da trasmettere
        if self.source:
            N, elapsed_time = self.source.generate_photons(self.config.raw_key_size)
        else:
            N = self.config.raw_key_size
            elapsed_time = 0.0

        # 1. Alice prepara i qubit
        alice_states, alice_bases, alice_bits = self.alice.prepare_qubits_with_bloch(N)

        # 2. Basi e array di misura di Bob (definiti PRIMA del loop)
        bob_bases      = np.random.randint(0, 2, size=N, dtype=np.uint8)
        bob_bits       = np.zeros(N, dtype=np.uint8)
        bob_detected   = np.zeros(N, dtype=bool)
        bob_confidence = np.zeros(N, dtype=float)
        arrival_times  = np.arange(N) * 1e-9
        purities       = []
        blinded_mask   = np.zeros(N, dtype=bool)
        trojan_leaked  = np.zeros(N, dtype=bool)

        eve_rate = self._current_eve_params["interception_rate"] if self._attack_active_this_iteration else 0.0
        eve_pns  = self._current_eve_params["pns_enabled"]       if self._attack_active_this_iteration else False
        blinding = self._current_eve_params["blinding_attack_active"] if self._attack_active_this_iteration else False
        trojan_p = self._current_eve_params["trojan_horse_prob"] if self._attack_active_this_iteration else 0.0
        isolator = self.config.optical_isolator_efficiency
        trojan_effective_p = trojan_p * (1.0 - isolator) if trojan_p > 0 else 0.0

        # 3. Loop unico: trasmissione + misurazione qubit per qubit
        for i in range(N):

            current_state = alice_states[i]
            eve_bit = None

            # Eve: Trojan Horse — leak della base Alice (bypass isolatore ottico)
            if trojan_effective_p > 0 and np.random.random() < trojan_effective_p:
                trojan_leaked[i] = True

            # Eve: Intercept-Resend
            if self._attack_active_this_iteration and eve_rate > 0:
                if np.random.random() < eve_rate:
                    eve_basis = int(alice_bases[i]) if trojan_leaked[i] else np.random.randint(0, 2)
                    eve_bit, _, _ = self.bob.measure_qubit(
                        alice_states[i], eve_basis, arrival_time=arrival_times[i]
                    )
                    if eve_basis == 0:
                        resent_state = BlochVector.z_basis_0() if eve_bit == 0 else BlochVector.z_basis_1()
                    else:
                        resent_state = BlochVector.x_basis_plus() if eve_bit == 0 else BlochVector.x_basis_minus()
                    current_state = resent_state
                else:
                    current_state = alice_states[i]
            else:
                current_state = alice_states[i]

            # Eve: PNS
            if self._attack_active_this_iteration and eve_pns and self.source:
                n_photons_in_pulse = np.random.poisson(self.config.mean_photon_num)
                if n_photons_in_pulse >= 2:
                    pass
                elif n_photons_in_pulse == 1 and self._current_eve_params.get("pns_block_ratio", 0) > 0:
                    if np.random.random() < self._current_eve_params["pns_block_ratio"]:
                        current_state = None

            state_to_transmit = current_state

            if state_to_transmit is None:
                bob_detected[i]   = False
                bob_bits[i]       = random_bit()
                bob_confidence[i] = 0.5
                if self.config.track_state_purities:
                    purities.append(0.5)
                continue

            received_state = self.channel.transmit_qubit(state_to_transmit)

            # Fotone perso per attenuazione
            if received_state is None:
                bob_detected[i]   = False
                bob_bits[i]       = random_bit()
                bob_confidence[i] = 0.5
                if self.config.track_state_purities:
                    purities.append(0.5)
                continue

            # Track purità
            if self.config.track_state_purities:
                purities.append(received_state.purity())

            # Misura di Bob
            bit, detected, confidence = self.bob.measure_qubit(
                received_state, bob_bases[i], arrival_time=arrival_times[i]
            )
            bob_bits[i]       = bit
            bob_detected[i]   = detected
            bob_confidence[i] = confidence

            # Eve: Blinding — forza l'output del detector di Bob
            if blinding:
                if eve_bit is not None:
                    bob_bits[i] = eve_bit
                    blinded_mask[i] = True
                elif trojan_leaked[i]:
                    bob_bits[i] = alice_bits[i]
                    blinded_mask[i] = True

        # 4. Sifting
        match        = (alice_bases == bob_bases) & bob_detected
        match_indices = np.where(match)[0]
        alice_sifted = alice_bits[match]
        bob_sifted   = bob_bits[match]
        sifted_len   = len(alice_sifted)

        # 5. Stima QBER
        if sifted_len > 0:
            share_size  = max(1, int(sifted_len * self.config.sharing_rate))
            alice_share = alice_sifted[:share_size]
            bob_share   = bob_sifted[:share_size].copy()

            true_qber_est = np.sum(alice_share != bob_share) / share_size

            # Blinding: azzera gli errori sui bit manipolati da Eve
            if blinding:
                for j in range(share_size):
                    if blinded_mask[match_indices[j]]:
                        bob_share[j] = alice_share[j]

            errors   = np.sum(alice_share != bob_share)
            qber_est = errors / share_size

            # Eve: QBER tampering — altera la stima prima che Alice la legga
            if self._attack_active_this_iteration and self._current_eve_params["qber_tamper_active"]:
                tamper_amount = self._current_eve_params["qber_tamper_amount"]
                if tamper_amount == 0.0:
                    bob_share = alice_share.copy()
                    qber_est = 0.0
                else:
                    errors_target = int(round(tamper_amount * share_size))
                    num_flip_to_zero = min(int(np.sum(alice_share[:share_size] == 1)), share_size - errors_target)
                    error_indices = np.where(alice_share[:share_size] == 1)[0]
                    if len(error_indices) > num_flip_to_zero:
                        flip_indices = error_indices[np.random.choice(len(error_indices), num_flip_to_zero, replace=False)]
                        bob_share[flip_indices] = 0
                    remaining_errors = errors_target - num_flip_to_zero
                    zero_indices = np.where(alice_share[:share_size] == 0)[0]
                    if remaining_errors > 0 and len(zero_indices) >= remaining_errors:
                        flip_indices = zero_indices[np.random.choice(len(zero_indices), remaining_errors, replace=False)]
                        bob_share[flip_indices] = 1
                    qber_est = np.sum(alice_share[:share_size] != bob_share) / share_size
        else:
            share_size = 0
            qber_est   = float('nan')
            true_qber_est = float('nan')

        # 6. Purità media
        avg_purity = float(np.mean(purities)) if purities else np.nan

        # 7. Detection flag
        detection_threshold = 0.11
        detected = qber_est > detection_threshold if not np.isnan(qber_est) else False

        # 8. Aggiornamento storage
        self.results['state_purities'].append(avg_purity)
        self.results['qber_estimates'].append(qber_est)
        self.results['true_qber_estimates'].append(true_qber_est if isinstance(true_qber_est, (int, float)) else 0.0)
        self.results['sifted_key_lengths'].append(sifted_len)
        self.results['elapsed_times'].append(elapsed_time)

        # 9. Chiave usabile
        if self.config.research_mode:
            usable_key = alice_sifted[share_size:].tolist() if sifted_len > 0 else []
        else:
            usable_key = None

        # 10. Tracking Eve per get_full_parameters
        if self._attack_active_this_iteration:
            self._eve_tracking['attack_iterations'].append(iteration_idx)
            self._eve_tracking['total_trojan_leaks'] += int(np.sum(trojan_leaked))
            self._eve_tracking['total_blinded_bits'] += int(np.sum(blinded_mask))

            # Determina strategia
            if self.config.blinding_attack_active:
                self._eve_tracking['strategies_used'].add('blinding')
            elif self.config.eve_pns_enabled:
                self._eve_tracking['strategies_used'].add('pns')
            elif self.config.trojan_horse_prob > 0:
                self._eve_tracking['strategies_used'].add('trojan_horse')
            elif eve_rate > 0:
                self._eve_tracking['strategies_used'].add('intercept-resend')

            self._eve_tracking['interception_rates'].append(eve_rate)

        return {
            'sifted_len':   sifted_len,
            'qber_est':     qber_est,
            'true_qber_est': true_qber_est,
            'avg_purity':   avg_purity,
            'usable_key':   usable_key,
            'detected':     detected,
            'attack_active': self._attack_active_this_iteration,
            'eve_params':   self._current_eve_params.copy() if self._attack_active_this_iteration else None,
            'trojan_leaks': int(np.sum(trojan_leaked)),
            'blinded_bits': int(np.sum(blinded_mask)),
            'elapsed_time': elapsed_time,
        }

    def analyze_security(self, sifted_key_length: int = None) -> Dict:
        """Restituisce analisi completa di sicurezza."""
        qber = np.nanmean(self.results['qber_estimates'])
        if sifted_key_length is None:
            sifted_key_length = int(self.config.raw_key_size * 0.5)

        if not np.isfinite(qber):
            return {
                'secure_key_length': 0,
                'eve_max_information': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'detection_power': 0.0,
                'shor_preskill_bound': 0.0,
            }

        analyzer = SecurityAnalyzer(
            n_raw=self.config.raw_key_size,
            qber=qber,
            sharing_rate=self.config.sharing_rate
        )

        secure_len, eve_info = analyzer.calculate_secure_key_length(n_sifted=sifted_key_length)
        hypothesis_test = analyzer.test_hypothesis()

        return {
            'secure_key_length': secure_len,
            'eve_max_information': float(eve_info),
            'false_positive_rate': hypothesis_test['false_positive_rate'],
            'false_negative_rate': hypothesis_test['false_negative_rate'],
            'detection_power': hypothesis_test['detection_power'],
            'shor_preskill_bound': analyzer.shor_preskill_bound(qber)
        }

    def get_full_parameters(self, iteration_idx: int = None) -> Dict:
        """
        Restituisce tutti i parametri osservabili dal canale BB84,
        nel formato richiesto dal Recon Agent.

        Se iteration_idx e' None, restituisce i parametri medi su
        tutte le iterazioni eseguite.

        Parametri restituiti:
            n_qubits_sent: numero di qubit/pulsi inviati da Alice
            raw_key_length: lunghezza della chiave grezza (prima del sifting)
            sifted_key_length: lunghezza della chiave siftata (basi concordate)
            basis_match_rate: tasso di corrispondenza delle basi
            qber: Quantum Bit Error Rate stimato
            true_qber: QBER reale (senza tampering di Eve)
            channel_loss: perdita stimata del canale
            detector_efficiency: efficienza del rivelatore di Bob
            dark_count_rate: tasso di dark counts (thermal noise)
            eve_present: True se Eve ha effettuato almeno un attacco
            eve_strategy: strategia di attacco usata (intercept-resend, pns, blinding, trojan_horse, qber_tamper, none)
            eve_interception_rate: interception rate di Eve
            eve_detection_probability: probabilita' di rilevamento di Eve
            trojan_leaks: numero di leak della base Alice via Trojan Horse
            blinded_bits: numero di bit manipolati da blinding attack
            error_correction_leakage: stima bit leakati durante error correction
            privacy_amplification_ratio: ratio di privacy amplification
            final_key_rate: tasso di chiave finale
            avg_state_purity: purita' media degli stati ricevuti
            protocol_aborted: True se QBER > soglia di abort
            secure_key_length: lunghezza sicura della chiave (Shor-Preskill)
            eve_max_information: informazione massima di Eve
            false_positive_rate: tasso di falso positivo
            false_negative_rate: tasso di falso negativo
            detection_power: potenza di rilevamento
            shor_preskill_bound: bound di Shor-Preskill
        """
        # Usa l'ultima iterazione se non specificato
        if iteration_idx is not None:
            # Esegue l'iterazione se non e' stata ancora fatta
            if iteration_idx >= len(self.results['sifted_key_lengths']):
                self.run_single_iteration_v2(iteration_idx)

        # Aggrega su tutte le iterazioni
        n_iter = len(self.results['sifted_key_lengths'])
        if n_iter == 0:
            # Nessuna iterazione eseguita: usa config
            return self._build_parameters_from_config()

        avg_qber = float(np.nanmean(self.results['qber_estimates'])) if self.results['qber_estimates'] else 0.0
        if not np.isfinite(avg_qber):
            avg_qber = 0.0
        avg_true_qber = float(np.nanmean(self.results['true_qber_estimates'])) if self.results['true_qber_estimates'] else 0.0
        if not np.isfinite(avg_true_qber):
            avg_true_qber = 0.0
        avg_sifted = float(np.mean(self.results['sifted_key_lengths']))
        avg_purity = float(np.nanmean(self.results['state_purities'])) if self.results['state_purities'] else None

        # Usa l'ultima iterazione per i dettagli
        last_idx = n_iter - 1
        last_qber = self.results['qber_estimates'][last_idx]
        last_sifted = self.results['sifted_key_lengths'][last_idx]

        # Eve detection: usa il tracking accumulato
        eve_present = len(self._eve_tracking['attack_iterations']) > 0
        eve_strategy = "none"
        if self._eve_tracking['strategies_used']:
            priority = ['blinding', 'trojan_horse', 'pns', 'intercept-resend']
            for strat in priority:
                if strat in self._eve_tracking['strategies_used']:
                    eve_strategy = strat
                    break
            else:
                eve_strategy = list(self._eve_tracking['strategies_used'])[0]

        trojan_leaks = self._eve_tracking['total_trojan_leaks']
        blinded_bits = self._eve_tracking['total_blinded_bits']
        eve_interception_rate = float(np.mean(self._eve_tracking['interception_rates'])) if self._eve_tracking['interception_rates'] else 0.0

        # Stima channel loss dal rapporto sifted/raw
        basis_match = last_sifted / self.config.raw_key_size if self.config.raw_key_size > 0 else 0.0
        channel_loss = max(0.0, 1.0 - basis_match * 2)  # 50% sifting baseline

        # Error correction leakage
        error_correction_leakage = int(last_sifted * avg_qber) if avg_qber > 0 and isinstance(avg_qber, (int, float)) else 0

        # Privacy amplification
        pa_ratio = max(0.0, 1.0 - avg_qber * 2) if avg_qber > 0 and isinstance(avg_qber, (int, float)) else 1.0

        # Final key rate
        final_key_rate = (last_sifted * pa_ratio) / self.config.raw_key_size if self.config.raw_key_size > 0 else 0.0

        # Protocol abort
        protocol_aborted = bool(avg_qber > 0.11) if isinstance(avg_qber, (int, float)) and avg_qber == avg_qber else False

        # Security analysis
        security = self.analyze_security(sifted_key_length=last_sifted)

        # Garantisce secure_key_length ≤ sifted_key_length (regola fisica fondamentale)
        if security["secure_key_length"] > last_sifted:
            security["secure_key_length"] = last_sifted

        # Parametri fisici del canale (per passaggio agli agent)
        mean_photon_num = self.config.mean_photon_num
        distance_km = self.config.distance_km if self.config.use_channel_attenuation else 0.0
        amplitude_damping_gamma = self.config.amplitude_damping_gamma if self.config.use_amplitude_damping else 0.0

        # Eve detection probability (approssimata dal QBER quando Eve e' presente)
        eve_detection_prob = avg_qber if eve_present and isinstance(avg_qber, (int, float)) else 0.0

        return {
            # Parametri base
            "n_qubits_sent": self.config.raw_key_size,
            "raw_key_length": self.config.raw_key_size,
            "sifted_key_length": int(last_sifted),
            "basis_match_rate": round(float(basis_match), 6),
            "channel_loss": round(float(channel_loss), 6),

            # QBER
            "qber": round(float(avg_qber), 6) if isinstance(avg_qber, (int, float)) and avg_qber == avg_qber else 0.0,
            "true_qber": round(float(avg_true_qber), 6) if isinstance(avg_true_qber, (int, float)) and avg_true_qber == avg_true_qber else 0.0,

            # Hardware / canale
            "detector_efficiency": self.config.detector_efficiency,
            "dark_count_rate": self.config.thermal_ratio if self.config.use_thermal_noise else 0.0,

            # Eve
            "eve_present": eve_present,
            "eve_strategy": eve_strategy,
            "eve_interception_rate": eve_interception_rate,
            "eve_detection_probability": round(float(eve_detection_prob), 6),
            "trojan_leaks": trojan_leaks,
            "blinded_bits": blinded_bits,

            # Post-processing
            "error_correction_leakage": error_correction_leakage,
            "privacy_amplification_ratio": round(float(pa_ratio), 6),
            "final_key_rate": round(float(final_key_rate), 6),

            # Qualita' stato
            "avg_state_purity": round(float(avg_purity), 6) if avg_purity is not None else None,

            # Decisione protocollo
            "protocol_aborted": protocol_aborted,

            # Security analysis
            "secure_key_length": security["secure_key_length"],
            "eve_max_information": security["eve_max_information"],
            "false_positive_rate": security["false_positive_rate"],
            "false_negative_rate": security["false_negative_rate"],
            "detection_power": security["detection_power"],
            "shor_preskill_bound": security["shor_preskill_bound"],

            # Parametri fisici del canale (per passaggio agli agent)
            "mean_photon_num": mean_photon_num,
            "distance_km": distance_km,
            "amplitude_damping_gamma": amplitude_damping_gamma,
        }

    def _build_parameters_from_config(self) -> Dict:
        """Restituisce parametri baseline dalla config senza eseguire simulazione."""
        return {
            "n_qubits_sent": self.config.raw_key_size,
            "raw_key_length": self.config.raw_key_size,
            "sifted_key_length": 0,
            "basis_match_rate": 0.0,
            "channel_loss": 0.0,
            "qber": 0.0,
            "true_qber": 0.0,
            "detector_efficiency": self.config.detector_efficiency,
            "dark_count_rate": self.config.thermal_ratio if self.config.use_thermal_noise else 0.0,
            "eve_present": self.config.eve_attack_enabled and self.config.interception_rate > 0,
            "eve_strategy": "intercept-resend" if self.config.interception_rate > 0 else "none",
            "eve_interception_rate": self.config.interception_rate,
            "eve_detection_probability": 0.0,
            "trojan_leaks": 0,
            "blinded_bits": 0,
            "error_correction_leakage": 0,
            "privacy_amplification_ratio": 1.0,
            "final_key_rate": 0.0,
            "avg_state_purity": None,
            "protocol_aborted": False,
            "secure_key_length": 0,
            "eve_max_information": 0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "detection_power": 0.0,
            "shor_preskill_bound": 0.0,
        }


if __name__ == "__main__":

    print("="*70)
    print("EXAMPLE: Integrazione Bloch Sphere nel BB84")
    print("="*70)
    
    config = SimulationConfig()

    # Esempio 1: Singolo qubit con Bloch sphere
    print("\n1. Preparazione singolo qubit:")
    print("-" * 70)
    
    state = BlochVector.from_basis_and_bit(0, 1)  # |1⟩ in Z basis
    print(f"Alice prepara |1>:")
    print(f"  Bloch vector: ({state.x:.3f}, {state.y:.3f}, {state.z:.3f})")
    print(f"  Purity: {state.purity():.4f}")
    
    # Applica canale
    state.apply_amplitude_damping(0.1)
    state.apply_phase_damping(0.05)
    print(f"\nDopo canale (gamma=0.1, lambda=0.05):")
    print(f"  Bloch vector: ({state.x:.3f}, {state.y:.3f}, {state.z:.3f})")
    print(f"  Purity: {state.purity():.4f}")
    
    # Misura
    bit, prob = state.measure(0)  # Bob misura in Z basis
    print(f"\nBob misura in Z basis:")
    print(f"  Misurato: {bit}, Probabilità: {prob:.4f}")
    
    # Esempio 2: Simulazione multi-iterazione con Eve e stampa di TUTTI i parametri
    print("\n\n2. Simulazione BB84 con Eve (5 iterazioni, attacco intercept-resend):")
    print("-" * 70)
    
    config_eve = SimulationConfig(
        raw_key_size=5000,
        num_iterations=5,
        interception_rate=0.25,
        eve_attack_enabled=True,
        use_amplitude_damping=True,
        amplitude_damping_gamma=0.15,
        detector_efficiency=0.72,
        depolarization_prob=0.01,
    )
    sim_eve = BB84SimulationV2(config_eve)
    sim_eve.run()
    
    all_params = sim_eve.get_full_parameters()
    
    print("\n=== PARAMETRI COMPLETI DEL CANALE BB84 ===")
    print()
    print("--- Parametri Base ---")
    print(f"  n_qubits_sent:           {all_params['n_qubits_sent']}")
    print(f"  raw_key_length:          {all_params['raw_key_length']}")
    print(f"  sifted_key_length:       {all_params['sifted_key_length']}")
    print(f"  basis_match_rate:        {all_params['basis_match_rate']:.6f}")
    print(f"  channel_loss:            {all_params['channel_loss']:.6f}")
    
    print()
    print("--- QBER ---")
    print(f"  qber:                    {all_params['qber']:.6f}")
    print(f"  true_qber:               {all_params['true_qber']:.6f}")
    
    print()
    print("--- Hardware / Canale ---")
    print(f"  detector_efficiency:     {all_params['detector_efficiency']}")
    print(f"  dark_count_rate:         {all_params['dark_count_rate']}")
    
    print()
    print("--- Eve ---")
    print(f"  eve_present:             {all_params['eve_present']}")
    print(f"  eve_strategy:            {all_params['eve_strategy']}")
    print(f"  eve_interception_rate:   {all_params['eve_interception_rate']:.6f}")
    print(f"  eve_detection_probability:{all_params['eve_detection_probability']:.6f}")
    print(f"  trojan_leaks:            {all_params['trojan_leaks']}")
    print(f"  blinded_bits:            {all_params['blinded_bits']}")
    
    print()
    print("--- Post-Processing ---")
    print(f"  error_correction_leakage:{all_params['error_correction_leakage']}")
    print(f"  privacy_amplification_ratio:{all_params['privacy_amplification_ratio']:.6f}")
    print(f"  final_key_rate:          {all_params['final_key_rate']:.6f}")
    
    print()
    print("--- Qualita' Stato ---")
    print(f"  avg_state_purity:        {all_params['avg_state_purity']}")
    
    print()
    print("--- Decisione Protocollo ---")
    print(f"  protocol_aborted:        {all_params['protocol_aborted']}")
    
    print()
    print("--- Security Analysis ---")
    print(f"  secure_key_length:       {all_params['secure_key_length']}")
    print(f"  eve_max_information:     {all_params['eve_max_information']}")
    print(f"  false_positive_rate:     {all_params['false_positive_rate']:.10f}")
    print(f"  false_negative_rate:     {all_params['false_negative_rate']:.6f}")
    print(f"  detection_power:         {all_params['detection_power']:.6f}")
    print(f"  shor_preskill_bound:     {all_params['shor_preskill_bound']:.6f}")
    
    print()
    print(f"Totale parametri: {len(all_params)}")
    print()
    
    # Esempio 3: Senza Eve
    print("\n3. Simulazione BB84 SENZA Eve (canale pulito):")
    print("-" * 70)
    
    config_clean = SimulationConfig(
        raw_key_size=5000,
        num_iterations=5,
        interception_rate=0.0,
        eve_attack_enabled=False,
        use_amplitude_damping=True,
        amplitude_damping_gamma=0.10,
        detector_efficiency=0.72,
    )
    sim_clean = BB84SimulationV2(config_clean)
    sim_clean.run()
    clean_params = sim_clean.get_full_parameters()
    
    print()
    print("--- Parametri Base ---")
    print(f"  n_qubits_sent:           {clean_params['n_qubits_sent']}")
    print(f"  sifted_key_length:       {clean_params['sifted_key_length']}")
    print(f"  channel_loss:            {clean_params['channel_loss']:.6f}")
    
    print()
    print("--- QBER ---")
    print(f"  qber:                    {clean_params['qber']:.6f}")
    
    print()
    print("--- Eve ---")
    print(f"  eve_present:             {clean_params['eve_present']}")
    print(f"  eve_strategy:            {clean_params['eve_strategy']}")
    
    print()
    print("--- Decisione Protocollo ---")
    print(f"  protocol_aborted:        {clean_params['protocol_aborted']}")
    
    print()
    print("--- Security Analysis ---")
    print(f"  secure_key_length:       {clean_params['secure_key_length']}")
    print(f"  detection_power:         {clean_params['detection_power']:.6f}")
    
    print()
    print("=" * 70)
    print("SIMULAZIONE COMPLETA")
    print("=" * 70)


   # viz = VisualizationSuite()
   # viz.plot_qber_vs_noise()
   # viz.plot_purity_vs_gamma()
   # viz.plot_key_distribution(n_iterations=100)
   # viz.plot_qber_vs_distance()
