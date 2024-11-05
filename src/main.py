import math
import numpy as np

class Qudit:
    def __init__(self, *amplitudes):
        self.amplitudes = list(amplitudes)
        self._validate()

    def _validate(self):
        total_prob = sum(abs(c) ** 2 for c in self.amplitudes)
        if round(total_prob, 10) != 1.0:
            raise ValueError(f"Qudit outcome probabilities do not sum to 1: {total_prob}")

    def measure(self):
        """Measures the qudit, collapsing to a single outcome based on probabilities."""
        obs = np.random.rand()
        cumulative_prob = 0
        for i, amplitude in enumerate(self.amplitudes):
            prob = abs(amplitude) ** 2
            cumulative_prob += prob
            if obs < cumulative_prob:
                # Collapse the state to the measured outcome
                self.amplitudes = [1.0 if j == i else 0.0 for j in range(len(self.amplitudes))]
                return i

    def reset_amplitudes(self, *amplitudes):
        self.amplitudes = list(amplitudes)
        self._validate()


class QuditAutomaton:
    def __init__(self, num_qudits, dimension):
        """Initialize the automaton with a chain of qudits."""
        self.dimension = dimension
        self.qudits = [Qudit(*(1.0 if i == 0 else 0.0 for i in range(dimension))) for _ in range(num_qudits)]
        self.entanglement_matrix = np.eye(num_qudits)

    def entangle(self, i, j, strength=1.0):
        """Entangle two qudits with a specified strength, modifying the entanglement matrix."""
        self.entanglement_matrix[i, j] = strength
        self.entanglement_matrix[j, i] = strength

    def evolve(self, rules):
        """Apply evolution rules to each qudit based on neighboring qudits."""
        new_amplitudes = []
        for idx, qudit in enumerate(self.qudits):
            new_amplitude = self.apply_rules(idx, qudit.amplitudes, rules)
            new_amplitudes.append(new_amplitude)

        # Update each qudit's amplitudes with the new values after all calculations
        for qudit, amplitude in zip(self.qudits, new_amplitudes):
            qudit.reset_amplitudes(*amplitude)

    def apply_rules(self, idx, amplitudes, rules):
        """Applies the evolution rules to a qudit's amplitudes, taking entanglement into account."""
        entangled_amplitudes = amplitudes[:]
        for neighbor_idx in range(len(self.qudits)):
            if self.entanglement_matrix[idx, neighbor_idx] != 0:
                neighbor_qudit = self.qudits[neighbor_idx]
                entangle_strength = self.entanglement_matrix[idx, neighbor_idx]
                # Combine amplitudes based on the entanglement strength and neighbor's state
                entangled_amplitudes = [
                    entangled_amplitudes[i] + entangle_strength * neighbor_qudit.amplitudes[i]
                    for i in range(self.dimension)
                ]
        
        # Apply the rules to evolve the qudit's state
        new_amplitudes = [0] * self.dimension
        for i, amplitude in enumerate(entangled_amplitudes):
            new_amplitudes[i] = rules.get(i, lambda x: x)(amplitude)

        # Normalize to ensure the sum of probabilities is 1
        norm = math.sqrt(sum(abs(a) ** 2 for a in new_amplitudes))
        if norm != 0:
            new_amplitudes = [a / norm for a in new_amplitudes]
        
        return new_amplitudes

    def measure_all(self):
        """Measure all qudits in the system, collapsing their states."""
        return [qudit.measure() for qudit in self.qudits]

    def simulate(self, steps, rules):
        """Simulate the evolution of the system over a number of steps."""
        for step in range(steps):
            print(f"Step {step}:")
            self.evolve(rules)
            measurements = self.measure_all()
            print(f"Measurements: {measurements}")

if __name__ == '__main__':
    rules = {
        0: lambda amp: amp * math.cos(math.pi / 4),  # 45-degree phase shift
        1: lambda amp: amp * math.sin(math.pi / 4),
        2: lambda amp: amp * (1 - 0.1)  # Slight damping factor
    }

    # Initialize a 5-qudit automaton with dimension 3 (each qudit has 3 possible states)
    automaton = QuditAutomaton(num_qudits=5, dimension=3)

    # Entangle qudits in pairs
    automaton.entangle(0, 1, strength=0.5)
    automaton.entangle(1, 2, strength=0.3)
    automaton.entangle(3, 4, strength=0.7)

    # Run simulation for 10 steps
    automaton.simulate(steps=10, rules=rules)