"""Small simulator-independent cache; never evaluates an unrequested modality."""
from collections import Counter


class DemandCache:
    def __init__(self, enabled):
        self.enabled = frozenset(enabled)
        self.step = 0
        self.values = {}
        self.calls = Counter()

    def advance(self):
        self.step += 1
        self.values.clear()

    def invalidate(self):
        self.values.clear()

    def get(self, key, compute):
        if key not in self.enabled:
            raise RuntimeError(f'Tactile modality {key!r} was disabled in the sensor configuration')
        if key not in self.values:
            value = compute()
            self.values[key] = value
            self.calls[key] += 1
        return self.values[key]
