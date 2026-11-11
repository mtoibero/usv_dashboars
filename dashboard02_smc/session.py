"""In-memory run history for one dashboard session (no Streamlit imports).

Each record keeps the ``SimulationResult`` with its own frozen config, so the
comparison with a previous run always shows what that run really used.
Nothing is persisted: the history lives as long as the browser session.
"""

from __future__ import annotations

from dataclasses import dataclass

from dashboard02_smc import config as cfg

MAX_STORED_RUNS = 10


@dataclass(frozen=True)
class RunRecord:
    number: int
    label: str
    result: object  # engine.SimulationResult

    @property
    def config(self):
        return self.result.config

    @property
    def title(self):
        return f"Run #{self.number}" + (f" – {self.label}" if self.label else "")


class RunHistory:
    """Bounded, ordered list of completed runs (oldest first)."""

    def __init__(self, max_runs=MAX_STORED_RUNS):
        self.max_runs = int(max_runs)
        self._records = []
        self._next_number = 1

    def add(self, result, label=""):
        record = RunRecord(self._next_number, str(label).strip(), result)
        self._next_number += 1
        self._records.append(record)
        del self._records[:-self.max_runs]
        return record

    @property
    def records(self):
        return tuple(self._records)

    @property
    def latest(self):
        return self._records[-1] if self._records else None

    @property
    def previous(self):
        return self._records[-2] if len(self._records) >= 2 else None

    def get(self, number):
        for record in self._records:
            if record.number == number:
                return record
        raise KeyError(number)

    def clear(self):
        """Forget all runs (run numbers keep increasing)."""
        self._records = []

    def __len__(self):
        return len(self._records)


def staleness(record, current_config):
    """``(is_stale, differences)`` between a displayed run and the sidebar.

    ``differences`` lists ``(flat key, value used by the run, sidebar value)``.
    An invalid sidebar (``current_config is None``) counts as stale.
    """
    if record is None:
        return False, []
    if current_config is None:
        return True, []
    differences = cfg.config_diff(record.config, current_config)
    return bool(differences), differences
