"""In-memory run history for one dashboard session (no Streamlit imports).

Each record keeps the ``SimulationResult`` together with its own frozen
``SimulationConfig``, so comparisons remain reproducible.  Nothing is
persisted: the history lives only as long as the browser session.
"""

from __future__ import annotations

from dataclasses import dataclass

from dashboard import config as cfg

MAX_STORED_RUNS = 8


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
        return f"Run #{self.number}" + (f" - {self.label}" if self.label else "")


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

    def remove(self, number):
        self._records = [r for r in self._records if r.number != number]

    def clear(self):
        self._records = []

    def __len__(self):
        return len(self._records)


def staleness(record, current_config):
    """Describe how the displayed ``record`` relates to the sidebar config.

    Returns ``(is_stale, differences)`` where ``differences`` is the list of
    ``(parameter, value used by the run, current sidebar value)``.  When the
    current sidebar config is invalid (``None``), the result is stale.
    """
    if record is None:
        return False, []
    if current_config is None:
        return True, []
    differences = cfg.config_diff(record.config, current_config)
    return bool(differences), differences
