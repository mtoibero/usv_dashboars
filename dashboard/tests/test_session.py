"""Run history and staleness detection."""

import unittest
from types import SimpleNamespace

from dashboard import config as cfg
from dashboard.session import RunHistory, staleness


def fake_result(config):
    return SimpleNamespace(config=config)


class RunHistoryTests(unittest.TestCase):
    def test_numbers_labels_and_bound(self):
        history = RunHistory(max_runs=3)
        for i in range(5):
            history.add(fake_result(cfg.default_config()), label=f" r{i} ")
        self.assertEqual([r.number for r in history.records], [3, 4, 5])
        self.assertEqual(history.latest.number, 5)
        self.assertEqual(history.previous.number, 4)
        self.assertEqual(history.latest.title, "Run #5 - r4")
        with self.assertRaises(KeyError):
            history.get(1)

    def test_numbers_are_not_reused_after_clear(self):
        history = RunHistory()
        history.add(fake_result(cfg.default_config()))
        history.clear()
        self.assertIsNone(history.latest)
        self.assertEqual(history.add(fake_result(cfg.default_config())).number, 2)

    def test_each_record_keeps_its_own_config(self):
        history = RunHistory()
        a = cfg.default_config()
        b = cfg.form_to_config(dict(cfg.default_form(), smc_k_psi=5.0))
        history.add(fake_result(a))
        history.add(fake_result(b))
        self.assertEqual(history.previous.config.controller.k_psi, 3.0)
        self.assertEqual(history.latest.config.controller.k_psi, 5.0)


class StalenessTests(unittest.TestCase):
    def setUp(self):
        self.history = RunHistory()
        self.record = self.history.add(fake_result(cfg.default_config()))

    def test_same_config_is_not_stale(self):
        self.assertEqual(staleness(self.record, cfg.default_config()),
                         (False, []))

    def test_changed_config_is_stale_with_differences(self):
        current = cfg.form_to_config(dict(cfg.default_form(), smc_phi_u=0.2))
        stale, differences = staleness(self.record, current)
        self.assertTrue(stale)
        self.assertEqual(differences, [("controller.phi_u", 0.08, 0.2)])

    def test_invalid_sidebar_config_counts_as_stale(self):
        self.assertEqual(staleness(self.record, None), (True, []))

    def test_no_record(self):
        self.assertEqual(staleness(None, cfg.default_config()), (False, []))


if __name__ == "__main__":
    unittest.main()
