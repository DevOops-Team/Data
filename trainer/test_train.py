"""train.py의 순수 함수 테스트 (torch · GPU 불필요): cd trainer && python3 -m unittest test_train.py"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import train  # noqa: E402

CONFIG = {
    "schema_version": 2,
    "model": "gpt2",
    "training_mode": "fsdp",
    "checkpoint_format": "fsdp-dcp",
    "dataset": {"uri": "s3://b/datasets/3f9a1c2e/v1/", "format": "text-jsonl", "manifest_sha256": "aa"},
    "epochs": 10,
    "max_steps": None,
    "batch_size_per_gpu": 4,
    "learning_rate": 3e-05,
    "warmup_steps": 100,
    "seed": 42,
    "world_size": 2,
    "checkpoint_policy": {"interval_seconds": 844, "estimated_save_seconds": 16.5, "mtbf_seconds": 21600,
                          "warmup_measure_steps": 20},
}


def raw(cfg=None) -> bytes:
    return json.dumps(cfg or CONFIG).encode()


class ParseConfig(unittest.TestCase):
    def test_reads_the_backend_config_and_its_hash(self):
        cfg, digest = train.parse_config(raw(), hashlib.sha256(raw()).hexdigest())
        self.assertEqual(cfg.checkpoint_policy.interval_seconds, 844)
        self.assertEqual(cfg.dataset.format, "text-jsonl")
        self.assertEqual(digest, hashlib.sha256(raw()).hexdigest())

    def test_rejects_a_changed_config(self):
        with self.assertRaises(SystemExit):
            train.parse_config(raw(), "0" * 64)

    def test_rejects_missing_or_unknown_keys(self):
        for change in ({"checkpoint_every_steps": 250}, {"checkpoint_policy": None}):
            cfg = dict(CONFIG)
            for k, v in change.items():
                if v is None:
                    cfg.pop(k)
                else:
                    cfg[k] = v
            with self.subTest(change=change), self.assertRaises(SystemExit):
                train.parse_config(raw(cfg), None)

    def test_rejects_the_old_schema_and_mismatched_formats(self):
        for change in ({"schema_version": 1}, {"checkpoint_format": "ddp-pt"}, {"training_mode": "zero3"}):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                train.parse_config(raw({**CONFIG, **change}), None)


class CheckpointEvery(unittest.TestCase):
    def test_long_runs_save_more_often_than_short_runs(self):
        # 10000 step × 0.5 s = 5000 s, τ 844 s → 5번, 2000 step마다
        self.assertEqual(train.checkpoint_every(10000, 0.5, 844), (2000, 5))
        # 1000 step × 0.5 s = 500 s < τ → 1번 (마지막 step)
        self.assertEqual(train.checkpoint_every(1000, 0.5, 844), (1000, 1))

    def test_validation_settings_give_several_saves_in_minutes(self):
        # 검증용 τ 141 s, 15분 학습(1800 step × 0.5 s) → 6번
        every, count = train.checkpoint_every(1800, 0.5, 141)
        self.assertEqual(count, 6)
        self.assertEqual(every, 300)


class Helpers(unittest.TestCase):
    def test_total_steps_respects_max_steps(self):
        self.assertEqual(train.total_steps(100, 10, None), 1000)
        self.assertEqual(train.total_steps(100, 10, 250), 250)

    def test_lr_warms_up_then_decays(self):
        self.assertAlmostEqual(train.lr_at(50, 1.0, 100, 1000), 0.5)
        self.assertAlmostEqual(train.lr_at(1000, 1.0, 100, 1000), 0.1)
        self.assertAlmostEqual(train.lr_at(1, 1.0, 0, 10), 1.0, places=1)

    def test_split_blocks_drops_the_tail_and_needs_one_block(self):
        self.assertEqual(train.split_blocks(list(range(7)), 3), [[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(SystemExit):
            train.split_blocks([1, 2], 3)

    def test_data_files_are_sorted_jsonl_only(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("b.jsonl", "a.jsonl", "note.txt"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual([os.path.basename(p) for p in train.data_files(d)], ["a.jsonl", "b.jsonl"])

    def test_wait_approved_sees_the_sidecar_marker(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "step-00000010.approved"), "w").close()
            self.assertTrue(train.wait_approved(d, 10, 1))
            self.assertFalse(train.wait_approved(d, 20, 0))


if __name__ == "__main__":
    unittest.main()
