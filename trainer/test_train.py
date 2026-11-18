"""train.py의 순수 함수 테스트 (torch · GPU 불필요): cd trainer && python3 -m unittest test_train.py"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import devoops_ckpt as ckpt  # noqa: E402
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
        cfg, digest = ckpt.parse_config(raw(), hashlib.sha256(raw()).hexdigest())
        self.assertEqual(cfg.checkpoint_policy.interval_seconds, 844)
        self.assertEqual(cfg.dataset.format, "text-jsonl")
        self.assertEqual(digest, hashlib.sha256(raw()).hexdigest())

    def test_rejects_a_changed_config(self):
        with self.assertRaises(SystemExit):
            ckpt.parse_config(raw(), "0" * 64)

    def test_rejects_missing_or_unknown_keys(self):
        for change in ({"checkpoint_every_steps": 250}, {"checkpoint_policy": None}):
            cfg = dict(CONFIG)
            for k, v in change.items():
                if v is None:
                    cfg.pop(k)
                else:
                    cfg[k] = v
            with self.subTest(change=change), self.assertRaises(SystemExit):
                ckpt.parse_config(raw(cfg), None)

    def test_rejects_the_old_schema_and_mismatched_formats(self):
        for change in ({"schema_version": 1}, {"checkpoint_format": "ddp-pt"}, {"training_mode": "zero3"}):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                ckpt.parse_config(raw({**CONFIG, **change}), None)


class CheckpointEvery(unittest.TestCase):
    def test_long_runs_save_more_often_than_short_runs(self):
        # 10000 step × 0.5 s = 5000 s, τ 844 s → 5번, 2000 step마다
        self.assertEqual(ckpt.checkpoint_every(10000, 0.5, 844), (2000, 5))
        # 1000 step × 0.5 s = 500 s < τ → 1번 (마지막 step)
        self.assertEqual(ckpt.checkpoint_every(1000, 0.5, 844), (1000, 1))

    def test_validation_settings_give_several_saves_in_minutes(self):
        # 검증용 τ 141 s, 15분 학습(1800 step × 0.5 s) → 6번
        every, count = ckpt.checkpoint_every(1800, 0.5, 141)
        self.assertEqual(count, 6)
        self.assertEqual(every, 300)


class Helpers(unittest.TestCase):
    def test_total_steps_respects_max_steps(self):
        self.assertEqual(ckpt.total_steps(100, 10, None), 1000)
        self.assertEqual(ckpt.total_steps(100, 10, 250), 250)

    def test_lr_warms_up_then_decays(self):
        self.assertAlmostEqual(ckpt.lr_at(50, 1.0, 100, 1000), 0.5)
        self.assertAlmostEqual(ckpt.lr_at(1000, 1.0, 100, 1000), 0.1)
        self.assertAlmostEqual(ckpt.lr_at(1, 1.0, 0, 10), 1.0, places=1)

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
            self.assertTrue(ckpt.wait_approved(d, 10, 1))
            self.assertFalse(ckpt.wait_approved(d, 20, 0))


class Resume(unittest.TestCase):
    """ckpt-init이 남기는 resume.json (checkpoint/ckpt/init_main.py _write_resume) 읽기."""

    def write(self, d, body):
        with open(os.path.join(d, "resume.json"), "w") as f:
            json.dump(body, f)

    def test_no_resume_file_means_start_from_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ckpt.read_resume(d), (0, None))

    def test_fresh_start_written_by_init(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"resume": False, "approved_gen": None, "approved_step": None, "step_dir": None, "start_step": 1})
            self.assertEqual(ckpt.read_resume(d), (0, None))

    def test_resume_returns_the_approved_step_and_folder(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "step-00001200"))
            self.write(d, {"resume": True, "approved_gen": 1, "approved_step": 1200, "step_dir": "step-00001200", "start_step": 1201})
            self.assertEqual(ckpt.read_resume(d), (1200, os.path.join(d, "step-00001200")))

    def test_resume_without_the_folder_fails_instead_of_starting_over(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, {"resume": True, "approved_gen": 1, "approved_step": 1200, "step_dir": "step-00001200", "start_step": 1201})
            with self.assertRaises(SystemExit):
                ckpt.read_resume(d)

    def test_resume_position_skips_batches_already_done_in_that_epoch(self):
        self.assertEqual(ckpt.resume_position(0, 100), (0, 0))
        self.assertEqual(ckpt.resume_position(250, 100), (2, 50))
        self.assertEqual(ckpt.resume_position(300, 100), (3, 0))


class ErrorLine(unittest.TestCase):
    """GPU-188: 실패 이유 한 줄."""

    def test_exception_type_and_message_on_one_line(self):
        self.assertEqual(ckpt.error_line(ValueError("bad\n  batch")), "ValueError: bad batch")
        self.assertEqual(ckpt.error_line(KeyError()), "KeyError")

    def test_system_exit_message_or_code(self):
        self.assertEqual(ckpt.error_line(SystemExit("WORLD_SIZE 2 != config.world_size 4")), "WORLD_SIZE 2 != config.world_size 4")
        self.assertEqual(ckpt.error_line(SystemExit(3)), "exit code 3")

    def test_long_reason_is_cut(self):
        self.assertEqual(len(ckpt.error_line(RuntimeError("x" * 900))), 500)


class MetricsHelpers(unittest.TestCase):
    """학습 지표 (GPU-96, Monitoring docs/training-metrics.md)."""

    def test_labels_come_from_job_env(self):
        env = {"JOB_ID": "3f9a1c2e", "GENERATION": "2", "JOB_COMPLETION_INDEX": "3", "RANK": "3"}
        self.assertEqual(ckpt.metric_labels(env), {"job_id": "3f9a1c2e", "generation": "2", "rank": "3"})

    def test_phase_is_warmup_until_warmup_steps(self):
        self.assertEqual(ckpt.phase_of(100, 100), "warmup")
        self.assertEqual(ckpt.phase_of(101, 100), "steady")

    def test_nccl_kernels_count_as_comm(self):
        class E:
            def __init__(self, name, us):
                self.name, self.self_device_time_total = name, us
        events = [E("ampere_sgemm_128x64", 600_000), E("ncclDevKernel_AllGather_RING_LL", 300_000), E("elementwise", 20_000)]
        self.assertEqual(ckpt.split_kernel_seconds(events), (0.62, 0.3))

    def test_metrics_off_is_a_no_op(self):
        m = ckpt.Metrics({"job_id": "j", "generation": "1", "rank": "0"}, port=0)
        self.assertFalse(m.on)
        m.stage("train"); m.step(1, 2.0, 0.5, 0.01, "warmup"); m.checkpoint(1); m.kernels([])  # noqa: E702


try:
    import prometheus_client  # noqa: F401
    HAVE_PROM = True
except ImportError:
    HAVE_PROM = False


@unittest.skipUnless(HAVE_PROM, "prometheus_client 없음 (trainer 이미지에는 있음)")
class MetricsExport(unittest.TestCase):
    def test_all_eight_metrics_are_exposed(self):
        import socket
        import urllib.request

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        m = ckpt.Metrics({"job_id": "3f9a1c2e", "generation": "1", "rank": "0"}, port=port)
        m.stage("checkpoint")
        m.step(5, 2.5, 0.8, 0.02, "steady")
        m.checkpoint(5)

        class E:
            name, self_device_time_total = "ncclKernel", 1000
        m.kernels([E()])
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
        for name in ("train_step_seconds_bucket", "train_data_wait_seconds_bucket", "train_rank_compute_seconds",
                     "train_rank_comm_seconds", "train_current_step", "train_loss", "train_stage",
                     "train_last_checkpoint_step"):
            self.assertIn(name, body)
        self.assertIn('train_stage{generation="1",job_id="3f9a1c2e",rank="0",stage="checkpoint"} 1.0', body)
        self.assertIn('train_step_seconds_bucket{generation="1",job_id="3f9a1c2e",le="1.0",phase="steady",rank="0"} 1.0', body)


if __name__ == "__main__":
    unittest.main()
