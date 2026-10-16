from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from three_level import experiment_analysis as analysis
from three_level import experiment_suite as suite
from three_level import train
from three_level.data import SyntheticCoTDataset
from three_level.evaluate import evaluate_teacher_forced
from three_level.model import CausalCoTTransformer


class PerStepExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.cfg = train.ExperimentConfig(
            train_size=16, batch_size=4, eval_interval=2, stage1_updates=2,
            stage2_updates=2, stage3_updates=2, uniform_updates=6, objective_family="per_step",
        )
        cls.artifacts = {}
        cls.coefficients = {}
        cls.step_losses = {}
        original_batch = train._batch_losses
        original_step = train._adam_step
        for method in suite.BASELINE_METHODS:
            observed = []
            values = []
            current = {}

            def batch(*args, **kwargs):
                losses = original_batch(*args, **kwargs)
                current["losses"] = losses
                return losses

            def step(optimizer, objective, tracker, prompts):
                losses = current["losses"]
                gradients = torch.autograd.grad(
                    objective, (losses.e1, losses.e2, losses.e3), retain_graph=True, allow_unused=True,
                )
                observed.append([0.0 if value is None else value.item() for value in gradients])
                values.append([losses.e1.item(), losses.e2.item(), losses.e3.item()])
                original_step(optimizer, objective, tracker, prompts)

            with mock.patch.object(train, "_batch_losses", side_effect=batch), mock.patch.object(train, "_adam_step", side_effect=step):
                cls.artifacts[method] = train.run_method(method, cls.cfg)
            cls.coefficients[method] = observed
            cls.step_losses[method] = values
        cls.cumulative_cfg = replace(cls.cfg, objective_family="cumulative")
        cls.cumulative_uniform = train.run_method("uniform", cls.cumulative_cfg)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_three_stages_use_individual_step_gradients(self):
        expected = [[1, 0, 0]] * 2 + [[0, 1, 0]] * 2 + [[0, 0, 1]] * 2
        self.assertEqual(self.coefficients["staged"], expected)

    def test_uniform_stays_the_same_joint_baseline_in_both_settings(self):
        for coefficients in self.coefficients["uniform"]:
            for value in coefficients:
                self.assertAlmostEqual(value, 1 / 3, places=6)
        new = self.artifacts["uniform"]
        for name in ("selected_state", "last_state"):
            for key, value in getattr(new, name).items():
                self.assertTrue(torch.equal(value, getattr(self.cumulative_uniform, name)[key]), key)

    def test_duals_constraints_and_thresholds_use_per_step_losses(self):
        artifacts = self.artifacts["constrained"]
        self.assertEqual(len(artifacts.dual_history), 4)
        self.assertEqual(self.coefficients["constrained"][:2], [[1, 0, 0]] * 2)
        for row in artifacts.dual_history:
            index = row["backward_calls"] - 1
            actual = self.coefficients["constrained"][index]
            expected = [row["lambda_after1"], 1, 0] if row["phase"] == "stage2" else [row["lambda_after1"], row["lambda_after2"], 1]
            for x, y in zip(actual, expected):
                self.assertAlmostEqual(x, y, places=6)
            self.assertEqual(row["objective_family"], "per_step")
            self.assertEqual(row["f_hat1"], self.step_losses["constrained"][index][0])
            if row["phase"] == "stage3":
                self.assertEqual(row["f_hat2"], self.step_losses["constrained"][index][1])
        transition = next(row for row in artifacts.history if row["phase"] == "stage3" and row["evaluation_reason"] == "stage_transition")
        self.assertAlmostEqual(artifacts.thresholds["b2"], transition["e2"] + self.cfg.tolerance)

    def test_evaluation_and_history_agree_with_saved_objective_family(self):
        dataset = SyntheticCoTDataset(16, seed=self.cfg.data_seed, sample_demo_count=True)
        model = CausalCoTTransformer()
        for method, artifacts in self.artifacts.items():
            self.assertEqual(artifacts.compute["backward_calls"], 6)
            for row in artifacts.history:
                for i in (1, 2, 3):
                    self.assertEqual(row[f"f{i}"], row[f"e{i}"])
            model.load_state_dict(artifacts.selected_state)
            measured = evaluate_teacher_forced(
                model, dataset, artifacts.scales, batch_size=4, device="cpu", objective_family="per_step",
            )
            self.assertEqual(measured, artifacts.selected_metrics, method)

    def test_suite_defaults_to_one_seed_three_methods_and_new_directory(self):
        args = suite.build_arg_parser().parse_args(["--objective-family", "per_step"])
        self.assertEqual(args.seeds, [11])
        self.assertEqual(args.methods, list(suite.BASELINE_METHODS))
        self.assertEqual(args.run_dir, Path("runs/per_step_objectives"))
        cfg = suite.SuiteConfig(**{**vars(args), "run_dir": str(args.run_dir)})
        for method in suite.BASELINE_METHODS:
            command = suite.build_train_command(method, 11, Path("checkpoint.pt"), cfg)
            self.assertEqual(command[command.index("--objective-family") + 1], "per_step")
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            suite.build_arg_parser().parse_args(["--objective-family", "per_step", "--methods", "emo"])
        old = suite.build_arg_parser().parse_args([])
        self.assertEqual(old.seeds, [11, 22, 33])
        self.assertEqual(old.methods, list(suite.ALL_METHODS))

    def test_checkpoint_family_is_checked_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = suite.build_arg_parser().parse_args(["--objective-family", "per_step", "--run-dir", directory])
            cfg = suite.SuiteConfig(**{**vars(args), "run_dir": directory})
            suite.initialize_run_dir(Path(directory), cfg)
            path = suite.job_paths(Path(directory), "uniform", 11).checkpoint
            train.save_artifacts(path, self.cumulative_uniform, self.cumulative_cfg)
            with self.assertRaisesRegex(ValueError, "Checkpoint.*objective family"):
                suite.run_train_and_eval("uniform", 11, cfg)

    def test_analysis_rejects_mixed_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train.save_artifacts(root / "checkpoints" / "uniform_seed11.pt", self.cumulative_uniform, self.cumulative_cfg)
            train.save_artifacts(root / "checkpoints" / "staged_seed11.pt", self.artifacts["staged"], self.cfg)
            with self.assertRaisesRegex(ValueError, "mix objective families"):
                analysis.generate_analysis(root, make_figures=False)

    def test_legacy_suite_config_defaults_to_cumulative(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            args = suite.build_arg_parser().parse_args(["--run-dir", directory])
            cfg = suite.SuiteConfig(**{**vars(args), "run_dir": directory})
            suite.initialize_run_dir(Path(directory), cfg)
            path = Path(directory) / "suite_config.json"
            payload = json.loads(path.read_text())
            payload.pop("objective_family")
            path.write_text(json.dumps(payload))
            suite.initialize_run_dir(Path(directory), cfg)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                suite.initialize_run_dir(Path(directory), replace(cfg, objective_family="per_step", methods=suite.BASELINE_METHODS))

    def test_analysis_rejects_evaluations_from_another_family(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train.save_artifacts(root / "checkpoints" / "staged_seed11.pt", self.artifacts["staged"], self.cfg)
            (root / "eval").mkdir()
            risk = {f"{mode}{i}": 1.0 for mode in ("rec", "tf") for i in (1, 2, 3)}
            (root / "eval" / "staged_seed11_eval_selected.json").write_text(json.dumps({
                "objective_family": "cumulative", "recursive_id": {"32": risk},
            }))
            with self.assertRaisesRegex(ValueError, "Evaluation objective family"):
                analysis.generate_analysis(root, make_figures=False)


if __name__ == "__main__":
    unittest.main()
