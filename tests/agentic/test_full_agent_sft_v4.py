from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from castfactory.agentic import distillation_protocol as protocol
from castfactory.agentic.full_agent_v4 import execute, sft_rows
from castfactory.training import full_agent_sft_v4 as sft
full_agent_sft = sft
from tests.agentic.test_full_agent_v4 import case, plan, t2, Script, review_keep, STOP, KEEP


def trajectory(factorize=False, identity="example"):
    public = case()
    public["case_id"] = identity
    p = plan(cuts=[12] if factorize else None)
    turns = [("T1", protocol.render_reply("Public accepted plan.", p))]
    turns.extend(("T2", t2(span, 123.456789)) for span in p["spans"])
    turns.append(("T3", protocol.render_reply("Public evidence supports stopping.", STOP)))
    if factorize:
        turns.append(("T4", protocol.render_reply("Keep the real teacher values.", KEEP)))
    return execute(public, Script(turns), reviewer=review_keep)


def config():
    return dict(epochs=2, batch_size=3, gradient_accumulation_steps=2,
                learning_rate=2e-5, weight_decay=0.03, warmup_ratio=0.1,
                lr_scheduler_type="linear", max_grad_norm=1.0, seed=7,
                context_tokens=100, dtype="float32", gradient_checkpointing=False,
                device="cpu", report_to="none")


class FullAgentSFTV4Tests(unittest.TestCase):
    def write_rows(self, directory, rows):
        path = Path(directory) / "roles.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    @patch.object(protocol, "student_training_tokens", return_value=([1, 2], [3, 4, 5]))
    def test_production_dispatch_masks_only_native_response_and_preserves_values(self, tokens):
        row = sft_rows(trajectory())[1]
        encoded = full_agent_sft.encode_row(row, object(), 5)
        self.assertEqual(encoded["labels"], [-100, -100, 3, 4, 5])
        self.assertEqual(tokens.call_args.args[2], row["assistant_response"])
        self.assertEqual(protocol.parse_reply(row["assistant_response"], "student")["answer"]["values"],
                         [123.456789] * 24)
        with self.assertRaisesRegex(ValueError, "truncation"):
            sft.encode_row(row, object(), 4)

    @patch.object(protocol, "student_training_tokens", return_value=([1], [2]))
    def test_review_provenance_is_required_and_not_model_input(self, tokens):
        row = sft_rows(trajectory())[0]
        encoded = sft.encode_row(row, object())
        self.assertEqual(encoded["role"], "T1")
        self.assertEqual(tokens.call_args.args[1], row["messages"])
        broken = deepcopy(row)
        broken["t1_supervision"]["pass1_call_id"] = 9
        with self.assertRaisesRegex(ValueError, "review provenance"):
            sft.encode_row(broken, object())

    @patch.object(protocol, "student_training_tokens", return_value=([1], [2]))
    def test_preexported_t1_capacity_projection_preserves_teacher_and_target(self, tokens):
        row = sft_rows(trajectory())[0]
        old = deepcopy(row)
        message = old["messages"][1]
        marker, end = "Compute limits for this plan:\n", "\n\nForecast evidence:"
        before, _, tail = message["content"].partition(marker)
        policy_text, _, after = tail.partition(end)
        policy = json.loads(policy_text)
        policy.update(max_response_tokens=16384, max_context_tokens=None)
        message["content"] = before + marker + json.dumps(policy, separators=(",", ":")) + end + after
        old["serialization_provenance"].pop("public_policy_adapter", None)
        archived = deepcopy(old)
        adapted = sft.adapt_t1_public_policy(old)
        self.assertEqual(adapted["messages"], row["messages"])
        self.assertEqual(adapted["teacher_messages"], archived["teacher_messages"])
        self.assertEqual(adapted["assistant_response"], archived["assistant_response"])
        self.assertEqual(old, archived)
        feature = sft.encode_row(old, object())
        self.assertEqual(tokens.call_args.args[1], row["messages"])
        self.assertEqual(feature["serialization_provenance"]["public_policy_adapter"]["omitted_fields"],
                         ["max_context_tokens", "max_response_tokens"])
        old.pop("teacher_messages")
        missing_archive = deepcopy(old)
        feature = sft.encode_row(old, object())
        self.assertEqual(feature["serialization_provenance"]["public_policy_adapter"]["original_student_messages"],
                         missing_archive["messages"])
        self.assertEqual(old, missing_archive)
        self.assertEqual(sft.prompt_adapter_summary([feature])["original_student_requests_archived"], 1)

    @patch.object(protocol, "student_training_tokens", return_value=([1], [2]))
    def test_manifest_actual_counts_and_no_legacy_quotas(self, tokens):
        rows = sft_rows(trajectory(identity="direct")) + sft_rows(trajectory(True, "segmented"))
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, rows)
            manifest = Path(directory) / "manifest.json"
            data = {"profile_version": sft.VERSION, "row_file": path.name, "complete": 2, "role_rows": 7}
            manifest.write_text(json.dumps(data))
            features, report = full_agent_sft.load_release(manifest, object(), 10, profile_version=sft.VERSION)
            self.assertEqual(report["cases_by_mode"], {"DIRECT": 1, "FACTORIZE": 1})
            self.assertEqual(report["rows_by_role"], {"T1": 2, "T2": 3, "T3": 1, "T4": 1})
            self.assertEqual(sft.training_plan(features, sft.TrainingConfig(**config()))["planned_optimizer_steps"], 2)
            data["role_rows"] = 9
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                sft.load_release(manifest, object())

    @patch.object(protocol, "student_training_tokens", return_value=([1], [2]))
    def test_incomplete_collector_episodes_are_explicitly_excluded(self, tokens):
        with tempfile.TemporaryDirectory() as directory:
            episodes = Path(directory) / "episodes"
            episodes.mkdir()
            episodes.joinpath("good.json").write_text(json.dumps({"episode": trajectory()}))
            failed = trajectory(True, "failed")
            failed["normal_completion"] = False
            episodes.joinpath("bad.json").write_text(json.dumps({"episode": failed}))
            features, report = sft.load_release(directory, object())
            self.assertEqual(len(features), 2)
            self.assertEqual(report["skipped_incomplete_episodes"], 1)
            self.assertFalse(report["collection_complete"])

    def test_factorize_requires_real_t4_and_direct_does_not(self):
        rows = sft_rows(trajectory(True))[:-1]
        with self.assertRaisesRegex(ValueError, "T4 coverage"):
            sft._complete_rows(rows)
        rows = sft_rows(trajectory())
        self.assertEqual(sft._complete_rows(rows)[1], {"DIRECT": 1})

    @patch.object(protocol, "student_training_tokens", return_value=([1], [2]))
    def test_dry_run_and_fake_backend_full_supplied_manifest(self, tokens):
        direct, factor = sft_rows(trajectory()), sft_rows(trajectory(True))
        rows = []
        for index in range(1800):
            for row in deepcopy(factor if index % 2 else direct):
                row["case_id"] = row["episode_id"] = "manifest-case-" + str(index)
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(directory, rows)
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"profile_version": sft.VERSION, "row_file": path.name,
                                            "complete": 1800, "role_rows": 6300}))
            args = SimpleNamespace(**config(), dataset=manifest, out=Path(directory) / "train", model="unused",
                                   dry_run=True, profile_version=sft.VERSION, parameterization="full")
            def forbidden(*values):
                self.fail("dry-run reached training backend")
            report = sft.train(args, tokenizer=object(), backend=forbidden)
            self.assertEqual(report["planned_optimizer_steps"], 600)
            self.assertEqual(report["dataset"]["complete"], 1800)
            self.assertFalse(args.out.exists())
            args.dry_run = False
            calls = []
            def backend(arguments, configuration, features, tokenizer):
                calls.append((configuration, len(features)))
                return {"status": "fake_backend_smoke", "all_parameters_trainable": True}
            trained = sft.train(args, tokenizer=object(), backend=backend)
            self.assertEqual(trained["status"], "fake_backend_smoke")
            self.assertEqual(calls[0][1], 6300)

    def test_training_config_rejects_invalid_explicit_settings(self):
        for updates in ({"epochs": 0}, {"learning_rate": float("nan")}, {"warmup_ratio": 1.5}):
            with self.assertRaises(ValueError):
                sft.TrainingConfig(**{**config(), **updates})

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch"), "install training extras")
    def test_padding_has_no_response_loss(self):
        rows = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
                {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]}]
        batch = sft.collate_rows(rows, 0)
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 3], [-100, 5, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch"), "install training extras")
    def test_complete_episode_loss_preserves_token_then_episode_mean(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        class Model:
            config = SimpleNamespace(model_type="test")
            def __call__(self, input_ids, **kwargs):
                return SimpleNamespace(loss=parameter * input_ids[0, 0])
        def row(mean, tokens, identity):
            return {"episode_id": identity, "input_ids": [mean] + [9] * tokens,
                    "attention_mask": [1] * (tokens + 1), "labels": [-100] + [9] * tokens}
        features = [row(1, 1, "a"), row(3, 3, "a"), row(5, 2, "b")]
        batch = sft.collate_episodes(sft.group_episodes(features))
        loss = sft.episode_normalized_loss(Model(), batch["episodes"])
        # Episode a=(1*1+3*3)/4=2.5; b=5; episode mean=3.75.
        self.assertAlmostEqual(float(loss.detach()), 3.75)
        loss.backward()
        self.assertAlmostEqual(float(parameter.grad), 3.75)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch"), "install training extras")
    def test_uneven_two_plus_one_accumulation_preserves_episode_gradient(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        class Model:
            config = SimpleNamespace(model_type="test")
            def __call__(self, input_ids, **kwargs):
                return SimpleNamespace(loss=parameter * input_ids[0, 0])
        class Objective(sft.EpisodeObjectiveMixin):
            accelerator = SimpleNamespace(num_processes=1)
        features = [{"episode_id": str(mean), "input_ids": [mean, 9],
                     "attention_mask": [1, 1], "labels": [-100, 9]} for mean in (1, 3, 5)]
        episodes = sft.group_episodes(features)
        batches = [sft.collate_episodes(episodes[:2]), sft.collate_episodes(episodes[2:])]
        objective = Objective()
        denominator = objective._get_num_items_in_batch(batches, torch.device("cpu"))
        self.assertEqual(int(denominator), 3)
        losses = [objective.compute_loss(Model(), batch, num_items_in_batch=denominator) for batch in batches]
        for loss in losses:
            loss.backward()
        self.assertAlmostEqual(float(parameter.grad), 3.0)
        self.assertAlmostEqual(float(sum(loss.detach() for loss in losses)), 3.0)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch"), "install training extras")
    def test_native_selected_response_logits_equal_dense_causal_mask(self):
        import torch
        from torch.nn import functional as F
        torch.manual_seed(7)
        class Native(torch.nn.Module):
            config = SimpleNamespace(model_type="qwen3_5_text")
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.randn(5, 4))
            def forward(self, input_ids, logits_to_keep=None, **kwargs):
                logits = self.weights[input_ids]
                return SimpleNamespace(logits=logits if logits_to_keep is None else logits[:, logits_to_keep])
        model = Native()
        class Wrapped:
            module = model
            def __call__(self, **kwargs):
                self.positions = kwargs["logits_to_keep"].tolist()
                return model(**kwargs)
        wrapped = Wrapped()
        features = [{"episode_id": "a", "input_ids": [0, 1, 2, 3],
                     "attention_mask": [1] * 4, "labels": [-100, -100, 2, 3]},
                    {"episode_id": "a", "input_ids": [0, 2, 1],
                     "attention_mask": [1] * 3, "labels": [-100, -100, 1]}]
        batch = sft.collate_episodes(sft.group_episodes(features))
        selected = sft.episode_normalized_loss(wrapped, batch["episodes"])
        dense = sum(F.cross_entropy(model(**row).logits[:, :-1].reshape(-1, 4),
                                    row["labels"][:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                    for row in batch["episodes"][0]["rows"]) / 3
        self.assertTrue(torch.allclose(selected, dense))
        selected.backward()
        selected_grad = model.weights.grad.clone()
        model.weights.grad = None
        dense.backward()
        self.assertTrue(torch.allclose(selected_grad, model.weights.grad))
        self.assertEqual(wrapped.positions, [1])

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("torch"), "install training extras")
    def test_checkpoint_reload_probe_keeps_live_parameter_storage_and_optimizer_binding(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self, live=False):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.randn(4, 5))
                self.live = live
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.weights[input_ids][:, -1:])
            def to(self, *values, **kwargs):
                if self.live:
                    raise AssertionError("Checkpoint probe moved live training weights")
                return super().to(*values, **kwargs)
        model = Tiny(live=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        pointer, parameter = model.weights.data_ptr(), model.weights
        probes = {r: [1, 2] for r in sft.ROLES}
        def reload(*args, **kwargs):
            copied = Tiny()
            copied.load_state_dict(model.state_dict())
            return copied
        with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=reload):
            parity = sft.verify_saved_export(model, "/fake-checkpoint", probes, torch.float32)
        self.assertEqual(parity, {r: 0.0 for r in sft.ROLES})
        self.assertTrue(model.training)
        self.assertEqual(model.weights.data_ptr(), pointer)
        self.assertIs(model.weights, parameter)
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
        def bad_reload(*args, **kwargs):
            copied = reload(*args, **kwargs)
            with torch.no_grad():
                copied.weights.add_(0.5)
            return copied
        with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=bad_reload):
            with self.assertRaisesRegex(RuntimeError, "reload parity"):
                sft.verify_saved_export(model, "/fake-checkpoint", probes, torch.float32)
        self.assertTrue(model.training)
        self.assertEqual(model.weights.data_ptr(), pointer)
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)



    @unittest.skipUnless(Path(str(Path(__import__("os").environ.get("ELECLOAD_MODELS_DIR", "models")) / 'Qwen3.5-4B/tokenizer_config.json')).exists(),
                         "local native tokenizer unavailable")
    def test_native_agent_response_stops_at_eos_without_supervised_template_tail(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(Path(__import__("os").environ.get("ELECLOAD_MODELS_DIR", "models")) / "Qwen3.5-4B"), local_files_only=True)
        for row in sft_rows(trajectory(True)):
            feature = sft.encode_row(row, tokenizer)
            response = feature["input_ids"][feature["prompt_tokens"]:]
            self.assertEqual(response[-1], tokenizer.eos_token_id)
            self.assertEqual(response.count(tokenizer.eos_token_id), 1)


if __name__ == "__main__":
    unittest.main()
