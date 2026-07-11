from __future__ import annotations

import unittest
from pathlib import Path

import run_smolvlm2_video_benchmark as benchmark


class SmolVLM2VideoBenchmarkTests(unittest.TestCase):
    def test_normalize_answer(self) -> None:
        self.assertEqual(benchmark.normalize_answer(" Giraffe.\n"), "giraffe")

    def test_extracts_and_restricts_fallback_ops(self) -> None:
        log = "\n".join(
            [
                "clip_ctx: CLIP using ET backend",
                "load_tensors: offloaded 33/33 layers to GPU",
                "warmup:           IM2COL: type = f32, ne = [1 1 1 1]",
                "warmup:             NORM: type = f32, ne = [1 1 1 1]",
                "srv log_server_r: done request: POST /completion 127.0.0.1 200",
                "ggml_et_init: using device ET",
            ]
        )
        self.assertEqual(benchmark.unsupported_vision_ops(log), {"IM2COL", "NORM"})
        self.assertEqual(
            benchmark.log_failures(log, mode="board", request_count=1, allowed_ops={"IM2COL", "NORM"}),
            [],
        )
        failures = benchmark.log_failures(log, mode="board", request_count=1, allowed_ops={"NORM"})
        self.assertIn("board: unexpected vision fallback ops: IM2COL", failures)

    def test_correctness_requires_host_agreement_and_image_order(self) -> None:
        cases = [
            ({"name": "forward", "accepted_answers": ["giraffe"]}, [Path("cat"), Path("giraffe")]),
            ({"name": "reverse", "accepted_answers": ["cat"]}, [Path("giraffe"), Path("cat")]),
        ]
        host = [
            {"case": "forward", "normalized_answer": "giraffe"},
            {"case": "reverse", "normalized_answer": "cat"},
        ]
        board = [dict(item) for item in host]
        self.assertEqual(benchmark.correctness_failures(cases, host, board, ["forward", "reverse"]), [])

        board[1]["normalized_answer"] = "giraffe"
        failures = benchmark.correctness_failures(cases, host, board, ["forward", "reverse"])
        self.assertTrue(any("board answer" in failure for failure in failures))
        self.assertTrue(any("image-order check failed" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
