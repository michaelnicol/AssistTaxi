"""The painted-stroke cost: exact lines, then one real LLAMAS photo."""

import json
import time
import unittest
from dataclasses import replace
from pathlib import Path

from spline.lanes import DEFAULT_JUDGE, read_lanes, stroke_points
from spline.score import measure

SPLIT = Path("/home/michaelnicol/github/MilestoneTwoExperiments/Qwen3-VL-8B-Splines/output/split.json")


def _judge(stroke_px: float):
    return replace(DEFAULT_JUDGE, stroke_px=stroke_px)


class StrokeCostTest(unittest.TestCase):
    def test_copy_costs_zero(self) -> None:
        gold = "10 0 10 40"
        one = measure(gold, gold, 30, 41, _judge(1))
        self.assertEqual(one.cost_px, 0.0)

    def test_shift_of_four(self) -> None:
        one = measure("10 0 10 40", "14 0 14 40", 30, 41, _judge(1))
        self.assertEqual(one.cost_px, 16.0)

    def test_empty_answer(self) -> None:
        one = measure("10 0 10 40", "", 30, 41, _judge(1))
        self.assertEqual(one.cost_px, 2500.0)
        self.assertEqual(one.prec, 0.0)
        self.assertEqual(one.rec, 0.0)

    def test_missed_gold_line_does_not_raise_cost(self) -> None:
        one = measure("10 0 10 40\n80 0 80 40", "10 0 10 40", 100, 41, _judge(1))
        self.assertEqual(one.cost_px, 0.0)
        self.assertLess(one.rec, 1.0)

    def test_extra_line_raises_cost(self) -> None:
        one = measure("10 0 10 40", "10 0 10 40\n80 0 80 40", 100, 41, _judge(1))
        self.assertGreater(one.cost_px, 0.0)

    def test_null_stroke_keeps_the_capped_cost(self) -> None:
        one = measure("10 0 10 40", "", 30, 41, replace(DEFAULT_JUDGE, stroke_px=None))
        self.assertEqual(one.cost_px, 50.0)

    def test_real_photo_matches_itself(self) -> None:
        self.assertEqual(DEFAULT_JUDGE.stroke_px, 3.0)
        split = json.loads(SPLIT.read_text())
        _path, gold, width, height = next(row for row in split["train"] if row[2] == 1276 and row[3] == 717)
        lanes = read_lanes(gold, width, height)
        pixels = stroke_points(lanes, width, height, DEFAULT_JUDGE)
        self.assertGreater(len(pixels), 0)
        self.assertLess(len(pixels), width * height / 10)
        started = time.perf_counter()
        one = measure(gold, gold, width, height, DEFAULT_JUDGE)
        elapsed = time.perf_counter() - started
        self.assertEqual(one.cost_px, 0.0)
        self.assertLess(elapsed, 2.0, f"paint and distance took {elapsed:.2f}s")
