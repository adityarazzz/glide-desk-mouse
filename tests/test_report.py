"""The test report's traceability matrix: every test point is backed by real tests, statuses are fair."""
import unittest
from pathlib import Path

from glide import report

ROOT = Path(__file__).resolve().parent.parent


def all_test_ids():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))

    def walk(s):
        for t in s:
            yield from walk(t) if isinstance(t, unittest.TestSuite) else [t.id().replace("tests.", "", 1)]
    return list(walk(suite))


class MatrixTests(unittest.TestCase):
    def test_every_test_point_names_tests_that_exist(self):
        ids = {i: "pass" for i in all_test_ids()}
        for p in report.POINTS:
            for prefix in p.get("tests", []):
                self.assertTrue(report.tests_for([prefix], ids), f"{p['id']}: no test matches {prefix}")

    def test_ids_are_unique_and_every_point_has_evidence_or_a_procedure(self):
        seen = [p["id"] for p in report.POINTS]
        self.assertEqual(len(seen), len(set(seen)))
        for p in report.POINTS:
            self.assertTrue(p.get("tests") or p.get("metric") or p.get("manual"), p["id"])

    def test_statuses(self):
        pts = report.POINTS
        report.POINTS = [
            dict(id="A", req="r", title="t", level="Unit", tests=["m.C"]),
            dict(id="B", req="r", title="t", level="Live", metric="x", rule=("<=", 300), target="v1.0"),
            dict(id="C", req="r", title="t", level="Live", metric="y", rule=(">=", 24)),
            dict(id="D", req="r", title="t", level="Manual", manual="do it"),
            dict(id="E", req="r", title="t", level="Unit", tests=["m.Missing"]),
        ]
        try:
            rows = {r["id"]: r["status"] for r in report.evaluate({"m.C.test_a": "pass", "m.C.test_b": "fail"},
                                                                  {"x": 316, "y": 30}, {"D": {"result": "pass"}})}
        finally:
            report.POINTS = pts
        self.assertEqual(rows, {"A": "FAIL", "B": "GAP (v1.0)", "C": "PASS", "D": "PASS", "E": "NO TESTS"})


if __name__ == "__main__":
    unittest.main()
