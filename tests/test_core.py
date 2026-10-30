import unittest
from arc_r2.grid import as_grid, GridError
from arc_r2.transforms import reverse_horizontal, reverse_vertical, negate_polarity, reflect_main_diagonal
from arc_r2.dsl import Program, parse_program
from arc_r2.search import SearchLimits, beam_search, mcts_search, SearchStats
from arc_r2.solver import Example, HybridSolver, score_program
from arc_r2.objects import detect_objects
from arc_r2.relations import relation_graph
from arc_r2.prte import reverse_polarity

class GridTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(as_grid([[0, 1], [2, 3]]), ((0,1),(2,3)))
        with self.assertRaises(GridError): as_grid([[1], [1,2]])
        with self.assertRaises(GridError): as_grid([[10]])
    def test_reversal_is_not_transpose(self):
        g = as_grid([[1,2,3],[4,5,6]])
        self.assertEqual(reverse_horizontal(g), ((3,2,1),(6,5,4)))
        self.assertEqual(reverse_vertical(g), ((4,5,6),(1,2,3)))
        self.assertEqual(reflect_main_diagonal(g), ((1,4),(2,5),(3,6)))
    def test_polarity(self):
        self.assertEqual(negate_polarity(as_grid([[0,1,2]])), ((1,0,2),))

class ObjectAndPRTETests(unittest.TestCase):
    def test_object_detection_and_relations(self):
        grid = as_grid([[0,1,0,2],[0,1,0,2]])
        objects = detect_objects(grid)
        self.assertEqual(len(objects), 2)
        self.assertIn("LEFT_OF", relation_graph(objects)[0][2])
    def test_prte_uses_dominant_background(self):
        self.assertEqual(reverse_polarity(as_grid([[0,0,1],[0,0,1]])), ((1,1,0),(1,1,0)))

class DSLTests(unittest.TestCase):
    def test_parse_apply(self):
        p = parse_program("REVERSE.HORIZONTAL -> POLARITY.NEGATE")
        self.assertEqual(p.apply(as_grid([[0,1]])), ((0,1),))
        self.assertEqual(parse_program("IDENTITY").complexity, 0)
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError): parse_program("MAGIC.DO_THING")

class SearchTests(unittest.TestCase):
    def test_beam_and_mcts_bounded(self):
        limits = SearchLimits(max_depth=2, candidate_cap=30, mcts_iterations=10)
        score = lambda p: 1.0 if p.complexity == 1 else 0.2
        bs = SearchStats(); ms = SearchStats()
        self.assertTrue(beam_search(score, limits, bs))
        self.assertTrue(mcts_search(score, limits, ms))
        self.assertLessEqual(bs.evaluated, 30)
        self.assertLessEqual(ms.evaluated, 10)
    def test_prte_ablation_removes_operator(self):
        limits = SearchLimits(max_depth=1, candidate_cap=30, include_prte=False)
        ranked = beam_search(lambda p: 0.5, limits)
        self.assertTrue(ranked)
        self.assertTrue(all("PRTE.POLARITY.REVERSE" not in p.canonical() for _, p in ranked))
    def test_duplicate_state_scores_are_cached(self):
        limits = SearchLimits(max_depth=1, candidate_cap=20, mcts_iterations=12)
        grid = as_grid([[0]])
        calls = {"n": 0}
        def score(program):
            calls["n"] += 1
            return 1.0
        key = lambda program: program.apply(grid)
        stats = SearchStats()
        beam_search(score, limits, stats, state_key=key)
        self.assertEqual(stats.evaluated, 1)
        self.assertEqual(calls["n"], 1)

class IntegrationTests(unittest.TestCase):
    def test_hybrid_fits_training_examples(self):
        examples = (
            Example(as_grid([[0,1,0]]), as_grid([[0,1,0]])),
            Example(as_grid([[1,0,0]]), as_grid([[0,0,1]])),
        )
        result = HybridSolver(SearchLimits(max_depth=2, candidate_cap=100, mcts_iterations=20)).solve(examples)
        self.assertEqual(result.exact_examples, 2)
        self.assertGreater(result.candidates_evaluated, 0)
    def test_search_strategies_are_selectable(self):
        examples = (Example(as_grid([[0,1]]), as_grid([[1,0]])),)
        for strategy in ("identity", "beam", "mcts", "hybrid"):
            result = HybridSolver(SearchLimits(max_depth=1, candidate_cap=20, mcts_iterations=5), strategy=strategy).solve(examples)
            self.assertGreaterEqual(result.exact_examples, 0)

if __name__ == "__main__": unittest.main()
