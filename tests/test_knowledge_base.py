import unittest

from arc_r2.knowledge_base import get_entry, iter_entries, search_knowledge


class KnowledgeBaseTests(unittest.TestCase):
    def test_catalog_is_nonempty_and_keys_are_unique(self):
        entries = iter_entries()
        self.assertGreaterEqual(len(entries), 6)
        self.assertEqual(len({entry.key for entry in entries}), len(entries))

    def test_get_entry_returns_known_source(self):
        entry = get_entry("constraint-processing")
        self.assertEqual(entry.discipline, "constraint satisfaction")
        self.assertIn("Dechter", entry.source_title)

    def test_search_finds_graph_mapping(self):
        results = search_knowledge("graph matching")
        self.assertTrue(results)
        self.assertTrue(any(entry.key == "graph-structures" for entry in results))

    def test_search_is_case_insensitive(self):
        self.assertEqual(search_knowledge("MORPHOLOGY"), search_knowledge("morphology"))

    def test_empty_search_returns_no_results(self):
        self.assertEqual(search_knowledge("   "), ())

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            get_entry("not-a-real-entry")


if __name__ == "__main__":
    unittest.main()
