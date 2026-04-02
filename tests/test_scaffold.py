from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reddit_pain_agent.config import build_run_paths, ensure_repo_layout, slugify
from reddit_pain_agent.main import init_run


class ScaffoldTests(unittest.TestCase):
    def test_slugify_normalizes_names(self) -> None:
        self.assertEqual(slugify("Vibe Coding Pain Run"), "vibe-coding-pain-run")

    def test_build_run_paths_uses_expected_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_repo_layout(root)
            run_paths = build_run_paths("Founder Workflow Pain", root)
            self.assertEqual(run_paths.output_dir, root / "outputs" / "runs" / "founder-workflow-pain")
            self.assertEqual(run_paths.final_memo_path.name, "final_memo.md")

    def test_init_run_creates_brief_and_memo_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, created = init_run("Reddit First Run", root)
            self.assertTrue(Path(created["brief"]).exists())
            self.assertTrue(Path(created["final_memo"]).exists())


if __name__ == "__main__":
    unittest.main()
