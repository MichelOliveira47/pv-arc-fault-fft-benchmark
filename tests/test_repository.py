from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_main_script_parses(self) -> None:
        source = (ROOT / "src" / "pv_arc_fault_fft_benchmark.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_validator_parses(self) -> None:
        source = (ROOT / "scripts" / "check_input_data.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_expected_repository_files_exist(self) -> None:
        expected = [
            "README.md",
            "CITATION.cff",
            ".zenodo.json",
            "assets/graphical_abstract.png",
            "data/README.md",
            "env/requirements.txt",
            "env/requirements-article.txt",
        ]
        for relative_path in expected:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_requirements_cover_core_stack(self) -> None:
        text = (ROOT / "env" / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("numpy", "pandas", "scikit-learn", "scikeras", "tensorflow"):
            with self.subTest(package=package):
                self.assertIn(package, text)

    def test_restricted_data_is_ignored(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data_labeled/", gitignore)
        self.assertIn("data_labeled.zip", gitignore)


if __name__ == "__main__":
    unittest.main()
