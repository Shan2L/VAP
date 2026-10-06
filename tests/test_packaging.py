from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_every_top_level_module_is_installed(self) -> None:
        listed = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "tool"
        ]["setuptools"]["py-modules"]
        # The editable install only exposes listed modules to the `vap` command.
        self.assertEqual(
            sorted(listed), sorted(path.stem for path in ROOT.glob("*.py"))
        )


if __name__ == "__main__":
    unittest.main()
