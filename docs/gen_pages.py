from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "index.md": ROOT / "USER_GUIDE.md",
    "deployment.md": ROOT / "DEPLOYMENT_AND_TECHNICAL_REPORT.md",
    "user-guide.md": ROOT / "USER_GUIDE.md",
    "THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
}


for output_path, source_path in PAGES.items():
    with mkdocs_gen_files.open(output_path, "w") as page:
        page.write(source_path.read_text(encoding="utf-8"))
