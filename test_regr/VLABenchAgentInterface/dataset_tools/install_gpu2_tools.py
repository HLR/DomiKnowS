"""Install the versioned GPU2 maintenance tools into their server evidence directory."""
from pathlib import Path
import shutil

EVIDENCE_ROOT = Path("/home/auszok/experiments/vlabench_dataset_fix_20261029")
FILES = (
    "download_composite.py", "download_videos.py", "generate_planning.py",
    "generate_supplemental.py", "finalize_datasets.py",
    "refresh_dataset_report.py", "audit_planner_lcs.py", "document_template.md",
)

def main():
    if not Path("/workspace/test_regr/VLABenchAgentInterface").is_dir() or not Path("/opt/dominows/VLABench").is_dir():
        raise SystemExit("Run this GPU2-specific installer inside vigorous_easley from /workspace.")
    source = Path(__file__).with_name("gpu2")
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        origin = source / name
        target = EVIDENCE_ROOT / name
        if target.exists() and target.read_bytes() != origin.read_bytes():
            backup = EVIDENCE_ROOT / "pre_install_sources" / name
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
        shutil.copy2(origin, target)
    print(f"Installed {len(FILES)} versioned tools/templates in {EVIDENCE_ROOT}")
    print("No dataset download, scene generation, or training was started.")

if __name__ == "__main__":
    main()
