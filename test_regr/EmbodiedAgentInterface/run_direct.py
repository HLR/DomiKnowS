import sys
import traceback
from pathlib import Path

p = Path(r"test_regr\EmbodiedAgentInterface").resolve()
sys.path.insert(0, str(p))
sys.path.insert(0, str(p.parents[1]))

import main

log_path = p / "run_direct_err.log"

try:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Starting main...\n")
    
    sys.argv = [
        "main.py",
        "--dataset", "all",
        "--limit", "10",
        "--two-stage",
        "--epochs", "2",
        "--rl-epochs", "2",
        "--max-steps", "30",
        "--evaluate",
    ]
    main.main()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("main completed successfully!\n")
except Exception as e:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Exception caught: {e}\n")
        f.write(traceback.format_exc())
except BaseException as be:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"BaseException caught: {be}\n")
        f.write(traceback.format_exc())
