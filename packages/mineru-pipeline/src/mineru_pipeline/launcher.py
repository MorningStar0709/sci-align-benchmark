"""Launch mineru_to_md.py as a completely detached process.
Usage: python pipeline/launcher.py --list-file data/textbook-batch-01.txt

Monitor: python pipeline/_health.py
         tail -5 data/logs/{stem}_stdout.log
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[4]  # restored Phase 5

# Default script path: prefer env var, fallback to project-relative path
_DEFAULT_SCRIPT = Path(os.environ.get(
    "MINERU_SCRIPT",
    str(WORK_DIR / "tools" / "mineru_to_md.py"),
))

OUTPUT_DIR = str(WORK_DIR / "data" / "processed")
LOG_DIR = WORK_DIR / "data" / "logs"

parser = argparse.ArgumentParser()
parser.add_argument("--list-file", required=True)
parser.add_argument("--output-dir", default=OUTPUT_DIR)
parser.add_argument("--script-path", default=None,
                    help="Override path to mineru_to_md.py (auto-detected if omitted)")
if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()

    # Resolve script path
    if args.script_path:
        SCRIPT = str(Path(args.script_path).resolve())
    else:
        script_candidate = _DEFAULT_SCRIPT
        if script_candidate.exists():
            SCRIPT = str(script_candidate.resolve())
        else:
            print(f"Error: default script not found at {script_candidate}", file=sys.stderr)
            print("Use --script-path to specify the correct path.", file=sys.stderr)
            sys.exit(1)

    list_file = str(Path(args.list_file).resolve())
    output_dir = args.output_dir
    stem = Path(list_file).stem
    log_file = str(LOG_DIR / f"{stem}_stdout.log")
    result_file = str(LOG_DIR / f"{stem}_result.json")
    failure_file = str(LOG_DIR / f"{stem}_failed.txt")

    cmd = [
        sys.executable, SCRIPT,
        "--list-file", list_file,
        "--output-dir", output_dir,
        "--skip-existing",
        "--result-file", result_file,
        "--failure-list", failure_file,
    ]

    with open(log_file, "w") as f:
        f.write(f"Launching: {' '.join(cmd)}\n")
        f.flush()
        popen_kwargs = {"start_new_session": True}  # 脱离控制终端，父会话退出不带走子进程

        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(WORK_DIR),
            **popen_kwargs,
        )
        f.write(f"PID={proc.pid}\n")

    print(f"PID={proc.pid}")
    print(f"Log: {log_file}")
    print("Monitor: python -c \"import json,urllib.request;print(json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health').read()))\"")
