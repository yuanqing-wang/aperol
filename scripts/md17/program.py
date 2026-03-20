#!/usr/bin/env python3
"""Iterative ML experimentation agent — implements program.md via LangChain."""

import os
import subprocess
import time
import collections
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent

SCRIPTS_DIR = Path.cwd()          # scripts/md17 — where run.sh is submitted from
REPO_ROOT = SCRIPTS_DIR.parents[1]  # .../aperol
BASE_SCRIPT = SCRIPTS_DIR / "run.py"


def _resolve(path: str) -> Path:
    """Resolve a path: absolute paths pass through; relative paths anchor to SCRIPTS_DIR."""
    p = Path(path)
    return p if p.is_absolute() else SCRIPTS_DIR / p


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file. Relative paths are resolved from the
    scripts/md17 directory (e.g. pass '1/run.py', not 'scripts/md17/1/run.py')."""
    try:
        return _resolve(path).read_text()
    except FileNotFoundError:
        return f"Error: file not found: {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed. Relative paths
    are resolved from the scripts/md17 directory (e.g. pass '1/run.py')."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {p}"


@tool
def run_experiment(n: int) -> str:
    """
    Run {n}/run.py in the aperol conda environment with
    the root of the current directory on PYTHONPATH. 
    Returns up to 200 lines of
    stdout+stderr. Times out after 5 minutes.
    """
    script = SCRIPTS_DIR / str(n) / "run.py"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    try:
        proc = subprocess.run(
            ["conda", "run", "-n", "aperol", "python", "-u", str(script)],
            capture_output=True, text=True, timeout=300, env=env,
            cwd=str(REPO_ROOT),
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        output = (out if isinstance(out, str) else out.decode()) + \
                 (err if isinstance(err, str) else err.decode()) + \
                 "\n[timed out after 5 min]"
    return "\n".join(output.strip().splitlines()[:200])


@tool
def list_experiments() -> str:
    """List existing experiment folders (numeric) under scripts/md17/."""
    dirs = sorted(
        [d for d in SCRIPTS_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    return "\n".join(d.name for d in dirs) if dirs else "none"


tools = [read_file, write_file, run_experiment, list_experiments]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=8192)
system = (SCRIPTS_DIR / "program.md").read_text()
agent = create_agent(llm, tools, system_prompt=system)

# Sliding-window TPM limiter: stay under this many tokens per minute.
TPM_LIMIT = 20_000

if __name__ == "__main__":
    # Each entry is (timestamp, token_count) for the last 60 s.
    token_window: collections.deque = collections.deque()

    for chunk in agent.stream({
        "messages": [(
            "user",
            "Start iterating. Keep n_epoch ≤ 10 for fast turnaround. "
            "After each run reflect on the val_f / val_e trend and improve."
        )]
    }):
        print(chunk)
        for msg in chunk.get("model", {}).get("messages", []):
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                continue
            tokens = usage.get("total_tokens", 0)
            now = time.monotonic()
            token_window.append((now, tokens))
            # Drop entries older than 60 s.
            while token_window and token_window[0][0] < now - 60:
                token_window.popleft()
            used = sum(t for _, t in token_window)
            print(f"[tokens] last-60s={used}/{TPM_LIMIT}")
            if used >= TPM_LIMIT:
                sleep_for = 60 - (now - token_window[0][0]) + 1
                print(f"[rate limit] sleeping {sleep_for:.1f}s …")
                time.sleep(sleep_for)
