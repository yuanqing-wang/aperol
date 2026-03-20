#!/usr/bin/env python3
"""Iterative ML experimentation agent — implements program.md via LangChain."""

import os
import subprocess
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent

REPO_ROOT = Path(__file__).resolve().parents[2]   # .../aperol
SCRIPTS_DIR = Path(__file__).resolve().parent     # .../scripts/md17
BASE_SCRIPT = SCRIPTS_DIR / "run.py"


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"Error: file not found: {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    p = Path(path)
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

llm = ChatAnthropic(model="claude-opus-4-6", max_tokens=8192)
system = (SCRIPTS_DIR / "program.md").read_text()
agent = create_agent(llm, tools, system_prompt=system)

if __name__ == "__main__":
    for chunk in agent.stream({
        "messages": [(
            "user",
            "Start iterating. Keep n_epoch ≤ 10 for fast turnaround. "
            "After each run reflect on the val_f / val_e trend and improve."
        )]
    }):
        print(chunk)
