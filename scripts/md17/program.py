#!/usr/bin/env python3
"""Iterative ML experimentation agent — implements program.md via LangChain."""

import os
import subprocess
from pathlib import Path

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain.agents import create_react_agent

SCRIPTS_DIR = Path(__file__).parent.resolve()  # scripts/md17
REPO_ROOT = SCRIPTS_DIR.parents[1]             # .../aperol
BASE_SCRIPT = SCRIPTS_DIR / "run.py"
EXPERIMENTS_DIR = SCRIPTS_DIR / "experiments"


def _resolve(path: str) -> Path:
    """Resolve a path: absolute paths pass through; relative paths anchor to SCRIPTS_DIR."""
    p = Path(path)
    return p if p.is_absolute() else SCRIPTS_DIR / p


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file. Relative paths are resolved from the
    scripts/md17 directory (e.g. pass 'experiments/1/run.py')."""
    try:
        return _resolve(path).read_text()
    except FileNotFoundError:
        return f"Error: file not found: {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed. Relative paths
    are resolved from the scripts/md17 directory (e.g. pass 'experiments/1/run.py')."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {p}"


@tool
def run_experiment(n: int) -> str:
    """
    Train experiment {n} for one epoch. Automatically resumes from
    experiments/{n}/checkpoint.pt if it exists, then saves back to it. Call
    repeatedly to continue training. Returns up to 200 lines of stdout+stderr.
    Times out after 5 minutes.
    """
    script = EXPERIMENTS_DIR / str(n) / "run.py"
    checkpoint = script.parent / "checkpoint.pt"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    try:
        proc = subprocess.run(
            ["conda", "run", "-n", "aperol", "python", "-u", str(script),
             "--n_epoch", "1", "--checkpoint", str(checkpoint)],
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
    """List existing experiment folders (numeric) under scripts/md17/experiments/."""
    dirs = sorted(
        [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        if EXPERIMENTS_DIR.exists() else [],
        key=lambda d: int(d.name),
    )
    return "\n".join(d.name for d in dirs) if dirs else "none"


@tool
def read_metrics(n: int) -> str:
    """Return the per-epoch error log for experiment {n} as JSONL.
    Each line is a JSON object with keys: epoch, train_energy_error,
    train_force_error, val_energy_error, val_force_error."""
    path = EXPERIMENTS_DIR / str(n) / "metrics.jsonl"
    if not path.exists():
        return f"No metrics found for experiment {n}."
    return path.read_text()


tools = [read_file, write_file, run_experiment, list_experiments, read_metrics]

llm = ChatOpenRouter(
    model="openai/gpt-5.4-nano",
    max_retries=3,
)


system = (SCRIPTS_DIR / "program.md").read_text()
agent = create_react_agent(llm, tools, prompt=system)

if __name__ == "__main__":
    import time
    from langchain_core.messages import AIMessage

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            for chunk in agent.stream(
                {"messages": [("human", "Start iterating. After each run reflect on the error trend and improve.")]},
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    for msg in update.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            print(msg.content, flush=True)
            break
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt
            print(f"[attempt {attempt}/{max_attempts}] Error: {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
