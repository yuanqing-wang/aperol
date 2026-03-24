#!/usr/bin/env python3
"""Iterative ML experimentation agent — implements program.md via LangChain."""

import os
import subprocess
from pathlib import Path

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

SCRIPTS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPTS_DIR.parents[1]
EXPERIMENTS_DIR = SCRIPTS_DIR / "experiments"


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPTS_DIR / p


@tool
def read_file(path: str) -> str:
    """Read a file. Relative paths resolve from scripts/md17/ (e.g. 'experiments/1/run.py')."""
    try:
        return _resolve(path).read_text()
    except FileNotFoundError:
        return f"Error: file not found: {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file, creating parent directories as needed. Relative paths resolve from scripts/md17/."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {p}"


@tool
def run_experiment(n: int) -> str:
    """Train experiment {n} for exactly one epoch. Resumes from checkpoint if it exists.
    Returns up to 200 lines of output. Times out after 5 minutes."""
    script = EXPERIMENTS_DIR / str(n) / "run.py"
    checkpoint = script.parent / "checkpoint.pt"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    try:
        proc = subprocess.run(
            ["conda", "run", "-n", "aperol", "python", "-u", str(script),
             "--n_epoch", "1", "--checkpoint", str(checkpoint)],
            capture_output=True, text=True, timeout=300, env=env, cwd=str(REPO_ROOT),
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        err = (exc.stderr or b"")
        output = (out if isinstance(out, str) else out.decode()) + \
                 (err if isinstance(err, str) else err.decode()) + \
                 "\n[timed out after 5 min]"
    lines = [l for l in output.strip().splitlines() if not l.startswith("wandb:")]
    return "\n".join(lines[:200])


@tool
def list_experiments() -> str:
    """List existing experiment folders under scripts/md17/experiments/."""
    dirs = sorted(
        [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        if EXPERIMENTS_DIR.exists() else [],
        key=lambda d: int(d.name),
    )
    return "\n".join(d.name for d in dirs) if dirs else "none"


@tool
def read_metrics(n: int) -> str:
    """Return the per-epoch error log for experiment {n} as JSONL (epoch, train_energy_error,
    train_force_error, val_energy_error, val_force_error)."""
    path = EXPERIMENTS_DIR / str(n) / "metrics.jsonl"
    return path.read_text() if path.exists() else f"No metrics found for experiment {n}."


llm = ChatOpenRouter(
    # model="openai/gpt-5.4-nano", 
    model="qwen/qwen3-coder:free",
    max_retries=3,
)
system = (SCRIPTS_DIR / "program.md").read_text()
agent = create_react_agent(llm, [read_file, write_file, run_experiment, list_experiments, read_metrics], prompt=system)


def _run_session():
    for chunk in agent.stream(
        {"messages": [("human", "Start iterating. After each run reflect on the error trend and improve. Never stop.")]},
        stream_mode="updates",
    ):
        for _, update in chunk.items():
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"[agent] {msg.content}", flush=True)
                    for tc in getattr(msg, "tool_calls", []):
                        args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                        print(f"[tool call] {tc['name']}({args})", flush=True)
                elif isinstance(msg, ToolMessage):
                    print(f"[tool result: {msg.name}]\n{msg.content[:500].rstrip()}", flush=True)


if __name__ == "__main__":
    session = 0
    while True:
        session += 1
        print(f"\n=== session {session} ===", flush=True)
        _run_session()
