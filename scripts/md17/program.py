#!/usr/bin/env python3
"""Iterative ML experimentation agent — implements program.md via LangChain."""

import os
import shutil
import subprocess
from pathlib import Path

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

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
def new_experiment(n: int, source: int | None = None) -> str:
    """Create experiment {n} by copying run.py from experiment {source} (or the base run.py if source is None).
    Returns the path written. Edit the file with write_file before calling run_experiment."""
    dest = EXPERIMENTS_DIR / str(n) / "run.py"
    if dest.exists():
        return f"Error: experiments/{n}/run.py already exists."
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = (EXPERIMENTS_DIR / str(source) / "run.py") if source is not None else (SCRIPTS_DIR / "run.py")
    if not src.exists():
        return f"Error: source not found: {src}"
    shutil.copy(src, dest)
    return f"Copied {src} → {dest}"


@tool
def run_experiment(n: int, epochs: int = 1) -> str:
    """Train experiment {n} for a given number of epochs (default 1). Resumes from checkpoint if it exists.
    Returns up to 200 lines of output. Times out after 5 minutes per epoch."""
    script = EXPERIMENTS_DIR / str(n) / "run.py"
    checkpoint = script.parent / "checkpoint.pt"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    try:
        proc = subprocess.run(
            ["conda", "run", "-n", "aperol", "python", "-u", str(script),
             "--n_epoch", str(epochs), "--checkpoint", str(checkpoint)],
            capture_output=True, text=True, timeout=300 * epochs, env=env, cwd=str(REPO_ROOT),
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



MODEL = "openai/gpt-5.4-nano"
# MODEL = "qwen/qwen3.5-9b"
# MODEL = "minimax/minimax-m2.5:free"

tools = [new_experiment, read_file, write_file, run_experiment, list_experiments, read_metrics]
system = (SCRIPTS_DIR / "program.md").read_text()

agent = create_react_agent(
    ChatOpenRouter(
        model=MODEL,
        max_retries=1,
        # request_timeout=60,
        # reasoning={"effort": "none"},
    ),
    tools,
    prompt=system,
    checkpointer=MemorySaver(),
)

THREAD = {"configurable": {"thread_id": "main"}}


def _stream_turn(human_msg: str) -> None:
    for chunk in agent.stream(
        {"messages": [("human", human_msg)]},
        config=THREAD,
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
    _stream_turn("Start iterating. After each run reflect on the error trend and improve. Never stop.")
    while True:
        try:
            _stream_turn("Continue.")
        except Exception as e:
            print(f"[error] {MODEL} — {e}", flush=True)
