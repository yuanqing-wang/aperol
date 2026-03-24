#!/usr/bin/env python3
"""Iterative ML experimentation agent — implements program.md via LangChain."""

import os
import subprocess
import threading
from pathlib import Path

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

SCRIPTS_DIR = Path(__file__).parent.resolve()  # scripts/md17
REPO_ROOT = SCRIPTS_DIR.parents[1]             # .../aperol
BASE_SCRIPT = SCRIPTS_DIR / "run.py"
EXPERIMENTS_DIR = SCRIPTS_DIR / "experiments"


# background jobs: n -> {"proc": Popen, "buf": list[str], "done": bool}
_jobs: dict[int, dict] = {}
_jobs_lock = threading.Lock()


def _drain(stream, buf: list):
    for line in stream:
        buf.append(line)
    stream.close()


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
def start_experiment(n: int) -> str:
    """
    Start training experiment {n} for one epoch in the background and return
    immediately. Automatically resumes from experiments/{n}/checkpoint.pt if it
    exists. Use poll_experiment(n) to check progress and retrieve output.
    Returns an error if experiment {n} is already running.
    """
    with _jobs_lock:
        job = _jobs.get(n)
        if job and not job["done"]:
            return f"Experiment {n} is already running."
        script = EXPERIMENTS_DIR / str(n) / "run.py"
        checkpoint = script.parent / "checkpoint.pt"
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.Popen(
            ["conda", "run", "-n", "aperol", "python", "-u", str(script),
             "--n_epoch", "1", "--checkpoint", str(checkpoint)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd=str(REPO_ROOT),
        )
        buf: list[str] = []
        t = threading.Thread(target=_drain, args=(proc.stdout, buf), daemon=True)
        t.start()
        _jobs[n] = {"proc": proc, "buf": buf, "thread": t, "done": False}
    return f"Experiment {n} started (pid {proc.pid})."


@tool
def poll_experiment(n: int) -> str:
    """
    Check the status of a background experiment {n} and return its output so far
    (up to 200 lines). Reports whether it is still running or has finished
    (including exit code). Safe to call multiple times.
    """
    with _jobs_lock:
        job = _jobs.get(n)
    if job is None:
        return f"No record of experiment {n}. Use start_experiment({n}) first."
    proc: subprocess.Popen = job["proc"]
    buf: list[str] = job["buf"]
    rc = proc.poll()
    if rc is None:
        status = "running"
    else:
        job["done"] = True
        status = f"finished (exit code {rc})"
    lines = "".join(buf).strip().splitlines()
    output = "\n".join(lines[:200])
    return f"[experiment {n}: {status}]\n{output}"


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


tools = [read_file, write_file, start_experiment, poll_experiment, list_experiments, read_metrics]

llm = ChatOpenRouter(
    model="openai/gpt-5.4-nano",
    max_retries=3,
)


system = (SCRIPTS_DIR / "program.md").read_text()
agent = create_react_agent(llm, tools, prompt=system)

def _stream_agent(messages: list) -> list:
    """Run one agent session, printing all output. Returns the final message list."""
    from langchain_core.messages import AIMessage, ToolMessage

    state = {"messages": messages}
    for chunk in agent.stream(state, stream_mode="updates"):
        for _, update in chunk.items():
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"[agent] {msg.content}", flush=True)
                    for tc in getattr(msg, "tool_calls", []):
                        args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                        print(f"[tool call] {tc['name']}({args})", flush=True)
                elif isinstance(msg, ToolMessage):
                    preview = msg.content[:500].rstrip()
                    print(f"[tool result: {msg.name}]\n{preview}", flush=True)
                messages = messages + [msg]
    return messages


if __name__ == "__main__":
    messages = [("human", "Start iterating. After each run reflect on the error trend and improve. Never stop.")]
    session = 0
    while True:
        session += 1
        print(f"\n=== session {session} ===", flush=True)
        messages = _stream_agent(messages)
        # nudge the agent to keep going with full context preserved
        messages = messages + [("human", "Continue. Analyse all experiments so far and keep iterating.")]
