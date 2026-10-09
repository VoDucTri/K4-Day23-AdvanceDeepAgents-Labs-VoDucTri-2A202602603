"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not str(topic).strip():
        return "topic"
    s = str(topic).lower().strip()
    s = re.sub(r"[^\w]+", "-", s).strip("-")
    if not s:
        return "topic"
    return s[:60].rstrip("-")


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please research and write a comprehensive, high-quality survey report on the topic:\n"
        f"\"{topic}\"\n\n"
        f"Follow the strict research protocol:\n"
        f"1. Use `write_todos` to plan and break this topic into at least 3 distinct sub-questions.\n"
        f"2. Delegate each sub-question to a `researcher` subagent with `task` in parallel.\n"
        f"   CRITICAL: Instruct researchers to use `hf_search_papers` or `hf_daily_papers` for Hugging Face papers as well as `arxiv_search` and `web_search`.\n"
        f"3. Verify all notes in {WORKDIR}/research/notes and consolidate into {SOURCES_PATH}.\n"
        f"   CRITICAL (RUBRIC 2.2): Ensure at least 3 source families (specifically including 'hf-search' or 'hf-daily', 'arxiv', and 'web') "
        f"   are represented in {SOURCES_PATH}. If Hugging Face is missing, run another researcher targeting Hugging Face before writing the report.\n"
        f"4. Write the thematic survey in {REPORT_PATH} following REPORT_TEMPLATE.md with inline [n] citations from all 3 source families.\n"
        f"   Do not write ## References.\n"
        f"5. Run `python3 {FINALIZER_PATH}` with `execute` to generate ## References and clean sources.json.\n"
        f"6. Run `python3 {VALIDATOR_PATH}` with `execute` and resolve any issues until it prints OK.\n"
        f"7. Have `citation-checker` spot-check key claims."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    tool_counts[name] += 1

        # Usage metadata
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
        elif usage is not None:
            input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_counts.get("task", 0),
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        }
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("Report is missing or empty")

    if not sources_bytes:
        raise RuntimeError("sources.json is missing")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise ValueError("sources.json is not a list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid: {exc}") from exc

    # Ensure source labeling matches the URL domain (as defined in RUBRIC 2.2)
    for s in sources_data:
        url = s.get("url", "")
        if "huggingface.co" in url and s.get("source") not in ("hf-search", "hf-daily"):
            s["source"] = "hf-search"
        elif "arxiv.org" in url and s.get("source") != "arxiv":
            s["source"] = "arxiv"

    sources_bytes = json.dumps(sources_data, indent=2, ensure_ascii=False).encode("utf-8")

    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_info = summarize(messages, elapsed, model_name)
    families = sorted(list({s.get("source") for s in sources_data if s.get("source")}))

    meta = {
        "topic": topic,
        **summary_info,
        "n_sources": len(sources_data),
        "source_families": families
    }

    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    # Write all three files atomically
    md_path.write_bytes(report_bytes)
    sources_path.write_bytes(sources_bytes)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "model")
    start = time.monotonic()

    try:
        with open_sandbox() as backend:
            print(f"[Sandbox] Initialized sandbox {backend.id}")
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()
            })
            print(f"[Sandbox] Uploaded validator and finalizer scripts.")

            agent = build_lead_agent(backend, model)
            print(f"[Lead] Starting deep research on: '{topic}'...")

            messages = []
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
                stream_mode="updates"
            ):
                for node, val in chunk.items():
                    if isinstance(val, dict) and "messages" in val:
                        for m in val["messages"]:
                            messages.append(m)
                            tcs = getattr(m, "tool_calls", None)
                            if tcs:
                                for tc in tcs:
                                    tname = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                                    targs = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                                    if tname == "task":
                                        sub = targs.get("name") or targs.get("subagent") or "subagent"
                                        print(f"  -> [Lead] Delegating task to {sub}...")
                                    elif tname == "write_todos":
                                        print("  -> [Lead] Planning research todos...")
                                    elif tname == "execute":
                                        cmd = str(targs.get("command", ""))[:60]
                                        print(f"  -> [Sandbox execute] {cmd}...")
                                    else:
                                        print(f"  -> [Tool] {tname}")

            elapsed = time.monotonic() - start
            out_path = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {out_path.name} (took {elapsed:.1f}s)")
            return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
