"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL_RE = re.compile(r"https?://\S+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources:
        return ["no sources in sources.json"]

    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        return ["sources.json must be a list of objects"]

    seen_urls = set()
    source_by_n = {}
    for entry in sources:
        n = entry.get("n")
        url = entry.get("url")

        if not isinstance(n, int):
            problems.append(f"source {entry} has non-integer n={n!r}")
            continue

        if n in source_by_n:
            problems.append(f"duplicate source number [{n}] in sources.json")
        source_by_n[n] = entry

        if not url or not (str(url).startswith("http://") or str(url).startswith("https://")):
            problems.append(f"source [{n}] has invalid or missing url={url!r}")

        if url:
            if url in seen_urls:
                problems.append(f"duplicate url in sources: {url}")
            seen_urls.add(url)

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading in report")
        return problems

    ref_start = matches[-1].end()
    body = report_text[:matches[-1].start()]
    refs_text = report_text[ref_start:]

    # Extract citations in body, skipping code blocks and markdown links
    segments = _CODE.split(body)
    cited_numbers = set()
    for i, segment in enumerate(segments):
        if i % 2:  # inside code block
            continue
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited_numbers.add(num)

    # Check citations against sources
    for n in sorted(cited_numbers):
        if n not in source_by_n:
            problems.append(f"[{n}] cited but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited_numbers:
            problems.append(f"source [{n}] never cited")

    # Check References section lines
    ref_lines = [line.strip() for line in refs_text.splitlines() if line.strip()]
    seen_ref_ns = set()
    for line in ref_lines:
        m = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if not m:
            continue
        num = int(m.group(1))
        content = m.group(2)

        if num in seen_ref_ns:
            problems.append(f"reference [{num}] appears multiple times in References section")
        seen_ref_ns.add(num)

        if num not in source_by_n:
            problems.append(f"reference [{num}] is not in sources.json")
            continue

        # Check URL in this reference line
        urls_in_line = _URL_RE.findall(line)
        # Clean trailing punctuation from URLs
        clean_urls = [re.sub(r"[,\.\)\]]+$", "", u) for u in urls_in_line]
        if len(clean_urls) == 0:
            problems.append(f"reference line [{num}] has no URL")
        elif len(clean_urls) > 1:
            problems.append(f"reference line [{num}] bundles multiple URLs: {clean_urls}")
        else:
            expected_url = source_by_n[num].get("url", "").rstrip(".,)]")
            if clean_urls[0] != expected_url:
                problems.append(f"reference line [{num}] URL mismatch: expected {expected_url}, got {clean_urls[0]}")

    for n in sorted(source_by_n.keys()):
        if n not in seen_ref_ns:
            problems.append(f"source [{n}] is missing from References section")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
