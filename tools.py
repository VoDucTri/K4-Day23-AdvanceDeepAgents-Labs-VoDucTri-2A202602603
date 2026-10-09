"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_last_arxiv_time = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _redact(text: str) -> str:
    """Never leak API keys in error strings."""
    for env_var in ("EXA_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LAB_API_KEY"):
        val = os.getenv(env_var, "").strip()
        if val and len(val) > 4 and val in text:
            text = text.replace(val, "[REDACTED]")
    return text


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Treat these as retryable: HTTP 429/500/502/503/504, RetryableError,
    httpx.TransportError. Read Retry-After header when present.
    Exponential backoff with jitter capped at `cap`. Last attempt re-raises without sleeping.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            is_retryable = False
            retry_after = None

            if isinstance(exc, RetryableError):
                is_retryable = True
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code in (429, 500, 502, 503, 504):
                    is_retryable = True
                    header_val = exc.response.headers.get("Retry-After")
                    if header_val:
                        try:
                            retry_after = float(header_val)
                        except ValueError:
                            pass
            elif isinstance(exc, httpx.TransportError):
                is_retryable = True

            if not is_retryable or attempt == attempts - 1:
                raise

            if retry_after is not None:
                delay = min(cap, max(0.5, retry_after))
            else:
                backoff = base * (2 ** attempt)
                jitter = random.uniform(0.1, 1.0)
                delay = min(cap, backoff + jitter)

            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_time
    try:
        # Keep only word characters and hyphens
        terms = re.findall(r"[\w\-]+", query)
        terms = [t for t in terms if t.lower() not in ("and", "or", "not")]
        if not terms:
            return "NO RESULTS"

        # Respect arXiv etiquette: at least 3 seconds between two calls
        elapsed = time.time() - _last_arxiv_time
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)

        search_query = " AND ".join(f"all:{t}" for t in terms)
        k = max(1, min(max_results, 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": k,
        }

        def fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        text = with_retry(fetch, attempts=5, base=2.0, cap=60.0)
        _last_arxiv_time = time.time()

        root = ET.fromstring(text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip()
            # Extract paper ID without version
            match = re.search(r"abs/([0-9]+\.[0-9]+|[\w\-]+/[0-9]+)(?:v\d+)?", raw_id)
            if match:
                paper_id = match.group(1)
            else:
                paper_id = raw_id.split("/")[-1].split("v")[0]

            url = f"https://arxiv.org/abs/{paper_id}"

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

            summary_elem = entry.find("atom:summary", ns)
            summary = " ".join(summary_elem.text.split())[:600] if summary_elem is not None and summary_elem.text else ""

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        k = max(1, min(limit, 100))
        params = {"limit": k}
        if date:
            params["date"] = date

        def fetch():
            with httpx.Client(timeout=25.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(fetch, attempts=4, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower()
        for item in items:
            paper = item.get("paper") or {}
            paper_id = paper.get("id") or item.get("id")
            if not paper_id:
                continue

            title = " ".join((paper.get("title") or item.get("title") or "").split())
            summary = " ".join((paper.get("summary") or item.get("summary") or "").split())[:600]

            if kw and kw not in (title + " " + summary).lower():
                continue

            published = (paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = paper.get("upvotes") or item.get("upvotes") or 0
            github = paper.get("githubRepo") or item.get("githubRepo") or ""
            stars = paper.get("githubStars") or item.get("githubStars") or 0

            records.append({
                "id": paper_id,
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": published,
                "title": title or "Untitled",
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars
            })

        if not records:
            return "NO RESULTS"

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not query.strip():
            return "NO RESULTS"

        k = max(1, min(limit, 50))
        params = {"q": query.strip(), "limit": k}

        def fetch():
            with httpx.Client(timeout=25.0) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(fetch, attempts=4, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") or item
            paper_id = paper.get("id")
            if not paper_id:
                continue

            title = " ".join((paper.get("title") or "").split())
            raw_summary = paper.get("ai_summary") or paper.get("summary") or ""
            summary = " ".join(raw_summary.split())[:600]

            published = (paper.get("publishedAt") or "")[:10]
            upvotes = paper.get("upvotes") or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars") or 0

            records.append({
                "id": paper_id,
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": published,
                "title": title or "Untitled",
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    """Execute Exa MCP tool via JSON-RPC HTTP POST, handling rate limits."""
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }

    def fetch():
        with httpx.Client(timeout=40.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = resp.text

            # Parse SSE or plain JSON
            data_line = ""
            for line in text.splitlines():
                if line.startswith("data:"):
                    data_line = line[len("data:"):].strip()
                    break
            raw_json = json.loads(data_line) if data_line else resp.json()

            if "error" in raw_json:
                err_msg = str(raw_json["error"])
                if "rate" in err_msg.lower() or "limit" in err_msg.lower():
                    raise RetryableError(f"Exa rate limit: {err_msg}", retry_after=20.0)
                raise RuntimeError(err_msg)

            res = raw_json.get("result", {})
            meta = res.get("_meta", {})

            # Check rate limiting in meta or content
            if meta.get("rateLimited") or meta.get("rate_limited") or "rate limit" in str(meta).lower():
                raise RetryableError("Exa rate limited in _meta", retry_after=20.0)

            contents = res.get("content", [])
            texts = []
            for c in contents:
                if isinstance(c, dict) and c.get("type") == "text":
                    t = c.get("text", "")
                    if "rate limit" in t.lower() and "exceeded" in t.lower():
                        raise RetryableError("Exa rate limit text", retry_after=20.0)
                    texts.append(t)

            return "\n\n".join(texts).strip()

    return with_retry(fetch, attempts=3, base=1.0, cap=15.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = query.strip()
        if not q:
            return "NO RESULTS"
        obj = objective.strip() if objective else f"Find authoritative research papers, surveys, benchmarks and articles about {q}"
        k = max(1, min(num_results, 10))

        content = _call_exa_mcp("web_search_exa", {"query": q, "objective": obj, "numResults": k})
        if not content:
            return "NO RESULTS"
        return content

    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        u = url.strip()
        if not u or not (u.startswith("http://") or u.startswith("https://")):
            return "ERROR: ValueError: invalid URL"

        content = _call_exa_mcp("web_fetch_exa", {"urls": [u]})
        if not content:
            return "NO RESULTS"
        return content[:12000]

    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
