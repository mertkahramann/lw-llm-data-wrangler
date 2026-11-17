from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

Messages = list

RAW_ROLE = "raw"
RAW_STOP = ["\n"]


def raw_prompt(messages: Messages) -> str | None:
    if len(messages) == 1 and messages[0]["role"] == RAW_ROLE:
        return messages[0]["content"]
    if any(m["role"] == RAW_ROLE for m in messages):
        raise ValueError("a raw message must be the only message of a request")
    return None


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    seed: int = 1234
    cache_prompt: bool = False
    top_k: int | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str | None = None
    served_model: str | None = None
    simulated: bool = False
    raw: dict = field(default_factory=dict, repr=False)


class Backend(Protocol):
    def describe(self) -> dict: ...
    def count_prompt_tokens(self, messages: Messages) -> int: ...
    def complete(self, messages: Messages, max_tokens: int, settings: GenerationSettings) -> Completion: ...


class LlamaCppBackend:

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_s: float = 300.0,
                 model_sha256: str | None = None, quantization: str | None = "Q4_K_M",
                 model_name: str = "llama-3.1-8b-instruct", chat_template_kwargs: dict | None = None):
        import httpx
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_s, trust_env=False)
        self.model_sha256 = model_sha256
        self.quantization = quantization
        self.model_name = model_name
        self._props: dict | None = None
        self._tok_cache: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()
        self.chat_template_kwargs = chat_template_kwargs or None

    def health(self) -> dict:
        r = self._http.get("/health")
        return {"status_code": r.status_code, **(r.json() if r.headers.get("content-type", "").startswith("application/json") else {})}

    def props(self) -> dict:
        if self._props is None:
            r = self._http.get("/props")
            r.raise_for_status()
            self._props = r.json()
        return self._props

    def describe(self) -> dict:
        p = self.props()
        gen = p.get("default_generation_settings", {}) or {}
        tmpl = p.get("chat_template") or ""
        return {
            "backend": "llama.cpp",
            "base_url": self.base_url,
            "model_name": self.model_name,
            "model_path": p.get("model_path"),
            "model_sha256": self.model_sha256,
            "quantization": self.quantization,
            "runtime_build": p.get("build_info"),
            "n_ctx": gen.get("n_ctx") or p.get("n_ctx"),
            "total_slots": p.get("total_slots"),
            "chat_template_sha256": hashlib.sha256(tmpl.encode("utf-8")).hexdigest() if tmpl else None,
            "chat_template_kwargs": self.chat_template_kwargs,
        }

    def apply_template(self, messages: Messages) -> str:
        body = {"messages": messages}
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        r = self._http.post("/apply-template", json=body)
        r.raise_for_status()
        return r.json()["prompt"]

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = True) -> list[int]:
        r = self._http.post("/tokenize", json={"content": text, "add_special": add_special,
                                               "parse_special": parse_special})
        r.raise_for_status()
        toks = r.json()["tokens"]
        return [t["id"] if isinstance(t, dict) else int(t) for t in toks]

    def count_prompt_tokens(self, messages: Messages) -> int:
        key = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            if key in self._tok_cache:
                self._tok_cache.move_to_end(key)
                return self._tok_cache[key]
        raw = raw_prompt(messages)
        n = len(self.tokenize(raw if raw is not None else self.apply_template(messages), add_special=True))
        with self._lock:
            self._tok_cache[key] = n
            if len(self._tok_cache) > 50_000:
                self._tok_cache.popitem(last=False)
        return n

    def complete(self, messages: Messages, max_tokens: int, settings: GenerationSettings) -> Completion:
        raw = raw_prompt(messages)
        body = {"max_tokens": int(max_tokens), "stream": False, **settings.as_dict()}
        if raw is not None:
            body.update(prompt=raw, stop=RAW_STOP)
        else:
            body["messages"] = messages
            if self.chat_template_kwargs:
                body["chat_template_kwargs"] = self.chat_template_kwargs
        t0 = time.perf_counter()
        r = self._http.post("/v1/completions" if raw is not None else "/v1/chat/completions", json=body)
        latency = time.perf_counter() - t0
        r.raise_for_status()
        js = r.json()
        choice = js["choices"][0]
        usage = js.get("usage") or {}
        text = choice.get("text") if raw is not None else choice["message"].get("content")
        return Completion(text=text or "",
                          prompt_tokens=int(usage.get("prompt_tokens", 0)),
                          completion_tokens=int(usage.get("completion_tokens", 0)),
                          latency_s=latency, finish_reason=choice.get("finish_reason"),
                          served_model=js.get("model"), raw={"timings": js.get("timings"), "usage": usage})


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def heuristic_responder(messages: Messages) -> str:
    text = "\n".join(m["content"] for m in messages if m["role"] != "assistant")
    cols = []
    m = re.search(r"^Missing columns?: (.+)$", text, re.M)
    if m:
        cols = [re.sub(r"\s*\(.*\)$", "", c).strip() for c in m.group(1).split(",")]
    else:
        q = re.findall(r"What is the (.+?)\?\s*$", text, re.M)
        if q:
            cols = [q[-1].strip()]

    def guess(col: str) -> str:
        cand = re.search(rf"^Candidate values for {re.escape(col)}[^:]*: (.+)$", text, re.M)
        if cand:
            return cand.group(1).split(" | ")[0].strip()
        rng = re.search(rf"^Observed values of {re.escape(col)}: .*median ([^,\s]+)", text, re.M)
        if rng:
            return rng.group(1)
        ex = re.findall(rf"(?:^|\. ){re.escape(col)}: ([^.\n]+?)(?:\.|$)", text, re.M)
        if ex:
            vals = [v.strip() for v in ex if v.strip()]
            if vals:
                return max(sorted(set(vals)), key=vals.count)
        return "unknown"

    if "Missing columns:" in text:
        return "\n".join(f"{c}: {guess(c)}" for c in cols)
    return guess(cols[0]) if cols else "unknown"


class MockBackend:

    def __init__(self, responder: Callable[[Messages], str] | None = None, latency_s: float = 0.05,
                 chars_per_token: float = 4.0, fail_every: int = 0, name: str = "mock"):
        self.responder = responder or heuristic_responder
        self.latency_s = latency_s
        self.chars_per_token = chars_per_token
        self.fail_every = fail_every
        self.name = name
        self.calls = 0
        self._lock = threading.Lock()

    def describe(self) -> dict:
        return {"backend": "mock", "model_name": self.name, "model_sha256": "mock", "quantization": None,
                "runtime_build": "mock-1", "total_slots": None}

    def _count(self, text: str) -> int:
        return max(1, int(round(len(text) / self.chars_per_token)))

    def count_prompt_tokens(self, messages: Messages) -> int:
        return sum(self._count(m["content"]) + 4 for m in messages) + 1

    def complete(self, messages: Messages, max_tokens: int, settings: GenerationSettings) -> Completion:
        with self._lock:
            self.calls += 1
            k = self.calls
        if self.fail_every and k % self.fail_every == 0:
            raise ConnectionError(f"mock failure on call {k}")
        out = self.responder(messages)
        if raw_prompt(messages) is not None:
            out = out.split(RAW_STOP[0], 1)[0]
        ctoks = min(self._count(out), max_tokens)
        if self._count(out) > max_tokens:
            out = out[: int(max_tokens * self.chars_per_token)]
        return Completion(text=out, prompt_tokens=self.count_prompt_tokens(messages), completion_tokens=ctoks,
                          latency_s=self.latency_s, finish_reason="stop", served_model=self.name, simulated=True)


class Clock:

    def __init__(self, real: bool = True):
        self.real = real
        self.virtual_s = 0.0
        self._lock = threading.Lock()

    def now(self) -> float:
        base = time.perf_counter() if self.real else 0.0
        return base + self.virtual_s

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.virtual_s += max(0.0, float(seconds))


@dataclass
class Reservation:
    tokens: int
    time_s: float


class Budget:

    def __init__(self, token_cap: float | None, time_cap_s: float | None, clock: Clock,
                 p95_latency_s: float = 0.0, enforce_time: bool = True):
        self.token_cap = token_cap
        self.time_cap_s = time_cap_s
        self.clock = clock
        self.p95 = float(p95_latency_s)
        self.enforce_time = enforce_time
        self.start = clock.now()
        self.used_tokens = 0
        self.inflight_tokens = 0
        self.stopped = False
        self.stop_reason: str | None = None

    def elapsed(self) -> float:
        return self.clock.now() - self.start

    def try_reserve(self, tokens: int) -> Reservation | None:
        if self.stopped:
            return None
        if self.token_cap is not None and self.used_tokens + self.inflight_tokens + tokens > self.token_cap:
            self.stopped, self.stop_reason = True, "token_cap"
            return None
        if self.enforce_time and self.time_cap_s is not None:
            if self.elapsed() >= self.time_cap_s:
                self.stopped, self.stop_reason = True, "time_elapsed"
                return None
            if self.elapsed() + self.p95 > self.time_cap_s:
                self.stopped, self.stop_reason = True, "time_cap"
                return None
        self.inflight_tokens += tokens
        return Reservation(tokens=tokens, time_s=self.p95)

    def settle(self, res: Reservation, actual_tokens: int) -> None:
        self.inflight_tokens -= res.tokens
        self.used_tokens += int(actual_tokens)

    def report(self) -> dict:
        el = self.elapsed()
        return {"token_cap": self.token_cap, "time_cap_s": self.time_cap_s, "used_tokens": self.used_tokens,
                "elapsed_s": round(el, 4), "budget_stop": self.stopped, "stop_reason": self.stop_reason,
                "token_exceedance": (self.token_cap is not None and self.used_tokens > self.token_cap),
                "time_exceedance": (self.time_cap_s is not None and el > self.time_cap_s),
                "time_enforced": self.enforce_time}


def request_key(context: dict, messages: Messages, model: dict, settings: GenerationSettings,
                max_tokens: int) -> str:
    payload = {
        "context": context,
        "messages": messages,
        "model_sha256": model.get("model_sha256"),
        "quantization": model.get("quantization"),
        "runtime_build": model.get("runtime_build"),
        "model_name": model.get("model_name"),
        "chat_template_kwargs": model.get("chat_template_kwargs"),
        "settings": settings.as_dict(),
        "max_tokens": int(max_tokens),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RequestCache:

    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, text TEXT, "
                         "prompt_tokens INT, completion_tokens INT, latency_s REAL, created TEXT)")
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT text, prompt_tokens, completion_tokens, latency_s FROM answers "
                                   "WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        return {"text": row[0], "prompt_tokens": row[1], "completion_tokens": row[2], "latency_s": row[3]}

    def put(self, key: str, c: Completion) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO answers VALUES (?, ?, ?, ?, ?, ?)",
                             (key, c.text, c.prompt_tokens, c.completion_tokens, c.latency_s,
                              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]


class CallLog:

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        with self._lock:
            self.records.append(rec)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


@dataclass
class CallRequest:
    messages: Messages
    max_tokens: int
    context: dict
    meta: dict = field(default_factory=dict)
    prompt_tokens: int | None = None


@dataclass
class CallResult:
    request: CallRequest
    text: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cache_hit: bool
    attempts: int
    error: str | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:

    def __init__(self, backend: Backend, settings: GenerationSettings | None = None,
                 cache: RequestCache | None = None, log: CallLog | None = None, retries: int = 3,
                 n_slots: int = 1, replay_cache_latency: bool = False, read_cache: bool = True,
                 write_cache: bool = True, max_context: int | None = None):
        self.backend = backend
        self.settings = settings or GenerationSettings()
        self.cache = cache
        self.log = log or CallLog(None)
        self.retries = retries
        self.n_slots = max(1, int(n_slots))
        self.replay_cache_latency = replay_cache_latency
        self.read_cache = read_cache
        self.write_cache = write_cache
        self.max_context = max_context
        self.context_overflows = 0
        self.model = backend.describe()
        self._pool = ThreadPoolExecutor(max_workers=self.n_slots) if self.n_slots > 1 else None

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)

    def count(self, req: CallRequest) -> int:
        if req.prompt_tokens is None:
            req.prompt_tokens = self.backend.count_prompt_tokens(req.messages)
        return req.prompt_tokens

    def _one(self, req: CallRequest, clock: Clock) -> CallResult:
        key = request_key(req.context, req.messages, self.model, self.settings, req.max_tokens)
        if self.cache is not None and self.read_cache:
            hit = self.cache.get(key)
            if hit is not None:
                if self.replay_cache_latency:
                    clock.advance(hit["latency_s"])
                res = CallResult(req, hit["text"], hit["prompt_tokens"], hit["completion_tokens"],
                                 hit["latency_s"], True, 0)
                self._log(req, res, key)
                return res
        if self.max_context is not None and self.count(req) + req.max_tokens > self.max_context:
            self.context_overflows += 1
            res = CallResult(req, None, 0, 0, 0.0, False, 0, error="context_overflow")
            self._log(req, res, key)
            return res
        err = None
        for attempt in range(1, self.retries + 1):
            try:
                c = self.backend.complete(req.messages, req.max_tokens, self.settings)
                if c.simulated:
                    clock.advance(c.latency_s)
                if self.cache is not None and self.write_cache:
                    self.cache.put(key, c)
                res = CallResult(req, c.text, c.prompt_tokens, c.completion_tokens, c.latency_s, False, attempt)
                self._log(req, res, key, c)
                return res
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if type(e).__name__ == "LedgerRefusal":
                    break
        res = CallResult(req, None, 0, 0, 0.0, False, self.retries, error=err)
        self._log(req, res, key)
        return res

    def call_wave(self, reqs: list[CallRequest], clock: Clock) -> list[CallResult]:
        if not reqs:
            return []
        if self._pool is None or len(reqs) == 1:
            return [self._one(r, clock) for r in reqs]
        sub = [Clock(real=False) for _ in reqs]
        futs = [self._pool.submit(self._one, r, c) for r, c in zip(reqs, sub)]
        out = [f.result() for f in futs]
        clock.advance(max(c.virtual_s for c in sub))
        return out

    def _log(self, req: CallRequest, res: CallResult, key: str, c: Completion | None = None) -> None:
        self.log.write({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **req.meta, "context": req.context, "request_sha256": key,
            "counted_prompt_tokens": req.prompt_tokens, "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "latency_s": round(res.latency_s, 4),
            "cache_hit": res.cache_hit, "attempts": res.attempts, "error": res.error,
            "finish_reason": c.finish_reason if c else None, "served_model": c.served_model if c else None,
            "timings": (c.raw or {}).get("timings") if c else None,
            "response": res.text,
        })
