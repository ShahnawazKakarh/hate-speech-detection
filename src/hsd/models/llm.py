"""Gemini-backed LLM classifier with disk caching and quota-resilient calls.

Wraps a Gemini model (default ``gemini-2.5-flash``) behind the same
``fit`` / ``predict`` / ``predict_proba`` interface as the other ``hsd`` models.

``fit`` is a no-op for zero-shot; for few-shot it deterministically samples K
labelled examples from the training set and bakes them into the prompt.

Inference asks the model to emit JSON ``{"label": "hate"|"non-hate", "p_hate": float}``
so we get both a hard label and a probability-like score for ROC-AUC.

**Quota resilience:**
- Every successful response is persisted to ``data/llm_cache/<run_name>.jsonl``
  immediately. Re-runs resume from the next un-cached row at zero cost.
- 429 / ``RESOURCE_EXHAUSTED`` errors trigger a controlled abort with a clear
  message, *after* the cache is flushed, so no work is lost.
- Inter-call pacing via ``request_interval_s`` keeps us under the RPM ceiling.

Env:
    GEMINI_API_KEY   required (or set in .env)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hsd.utils import REPO_ROOT, get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """Config for the Gemini LLM classifier."""

    model_name: str = "gemini-2.5-flash"
    regime: str = "zero_shot"  # zero_shot | few_shot
    k_shots: int = 4
    seed: int = 42
    max_output_tokens: int = 64
    temperature: float = 0.0
    request_timeout_s: float = 30.0
    retry_attempts: int = 4
    retry_backoff_s: float = 4.0
    # Pacing — Gemini free tier is RPM-limited. Default 5 = 12 req/min.
    request_interval_s: float = 5.0
    cache_dir: str = "data/llm_cache"
    cache_name: str = "gemini"


# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a content-moderation classifier. Given a social-media comment, decide whether it is HATE SPEECH or not.

Definition of HATE SPEECH (label = "hate"):
- Attacks or demeans a group based on protected attributes such as race, ethnicity, national origin, religion, gender, sexual orientation, or disability.
- The attack must target a *group* or someone *as a member of that group*, not merely insult an individual.

NOT hate speech (label = "non-hate"):
- Profanity, vulgarity, or insults directed at individuals without targeting a protected group.
- Offensive language, criticism, sarcasm, or rude comments that do not invoke protected-group attributes.
- Reclaimed in-group slurs used by members of the group themselves.

Output strict JSON only, no other text:
{"label": "hate" | "non-hate", "p_hate": <float between 0.0 and 1.0>}

`p_hate` is your calibrated probability that the comment is hate speech. Be honest about uncertainty: borderline / ambiguous cases should get probabilities near 0.5, not extremes.
"""


def _build_user_prompt(text: str, examples: list[tuple[str, int]] | None = None) -> str:
    parts = []
    if examples:
        parts.append("Here are some labelled examples for reference:\n")
        for i, (ex_text, ex_label) in enumerate(examples, 1):
            tag = "hate" if ex_label == 1 else "non-hate"
            parts.append(f'Example {i}: "{ex_text}"\nLabel: {tag}\n')
        parts.append("\nNow classify the following comment.\n")
    parts.append(f'Comment: "{text}"\n\nJSON:')
    return "".join(parts)


# --------------------------------------------------------------------------- #
class QuotaExhausted(Exception):
    """Raised when the API returns a 429 / quota error so we can stop cleanly."""


# --------------------------------------------------------------------------- #
class _DiskCache:
    """Append-only JSONL cache keyed by sha1(prompt). Crash-safe."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        self._mem[row["key"]] = row["response"]
                    except Exception:  # noqa: BLE001
                        continue
            log.info("cache: loaded %d entries from %s", len(self._mem), self.path)

    @staticmethod
    def key_for(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return self._mem.get(key)

    def put(self, key: str, response: dict) -> None:
        self._mem[key] = response
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "response": response}) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


# --------------------------------------------------------------------------- #
def _parse_response_text(text: str) -> tuple[int, float]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        log.warning("failed to parse JSON, treating as non-hate: %r", text[:120])
        return 0, 0.0
    label = str(obj.get("label", "")).lower().strip()
    p_hate = float(obj.get("p_hate", 0.0))
    p_hate = max(0.0, min(1.0, p_hate))
    label_int = (
        1 if (label == "hate" or (label not in {"hate", "non-hate"} and p_hate >= 0.5)) else 0
    )
    return label_int, p_hate


def _is_quota_error(exc: Exception) -> bool:
    """Detect Gemini quota / rate-limit errors so we abort cleanly."""
    s = str(exc).lower()
    return any(k in s for k in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit"))


# --------------------------------------------------------------------------- #
class GeminiClassifier:
    """LLM-as-classifier with on-disk caching and quota-aware retries."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._examples: list[tuple[str, int]] = []
        self._client = None
        self._cache = _DiskCache(REPO_ROOT / cfg.cache_dir / f"{cfg.cache_name}.jsonl")

    # ------------------------------------------------------------------ #
    def _get_client(self):
        if self._client is None:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. `export GEMINI_API_KEY=...` or put it in .env"
                )
            self._client = genai.Client(api_key=key)
        return self._client

    # ------------------------------------------------------------------ #
    def fit(self, texts: list[str], labels: list[int]) -> GeminiClassifier:
        if self.cfg.regime == "few_shot":
            rng = random.Random(self.cfg.seed)
            pos = [i for i, y in enumerate(labels) if y == 1]
            neg = [i for i, y in enumerate(labels) if y == 0]
            k_each = max(1, self.cfg.k_shots // 2)
            picks = rng.sample(pos, min(k_each, len(pos))) + rng.sample(
                neg, min(self.cfg.k_shots - k_each, len(neg))
            )
            rng.shuffle(picks)
            self._examples = [(texts[i], labels[i]) for i in picks]
            log.info("few-shot examples: %d", len(self._examples))
        else:
            self._examples = []
        return self

    # ------------------------------------------------------------------ #
    def _call(self, prompt: str) -> dict:
        key = _DiskCache.key_for(prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        client = self._get_client()
        from google.genai import types

        gen_config = types.GenerateContentConfig(
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_output_tokens,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["hate", "non-hate"]},
                    "p_hate": {"type": "number"},
                },
                "required": ["label", "p_hate"],
            },
            # Disable Gemini 2.5 Flash's thinking budget; we want all output tokens
            # spent on the JSON, not on internal reasoning that gets dropped.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        last_err: Exception | None = None
        for attempt in range(self.cfg.retry_attempts):
            try:
                response = client.models.generate_content(
                    model=self.cfg.model_name,
                    contents=prompt,
                    config=gen_config,
                )
                text = (response.text or "").strip()
                label_int, p_hate = _parse_response_text(text)
                out = {"label": label_int, "p_hate": p_hate, "raw": text}
                self._cache.put(key, out)
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_quota_error(e):
                    log.error("quota / rate limit hit on attempt %d: %s", attempt + 1, e)
                    raise QuotaExhausted(str(e)) from e
                sleep = self.cfg.retry_backoff_s * (2**attempt)
                log.warning(
                    "gemini call failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self.cfg.retry_attempts,
                    e,
                    sleep,
                )
                time.sleep(sleep)

        log.error("gemini call failed permanently: %s", last_err)
        out = {"label": 0, "p_hate": 0.0, "raw": f"<error: {last_err}>", "errored": True}
        self._cache.put(key, out)
        return out

    # ------------------------------------------------------------------ #
    def _classify_one(self, text: str) -> tuple[int, float]:
        examples = self._examples if self.cfg.regime == "few_shot" else None
        user_prompt = _build_user_prompt(text, examples)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        resp = self._call(full_prompt)
        return int(resp["label"]), float(resp["p_hate"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        scores = np.zeros(len(texts), dtype=float)
        cached_hits = 0
        new_calls = 0

        # Pre-build prompts once so we can count cache hits before any API call
        prompts = []
        for t in texts:
            examples = self._examples if self.cfg.regime == "few_shot" else None
            up = _build_user_prompt(t, examples)
            prompts.append(f"{SYSTEM_PROMPT}\n\n{up}")
        keys = [_DiskCache.key_for(p) for p in prompts]
        n_cached = sum(1 for k in keys if self._cache.get(k) is not None)
        log.info(
            "predict: %d total, %d cached (%.1f%%), %d new calls needed",
            len(texts),
            n_cached,
            100.0 * n_cached / max(1, len(texts)),
            len(texts) - n_cached,
        )

        pbar = tqdm(texts, desc=self.cfg.cache_name, ncols=90)
        for i, _ in enumerate(pbar):
            try:
                if self._cache.get(keys[i]) is not None:
                    cached_hits += 1
                    resp = self._cache.get(keys[i])
                    scores[i] = float(resp["p_hate"])
                else:
                    # Pace only fresh calls, not cache hits
                    if new_calls > 0 and self.cfg.request_interval_s > 0:
                        time.sleep(self.cfg.request_interval_s)
                    resp = self._call(prompts[i])
                    scores[i] = float(resp["p_hate"])
                    new_calls += 1
            except QuotaExhausted as e:
                pbar.close()
                done = i
                log.error(
                    "\n\nQUOTA EXHAUSTED at row %d / %d (%.1f%% done).\n"
                    "Progress saved to cache: %s\n"
                    "Re-run the same config to resume from row %d.\n"
                    "Underlying error: %s\n",
                    done,
                    len(texts),
                    100.0 * done / len(texts),
                    self._cache.path,
                    done,
                    e,
                )
                # Exit non-zero so shell scripts can detect it
                sys.exit(2)

        log.info("done: %d cache hits, %d new calls", cached_hits, new_calls)
        return scores

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "cfg": self.cfg.__dict__,
            "examples": self._examples,
        }
        (path / "llm_state.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GeminiClassifier:
        path = Path(path)
        meta = json.loads((path / "llm_state.json").read_text())
        cfg = LLMConfig(**meta["cfg"])
        obj = cls(cfg)
        obj._examples = [tuple(e) for e in meta["examples"]]
        return obj
