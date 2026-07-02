"""LLM-as-classifier via OpenRouter (OpenAI-compatible API).

Wraps any OpenRouter-hosted model behind the ``fit`` / ``predict`` /
``predict_proba`` interface used by the other ``hsd`` models. Switch
models by editing ``model_name`` in the YAML config.

Design goals, prioritising no-wastage:

  1. **Per-call disk cache.** Every successful response is persisted
     to ``data/llm_cache/<cache_name>.jsonl`` immediately, keyed by
     ``sha1(model_name || prompt)``. Re-running the same config is
     free. Errored entries are stored but treated as cache-misses on
     the next load so they retry naturally.

  2. **Pre-flight cost estimate.** Before any API call, the number of
     cache-miss calls is multiplied by an average-token estimate and
     the model's price. If the estimate exceeds
     ``confirm_over_usd`` (default $0.20), the script aborts with a
     clear instruction unless ``HSD_CONFIRM_COST=1`` is set. If it
     exceeds ``cost_ceiling_usd`` (default $1.00), the script always
     aborts.

  3. **Live spend counter.** Every call records actual usage from
     the API response. Running total is printed on the progress bar.
     If total exceeds ``cost_ceiling_usd`` mid-run, we hard-stop.

Env:
    OPENROUTER_API_KEY   required, from https://openrouter.ai/keys
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
# Pricing table (USD per 1M tokens). Adjust as OpenRouter prices change.
# Fetch live prices from https://openrouter.ai/api/v1/models if precision
# matters more than deterministic pre-flight estimates.
OPENROUTER_PRICING: dict[str, dict[str, float]] = {
    "google/gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "google/gemini-2.5-flash-lite": {"input": 0.025, "output": 0.10},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "anthropic/claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.13, "output": 0.40},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.03, "output": 0.06},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.35, "output": 0.40},
}


# --------------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    """Config for the OpenRouter-backed LLM classifier."""

    model_name: str = "google/gemini-2.5-flash"
    regime: str = "zero_shot"  # zero_shot | few_shot
    k_shots: int = 4
    seed: int = 42
    max_output_tokens: int = 96
    temperature: float = 0.0
    request_timeout_s: float = 60.0
    retry_attempts: int = 4
    retry_backoff_s: float = 4.0
    # Pacing (OpenRouter passes through to upstream; free-tier models are RPM-limited)
    request_interval_s: float = 0.0
    # Cache
    cache_dir: str = "data/llm_cache"
    cache_name: str = "llm"
    # Cost safety
    cost_ceiling_usd: float = 1.0  # hard stop, always enforced
    confirm_over_usd: float = 0.20  # require HSD_CONFIRM_COST=1 above this
    avg_output_tokens_estimate: int = 50  # for pre-flight, before we know actual


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

`p_hate` is your calibrated probability that the comment is hate speech. Borderline / ambiguous cases should get probabilities near 0.5, not extremes.
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
    """Raised on 429 / RESOURCE_EXHAUSTED so callers can abort cleanly."""


class CostCeilingHit(Exception):
    """Raised when running spend crosses ``cost_ceiling_usd``."""


# --------------------------------------------------------------------------- #
class _DiskCache:
    """Append-only JSONL cache. Errored entries are stored (for debugging)
    but ``get()`` returns None for them so they retry on the next run."""

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
            n_ok = sum(1 for r in self._mem.values() if not r.get("errored"))
            n_err = len(self._mem) - n_ok
            log.info("cache: %d ok + %d errored from %s", n_ok, n_err, self.path)

    @staticmethod
    def key_for(model_name: str, prompt: str) -> str:
        return hashlib.sha1(f"{model_name}\n{prompt}".encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        entry = self._mem.get(key)
        if entry is None or entry.get("errored"):
            return None
        return entry

    def put(self, key: str, response: dict) -> None:
        self._mem[key] = response
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "response": response}) + "\n")


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
    s = str(exc).lower()
    return any(k in s for k in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit"))


# --------------------------------------------------------------------------- #
class LLMClassifier:
    """OpenRouter-backed classifier with disk caching + cost safeguards."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._examples: list[tuple[str, int]] = []
        self._client = None
        self._cache = _DiskCache(REPO_ROOT / cfg.cache_dir / f"{cfg.cache_name}.jsonl")
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_spent_usd = 0.0

    # ------------------------------------------------------------------ #
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set. Put it in .env or export it.")
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                timeout=self.cfg.request_timeout_s,
                default_headers={
                    "HTTP-Referer": "https://github.com/ShahnawazKakarh/hate-speech-detection",
                    "X-Title": "hsd v0.2.1 LLM baselines",
                },
            )
        return self._client

    # ------------------------------------------------------------------ #
    def fit(self, texts: list[str], labels: list[int]) -> LLMClassifier:
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
            log.info(
                "few-shot exemplars: %d (%d hate, %d non-hate)",
                len(self._examples),
                sum(1 for _, y in self._examples if y == 1),
                sum(1 for _, y in self._examples if y == 0),
            )
        else:
            self._examples = []
        return self

    # ------------------------------------------------------------------ #
    def _cost_for_call(self, input_tokens: int, output_tokens: int) -> float:
        p = OPENROUTER_PRICING.get(self.cfg.model_name)
        if p is None:
            return 0.0
        return input_tokens / 1e6 * p["input"] + output_tokens / 1e6 * p["output"]

    def _estimate_cost(self, prompts: list[str]) -> tuple[float, float, float]:
        """Return (est_usd, est_input_tokens, est_output_tokens)."""
        chars = sum(len(p) for p in prompts)
        input_tokens = chars / 4.0  # rough industry-standard heuristic
        output_tokens = len(prompts) * self.cfg.avg_output_tokens_estimate
        return (
            self._cost_for_call(int(input_tokens), int(output_tokens)),
            input_tokens,
            output_tokens,
        )

    # ------------------------------------------------------------------ #
    def _call(self, prompt: str) -> dict:
        key = _DiskCache.key_for(self.cfg.model_name, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        client = self._get_client()
        last_err: Exception | None = None
        for attempt in range(self.cfg.retry_attempts):
            try:
                response = client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_output_tokens,
                    response_format={"type": "json_object"},
                )
                text = (response.choices[0].message.content or "").strip()
                label_int, p_hate = _parse_response_text(text)

                usage = response.usage
                in_tok = int(usage.prompt_tokens) if usage else 0
                out_tok = int(usage.completion_tokens) if usage else 0
                call_cost = self._cost_for_call(in_tok, out_tok)

                self._total_input_tokens += in_tok
                self._total_output_tokens += out_tok
                self._total_spent_usd += call_cost

                out = {
                    "label": label_int,
                    "p_hate": p_hate,
                    "raw": text,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": call_cost,
                }
                self._cache.put(key, out)

                # Hard cost ceiling — abort if the running total is over budget
                if self._total_spent_usd > self.cfg.cost_ceiling_usd:
                    raise CostCeilingHit(
                        f"Running spend ${self._total_spent_usd:.4f} > ceiling "
                        f"${self.cfg.cost_ceiling_usd:.4f}"
                    )
                return out
            except CostCeilingHit:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_quota_error(e):
                    log.error("quota / rate limit hit on attempt %d: %s", attempt + 1, e)
                    raise QuotaExhausted(str(e)) from e
                sleep = self.cfg.retry_backoff_s * (2**attempt)
                log.warning(
                    "openrouter call failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    self.cfg.retry_attempts,
                    e,
                    sleep,
                )
                time.sleep(sleep)

        # Retries exhausted — persist as errored so it retries on next run
        log.error("call failed permanently: %s", last_err)
        err_entry = {"errored": True, "error": str(last_err)}
        self._cache.put(key, err_entry)
        # Return a conservative response for this row
        return {"label": 0, "p_hate": 0.0, "raw": f"<error: {last_err}>", "errored": True}

    # ------------------------------------------------------------------ #
    def _preflight(self, prompts: list[str]) -> None:
        """Count cache misses, estimate cost, guardrail with env var confirmation."""
        misses = [
            p
            for p in prompts
            if self._cache.get(_DiskCache.key_for(self.cfg.model_name, p)) is None
        ]
        n_cached = len(prompts) - len(misses)
        est_usd, _, _ = self._estimate_cost(misses)

        log.info(
            "pre-flight: %d total | %d cached (free) | %d new calls | model=%s",
            len(prompts),
            n_cached,
            len(misses),
            self.cfg.model_name,
        )
        log.info(
            "pre-flight cost estimate: $%.4f (ceiling $%.2f, confirm-over $%.2f)",
            est_usd,
            self.cfg.cost_ceiling_usd,
            self.cfg.confirm_over_usd,
        )

        if est_usd > self.cfg.cost_ceiling_usd:
            log.error(
                "pre-flight estimate $%.4f exceeds cost_ceiling_usd $%.4f. "
                "Raise the ceiling in the config only if you really want to spend that much.",
                est_usd,
                self.cfg.cost_ceiling_usd,
            )
            sys.exit(1)

        if est_usd > self.cfg.confirm_over_usd and os.environ.get("HSD_CONFIRM_COST") != "1":
            log.error(
                "pre-flight estimate $%.4f exceeds confirm-over threshold $%.4f. "
                "Set HSD_CONFIRM_COST=1 to proceed:\n"
                "    HSD_CONFIRM_COST=1 python -m hsd.train --config <config>",
                est_usd,
                self.cfg.confirm_over_usd,
            )
            sys.exit(1)

    # ------------------------------------------------------------------ #
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        # Build all prompts once
        prompts = []
        for t in texts:
            examples = self._examples if self.cfg.regime == "few_shot" else None
            prompts.append(_build_user_prompt(t, examples))

        self._preflight(prompts)

        scores = np.zeros(len(texts), dtype=float)
        n_cached = 0
        n_new = 0

        pbar = tqdm(
            range(len(texts)),
            desc=self.cfg.cache_name,
            ncols=100,
            postfix={"$": "0.0000"},
        )
        for i in pbar:
            key = _DiskCache.key_for(self.cfg.model_name, prompts[i])
            try:
                cached = self._cache.get(key)
                if cached is not None:
                    n_cached += 1
                    scores[i] = float(cached["p_hate"])
                else:
                    if n_new > 0 and self.cfg.request_interval_s > 0:
                        time.sleep(self.cfg.request_interval_s)
                    resp = self._call(prompts[i])
                    scores[i] = float(resp["p_hate"])
                    n_new += 1
                    pbar.set_postfix({"$": f"{self._total_spent_usd:.4f}"})
            except QuotaExhausted as e:
                pbar.close()
                log.error(
                    "\nQUOTA EXHAUSTED at row %d / %d (%.1f%% done). "
                    "Progress cached at %s. Re-run to resume.\n"
                    "Underlying error: %s",
                    i,
                    len(texts),
                    100.0 * i / len(texts),
                    self._cache.path,
                    e,
                )
                sys.exit(2)
            except CostCeilingHit as e:
                pbar.close()
                log.error(
                    "\nCOST CEILING HIT at row %d / %d ($%.4f). "
                    "Progress cached at %s. Raise cost_ceiling_usd and re-run "
                    "if you want to continue.\n%s",
                    i,
                    len(texts),
                    self._total_spent_usd,
                    self._cache.path,
                    e,
                )
                sys.exit(3)

        log.info(
            "predict_proba done: %d cached + %d new calls | spent $%.4f | " "tokens: in=%d out=%d",
            n_cached,
            n_new,
            self._total_spent_usd,
            self._total_input_tokens,
            self._total_output_tokens,
        )
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
            "spend_usd": self._total_spent_usd,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }
        (path / "llm_state.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LLMClassifier:
        path = Path(path)
        meta = json.loads((path / "llm_state.json").read_text())
        cfg = LLMConfig(**meta["cfg"])
        obj = cls(cfg)
        obj._examples = [tuple(e) for e in meta["examples"]]
        return obj


# Backward-compat alias for older imports
GeminiClassifier = LLMClassifier
