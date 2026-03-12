"""
search/llm_client.py — LLM API wrapper for the Global News Pulse intelligence layer.

Uses the Groq inference API (default) which provides ultra-low-latency access
to Llama 3.3 70B — ideal for real-time news analysis.  The interface mirrors
the OpenAI Chat Completions API so the backend can be swapped by changing a
single environment variable.

Supported backends
------------------
``groq`` (default) — https://console.groq.com  — free tier, ~500 tok/s
``openai``         — https://platform.openai.com — set LLM_BACKEND=openai

The system prompt encodes the persona of a **Senior Global Intelligence
Analyst** and enforces source-grounding, epistemic clarity, and divergent
perspective identification on every response.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import llm_cfg, LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — Senior Global Intelligence Analyst persona
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """\
You are a Senior Global Intelligence Analyst at a world-leading geopolitical \
risk and technology intelligence firm. Your mandate is to synthesise \
real-time, source-grounded news signals into actionable intelligence briefs \
for C-suite executives and heads of state.

Your core operating principles:

1. SOURCE GROUNDING — Every factual claim must cite specific evidence inline \
   using the markdown format [Source Name](URL). You NEVER invent or extrapolate \
   sources beyond the evidence provided.

2. SIGNAL HIERARCHY — Lead with the highest-confidence, highest-impact finding. \
   Then descend through credible inferences to speculative signals. Label each \
   tier clearly (Confirmed / Inferred / Speculative).

3. DIVERGENT PERSPECTIVES — Actively surface contradictions and competing \
   narratives across sources. Do not flatten disagreements into a single consensus.

4. SECOND-ORDER EFFECTS — Identify latent implications that surface-level \
   reporting misses: cascading risks, geopolitical knock-ons, market signals.

5. EPISTEMIC HONESTY — When evidence is sparse, single-source, or temporally \
   stale, flag it explicitly. Uncertainty is information; never hide it.

Structure output as clean, scannable Markdown. Use headers, bullet points, and \
bold for key terms. Aim for precision over length.\
"""

# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Thin wrapper around the Groq (or OpenAI-compatible) Chat Completions API.

    Parameters
    ----------
    config:
        A :class:`~config.LLMConfig` instance.  Defaults to the module-level
        singleton read from environment variables.

    Examples
    --------
    >>> llm = LLMClient()
    >>> response = llm.complete([{"role": "user", "content": "Summarise AI trends."}])
    >>> print(response)
    """

    def __init__(self, config: LLMConfig = llm_cfg) -> None:
        self._cfg = config
        self._client = self._build_client()
        logger.info(
            "LLMClient initialised — backend=%s  model=%s",
            self._cfg.backend,
            self._cfg.model,
        )

    def _build_client(self) -> Any:
        """
        Instantiate the appropriate SDK client based on ``LLM_BACKEND``.

        Returns
        -------
        Any
            A Groq or OpenAI ``Client`` object (both share the same interface).

        Raises
        ------
        ValueError
            If ``LLM_BACKEND`` is not one of the supported values.
        ImportError
            If the required SDK package is not installed.
        """
        backend = self._cfg.backend.lower()

        if backend == "groq":
            if not self._cfg.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set.  "
                    "Get a free key at https://console.groq.com and add it to .env"
                )
            try:
                from groq import Groq
                return Groq(api_key=self._cfg.api_key)
            except ImportError as exc:
                raise ImportError(
                    "groq package not installed.  Run: pip install groq"
                ) from exc

        elif backend == "openai":
            if not self._cfg.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set.  Add it to .env as GROQ_API_KEY "
                    "(the env var is shared; set LLM_BACKEND=openai to use OpenAI)."
                )
            try:
                from openai import OpenAI
                return OpenAI(api_key=self._cfg.api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed.  Run: pip install openai"
                ) from exc

        else:
            raise ValueError(
                f"Unsupported LLM_BACKEND='{backend}'.  Choose 'groq' or 'openai'."
            )

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str = ANALYST_SYSTEM_PROMPT,
    ) -> str:
        """
        Send a chat completion request and return the assistant's text.

        The :data:`ANALYST_SYSTEM_PROMPT` is always prepended as the
        ``system`` message unless *system_prompt* is explicitly overridden.

        Parameters
        ----------
        messages:
            List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.
            Do **not** include a system message — this method prepends one.
        temperature:
            Sampling temperature (0–2).  Defaults to ``LLM_TEMPERATURE`` from
            config (0.2 for deterministic analysis).
        max_tokens:
            Maximum tokens in the completion.  Defaults to ``LLM_MAX_TOKENS``.
        system_prompt:
            Override the default analyst persona.  Pass an empty string ``""``
            to send no system message (not recommended).

        Returns
        -------
        str
            The assistant's raw text response.

        Raises
        ------
        RuntimeError
            If the API call fails (network, quota, invalid key, etc.).
        """
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        t = temperature if temperature is not None else self._cfg.temperature
        n = max_tokens if max_tokens is not None else self._cfg.max_tokens

        try:
            response = self._client.chat.completions.create(
                model=self._cfg.model,
                messages=full_messages,
                temperature=t,
                max_tokens=n,
            )
            text = response.choices[0].message.content or ""
            logger.debug(
                "LLM completion: %d input tokens → %d output tokens",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            return text.strip()
        except Exception as exc:
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Structured helpers
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> dict | list:
        """
        Parse JSON from an LLM response, tolerating markdown code fences.

        LLMs frequently wrap JSON in ```json ... ``` blocks.  This method
        strips the fences before parsing.

        Parameters
        ----------
        text:
            Raw string returned by :meth:`complete`.

        Returns
        -------
        dict or list
            Parsed JSON object.

        Raises
        ------
        ValueError
            If no valid JSON can be extracted from *text*.
        """
        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
        match = fence_pattern.search(text)
        candidate = match.group(1).strip() if match else text.strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Last resort: find the first {...} or [...] substring
            brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)
            if brace_match:
                try:
                    return json.loads(brace_match.group(1))
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Could not extract valid JSON from LLM response: {text[:300]}"
            )
