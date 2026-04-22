"""
config/llm_client.py
Central Ollama LLM wrapper for the Healthcare MAS pipeline.

All agents call get_llm_commentary() after their Python tool completes.
If Ollama is not running or the model is unavailable, the function returns
an empty string and logs a warning — it never raises.  This ensures all
existing tests continue to pass without any Ollama server present.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def get_llm_commentary(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
) -> str:
    """
    Call the local Ollama LLM and return a clinical reasoning string.

    If Ollama is unavailable (server not running, model not pulled, timeout,
    etc.) the function logs a WARNING and returns "" so the pipeline continues
    unaffected.

    Args:
        agent_name:    Name of the calling agent (used in log messages).
        system_prompt: The agent-specific clinical persona / instructions.
        user_message:  The tool result context for the LLM to reason about.
        model:         Override the global OLLAMA_MODEL env var if needed.

    Returns:
        A non-empty clinical commentary string, or "" on any failure.
    """
    try:
        import ollama
        response = ollama.chat(
            model=model or OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return (response.message.content or "").strip()
    except Exception as exc:
        logger.warning(
            "[%s] Ollama unavailable (%s). Skipping LLM reasoning.",
            agent_name, exc,
        )
        return ""
