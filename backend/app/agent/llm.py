"""
ScholarMind — Gemini LLM Client

Shared wrapper for Google Gemini API calls.
Used by all agentic nodes for reasoning, grading, and generation.
"""

import json

import google.generativeai as genai
from app.config import get_settings

_model = None


def _get_model():
    """Lazy-load the Gemini generative model."""
    global _model
    if _model is None:
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        _model = genai.GenerativeModel(settings.gemini_model)
    return _model


def llm_call(prompt: str, temperature: float = 0.3) -> str:
    """
    Make a text generation call to Gemini.

    Args:
        prompt: The full prompt string.
        temperature: Creativity level (0=deterministic, 1=creative).

    Returns:
        The model's text response.
    """
    model = _get_model()
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )
    return response.text


def llm_json_call(prompt: str, temperature: float = 0.1) -> dict | list:
    """
    Make a JSON-mode call to Gemini.

    Returns parsed JSON (dict or list).
    """
    model = _get_model()
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)
