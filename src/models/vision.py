"""VLM captions for figure chunks (PNG bytes → summary text)."""

from __future__ import annotations

import base64

from openai import OpenAI

from src import config


_VLM_PROMPT = (
    "You are indexing technical regulatory PDFs (e.g. automotive / ADAS). "
    "Describe this figure or diagram in plain English for semantic search: "
    "layout, axes/labels if any, shapes, arrows, vehicles, and any readable text. "
    "Be factual; max ~200 words."
)


def summarize_figure_png(client: OpenAI, png_bytes: bytes) -> str:
    if not png_bytes:
        return "[No image bytes available]"
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    url = f"data:image/png;base64,{b64}"
    r = client.chat.completions.create(
        model=config.get_vision_model(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        max_tokens=500,
    )
    msg = r.choices[0].message
    return (msg.content or "").strip() or "[Empty VLM response]"
