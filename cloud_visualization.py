"""Dependency-free SVG rendering for token clouds."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from model_fingerprint import ModelFingerprint, top_token_cloud
from token_analysis import TokenAnalyzer


CloudEntry = Mapping[str, object]


def _scale_font(count: int, minimum: int, maximum: int, highest_count: int) -> int:
    if highest_count <= 0:
        return minimum
    if highest_count == 1:
        return maximum
    return round(minimum + (maximum - minimum) * (count / highest_count) ** 0.65)


def _entry_text(entry: CloudEntry) -> str:
    if "text" in entry:
        text = str(entry["text"])
    else:
        text = f"#{entry['token_id']}"
    return text.replace("\n", "\\n").replace("\r", "\\r")


def render_token_cloud_svg(
    entries: Sequence[CloudEntry],
    width: int = 960,
    height: int = 540,
    title: str | None = None,
    min_font_size: int = 14,
    max_font_size: int = 64,
) -> str:
    """Render token-cloud entries as a standalone SVG string.

    The layout is deterministic and intentionally simple: high-frequency tokens
    are larger and placed first in a flowing cloud. This avoids extra plotting
    dependencies while producing a portable artifact that can be opened in any
    browser.
    """
    if width < 320:
        raise ValueError("width must be at least 320")
    if height < 240:
        raise ValueError("height must be at least 240")
    if min_font_size < 1 or max_font_size < min_font_size:
        raise ValueError("font sizes must be positive and ordered")

    palette = [
        "#2563eb",
        "#0f766e",
        "#be123c",
        "#7c3aed",
        "#b45309",
        "#0369a1",
        "#15803d",
        "#a21caf",
    ]
    background = "#f8fafc"
    text_elements: list[str] = []

    sorted_entries = sorted(
        entries,
        key=lambda entry: int(entry["count"]),
        reverse=True,
    )
    highest_count = int(sorted_entries[0]["count"]) if sorted_entries else 0

    x = 40
    y = 82 if title else 50
    row_height = 0
    max_x = width - 40

    for index, entry in enumerate(sorted_entries):
        count = int(entry["count"])
        text = _entry_text(entry).strip() or "<space>"
        font_size = _scale_font(count, min_font_size, max_font_size, highest_count)
        approx_width = max(24, int(len(text) * font_size * 0.58))

        if x + approx_width > max_x:
            x = 40 + (index * 17) % 90
            y += max(row_height, max_font_size) + 18
            row_height = 0

        if y > height - 35:
            break

        color = palette[index % len(palette)]
        text_elements.append(
            (
                f'<text x="{x}" y="{y}" font-size="{font_size}" '
                f'font-weight="{700 if index < 5 else 500}" fill="{color}">'
                f"{escape(text)}</text>"
            )
        )
        x += approx_width + 22
        row_height = max(row_height, font_size)

    title_element = ""
    if title:
        title_element = (
            f'<text x="40" y="38" font-size="24" font-weight="700" '
            f'fill="#111827">{escape(title)}</text>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="100%" height="100%" rx="0" fill="{background}"/>',
            title_element,
            *text_elements,
            "</svg>",
        ]
    )


def write_token_cloud_svg(
    entries: Sequence[CloudEntry],
    path: str | Path,
    width: int = 960,
    height: int = 540,
    title: str | None = None,
) -> Path:
    """Write token-cloud entries to an SVG file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_token_cloud_svg(entries, width=width, height=height, title=title),
        encoding="utf-8",
    )
    return output_path


def fingerprint_to_svg(
    fingerprint: ModelFingerprint,
    path: str | Path,
    analyzer: TokenAnalyzer | None = None,
    limit: int = 50,
    width: int = 960,
    height: int = 540,
) -> Path:
    """Render a model fingerprint's most frequent tokens to SVG."""
    entries = top_token_cloud(fingerprint, analyzer=analyzer, limit=limit)
    return write_token_cloud_svg(
        entries,
        path,
        width=width,
        height=height,
        title=f"{fingerprint.model} token cloud",
    )


def text_to_svg(
    text: str,
    path: str | Path,
    analyzer: TokenAnalyzer | None = None,
    title: str = "Token cloud",
    limit: int = 50,
) -> Path:
    """Build and render a token cloud directly from text."""
    active_analyzer = analyzer or TokenAnalyzer()
    counts = active_analyzer.encode(text)
    from collections import Counter

    fingerprint = ModelFingerprint(
        model=title,
        token_counts=Counter(counts),
        responses=(text,),
        prompt_count=1,
        sample_count=1,
    )
    return fingerprint_to_svg(
        fingerprint,
        path,
        analyzer=active_analyzer,
        limit=limit,
    )
