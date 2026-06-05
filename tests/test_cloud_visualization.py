import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cloud_visualization import render_token_cloud_svg, text_to_svg, write_token_cloud_svg


def test_render_token_cloud_svg_outputs_standalone_svg() -> None:
    svg = render_token_cloud_svg(
        [
            {"token_id": 1, "text": "hello", "count": 5},
            {"token_id": 2, "text": "<world>", "count": 2},
        ],
        title="Demo",
    )

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Demo" in svg
    assert "hello" in svg
    assert "&lt;world&gt;" in svg
    assert svg.endswith("</svg>")


def test_write_token_cloud_svg_creates_parent_directory() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "cloud.svg"

        result = write_token_cloud_svg(
            [{"token_id": 1, "text": "token", "count": 3}],
            path,
        )

        assert result == path
        assert path.exists()
        assert "token" in path.read_text(encoding="utf-8")


def test_text_to_svg_generates_cloud_from_text() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "text-cloud.svg"

        result = text_to_svg("alpha beta alpha", path, title="Example")

        assert result == path
        svg = path.read_text(encoding="utf-8")
        assert "Example token cloud" in svg
        assert "alpha" in svg
