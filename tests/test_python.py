"""Ensure the CLI wrapper preserves the original pipeline and timing boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy", "matplotlib", "PIL")):
    print("SKIP: Python reference packages unavailable")
    sys.exit(77)

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from main import convert
from AsciiArt import AsciiArt
import numpy as np
from PIL import Image

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    source, expected, actual = (folder / name for name in ("input.png", "expected.png", "actual.png"))
    rng = np.random.default_rng(4242)
    Image.fromarray(rng.integers(0, 256, (35, 43, 3), dtype=np.uint8)).save(source)
    for bloom in (False, True):
        # Direct baseline calls, rather than relying on the new wrapper's sequence.
        edge = AsciiArt(str(source), "res/edgesASCII.png")
        edge.difference_of_gaussian(2.0, 1.6, 1.0, 0.3)
        edge.sobel()
        edge.get_magnitude(0.2)
        edge.find_angle()
        edge.edge_quantize()
        edge.controlled_downsample(8, 12)
        edge.to_ascii_art(True)
        fill = AsciiArt(str(source), "res/fillASCII.png")
        if bloom:
            fill.get_bloom_data(0.8, 50)
        fill.downsample(8)
        fill.quantize(10)
        fill.to_ascii_art(False)
        fill.combine(edge)
        if bloom:
            fill.add_bloom_data()
        fill.store_image(str(expected))
        metrics = convert(source, actual, bloom)
        assert np.array_equal(np.asarray(Image.open(expected)), np.asarray(Image.open(actual)))
        assert metrics["width"] == 40 and metrics["height"] == 32
        accounted = sum(metrics[name] for name in ("load_ms", "write_ms", "pipeline_ms", "overhead_ms"))
        assert abs(accounted - metrics["total_ms"]) < 1e-6
        assert abs(sum(metrics["stage_ms"].values()) - metrics["pipeline_ms"]) < 1e-6
        assert all(value >= 0 for value in metrics["stage_ms"].values())
    metrics_path = folder / "metrics.json"
    result = subprocess.run([sys.executable, ROOT / "main.py", source, actual,
                             "--no-bloom", "--warmup", "1", "--repeat", "2", "--metrics-json", metrics_path],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(metrics_path.read_text())
    assert metrics["warmup"] == 1 and len(metrics["samples"]) == 2
    for sample in metrics["samples"]:
        assert "bloom_blur" not in sample["stage_ms"]
        assert sample["total_ms"] > sample["pipeline_ms"] > 0
print("Python pipeline equivalence, timing accounting, and repeat handling passed")
