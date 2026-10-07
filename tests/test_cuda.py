"""Independent NumPy oracle. No SciPy, matplotlib, or CPU fallback in the converter."""
import pathlib
import json
import struct
import subprocess
import sys
import tempfile
import zlib

EXE = str(pathlib.Path(sys.argv[1]).resolve())
ROOT = pathlib.Path(sys.argv[2]).resolve()


def run(*arguments):
    return subprocess.run([EXE, *map(str, arguments)], cwd=ROOT,
                          capture_output=True, text=True)


def host_tests():
    cases = [([], "Usage:"), (["--bogus", "x"], "Unknown option"),
             (["x", "y", "--sigma", "nan"], "Invalid value"),
             (["x", "y", "--edge-votes", "-1"], "Invalid value"),
             (["x", "y", "--device", "999999999999999999999"], "Invalid value"),
             (["x", "y", "--sigma"], "Missing value"),
             (["same.png", "same.png"], "must not overlap"),
             (["images/sample_resize.png", "res/fillASCII.png"], "must not overlap"),
             (["images/sample_resize.png", "x.png", "--sigma", "0"], "invalid numeric"),
             (["images/sample_resize.png", "x.png", "--scale", "-2"], "invalid numeric"),
             (["images/sample_resize.png", "x.png", "--bloom-sigma", "1025"], "invalid numeric")]
    for arguments, message in cases:
        result = run(*arguments)
        assert result.returncode != 0, (arguments, result.stdout)
        assert message in result.stderr, (arguments, result.stderr)
    result = run("--help")
    assert result.returncode == 0 and "--no-bloom" in result.stdout

    # Generate fixtures using only the standard library; host tests need no Python packages.
    def write_png(path, width, height):
        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data +
                    struct.pack(">I", zlib.crc32(kind + data)))
        data = (b"\x00" + b"\x80\x80\x80" * width) * height
        path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                         chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                         chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b""))

    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        output, fixture = folder / "output.png", folder / "fixture.png"
        write_png(fixture, 7, 8)
        result = run(fixture, output)
        assert result.returncode != 0 and "smaller than one glyph" in result.stderr
        write_png(fixture, 39, 8)
        result = run("images/sample_resize.png", output, "--edge-atlas", fixture)
        assert result.returncode != 0 and "atlases must be" in result.stderr
        write_png(fixture, 40, 4)
        result = run("images/sample_resize.png", output, "--fill-atlas", fixture)
        assert result.returncode != 0 and "atlases must be" in result.stderr
        fixture.write_bytes(b"not an image")
        result = run(fixture, output)
        assert result.returncode != 0 and "unsupported input" in result.stderr
        assert not output.exists()
    print("CLI parsing and pre-CUDA validation passed")


def gpu_tests():
    # Probe before importing optional packages so a missing GPU is a clear skip.
    with tempfile.TemporaryDirectory() as folder:
        probe = run("images/sample_resize.png", pathlib.Path(folder) / "probe.png", "--no-bloom")
    if probe.returncode:
        unavailable = ("CUDA driver version is insufficient", "no CUDA-capable device",
                       "system has unsupported display driver", "initialization error",
                       "driver shutting down", "system not yet initialized")
        if any(message in probe.stderr for message in unavailable):
            print("SKIP: CUDA device/driver unavailable: " + probe.stderr.strip())
            return 77
        raise AssertionError(probe.stderr)
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        print(f"SKIP: reference tests require NumPy and Pillow: {error}")
        return 77

    def blur(image, sigma):
        radius = int(4 * sigma + 0.5)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
        result = image.astype(np.float64)
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius, radius)
            padded = np.pad(result, padding, mode="symmetric")
            result = np.zeros_like(result)
            for offset, weight in enumerate(weights):
                window = [slice(None), slice(None)]
                window[axis] = slice(offset, offset + result.shape[axis])
                result += padded[tuple(window)] * weight
        return result

    def oracle(rgb, fill, edge, bloom=True, normalize=True, sigma=2,
               scale=1.6, tau=1, dog_threshold=0.3, magnitude=0, votes=12,
               bloom_sigma=50, bloom_threshold=0.8):
        gray = np.dot(rgb[..., :3].astype(float) / 255, [0.2989, 0.5870, 0.1140])
        dog = ((1 + tau) * blur(gray, sigma) - tau * blur(gray, sigma * scale) >= dog_threshold).astype(float)
        p = np.pad(dog, 1, mode="symmetric")
        gx = p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:]
        gy = p[:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2*p[1:-1, :-2] - p[2:, :-2]
        theta = np.arctan2(gy, gx)
        angle = abs(theta) / np.pi
        theta = np.where(angle <= 0.2, 0, theta)
        angle = np.where(angle <= 0.2, 0, angle)
        directions = np.full(gray.shape, -1, dtype=int)
        present = ((gx != 0) | (gy != 0)) & (np.hypot(gx, gy) >= magnitude)
        directions[present & ((angle < 0.05) | (angle > 0.9))] = 1
        directions[present & (angle > 0.45) & (angle < 0.55)] = 0
        lower = present & (angle > 0.05) & (angle < 0.45)
        upper = present & (angle > 0.55) & (angle < 0.9)
        directions[lower] = np.where(theta[lower] > 0, 2, 3)
        directions[upper] = np.where(theta[upper] > 0, 3, 2)
        size = fill.shape[0]
        h, w = (gray.shape[0] // size) * size, (gray.shape[1] // size) * size
        fills, edges, combined = (np.zeros((h, w)) for _ in range(3))
        glyphs = fill.shape[1] // size
        for y in range(0, h, size):
            for x in range(0, w, size):
                chunk = directions[y:y+size, x:x+size].ravel()
                valid = chunk[chunk >= 0].tolist()
                chosen = max(valid, key=valid.count) if valid else -1
                if chosen >= 0 and valid.count(chosen) <= votes:
                    chosen = -1
                index = min(int(gray[y, x] * glyphs), glyphs - 1)
                fg = fill[:, index*size:(index+1)*size] / 255
                eg = edge[:, (chosen+1)*size:(chosen+2)*size] / 255
                fills[y:y+size, x:x+size] = fg
                edges[y:y+size, x:x+size] = eg
                combined[y:y+size, x:x+size] = eg if np.any(eg) else fg
        if bloom:
            combined += blur(np.where(gray > bloom_threshold, gray, 0), bloom_sigma)[:h, :w]

        def encode(image):
            if normalize:
                lo, hi = image.min(), image.max()
                image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
            return np.clip(np.floor(image * 256), 0, 255).astype(np.uint8)
        return tuple(encode(a) for a in (combined, edges, fills))

    rng = np.random.default_rng(9321)
    # Distinct, asymmetric glyphs detect transposition, indexing, and whole-cell replacement.
    fill = rng.integers(0, 256, (8, 80), dtype=np.uint8)
    edge = rng.integers(0, 256, (8, 40), dtype=np.uint8)
    edge[:, :8] = 0
    ramp = np.broadcast_to(np.linspace(0, 255, 43, dtype=np.uint8)[None, :, None], (35, 43, 3)).copy()
    step = np.zeros((48, 48, 3), dtype=np.uint8)
    step[:, 24:] = 255
    diagonal = np.repeat(((np.indices((48, 48)).sum(axis=0) > 47)*255).astype(np.uint8)[..., None], 3, axis=2)
    rgba = rng.integers(0, 256, (33, 35, 4), dtype=np.uint8)
    cases = [
        ("random", rng.integers(0, 256, (37, 51, 3), dtype=np.uint8), {}, []),
        ("ramp_no_bloom", ramp, {"bloom": False}, ["--no-bloom"]),
        ("step", step, {"bloom": False}, ["--no-bloom"]),
        ("diagonal", diagonal, {"bloom": False, "votes": 0}, ["--no-bloom", "--edge-votes", "0"]),
        ("dark", np.zeros((8, 8, 3), dtype=np.uint8), {}, []),
        ("white", np.full((17, 19, 3), 255, dtype=np.uint8), {}, []),
        ("fixed_range", ramp, {"normalize": False}, ["--fixed-range"]),
        ("magnitude", step, {"magnitude": 5}, ["--magnitude-threshold", "5"]),
        ("votes", diagonal, {"votes": 64}, ["--edge-votes", "64"]),
        ("alpha", rgba, {"bloom_sigma": 1.25}, ["--bloom-sigma", "1.25"]),
        ("wide_kernel", ramp[-16:, -16:], {"bloom_sigma": 110}, ["--bloom-sigma", "110"]),
        ("custom", ramp, {"sigma": 0.7, "scale": 2, "tau": 0.5, "dog_threshold": 0.45,
                            "bloom_threshold": 0.5, "bloom_sigma": 3},
         ["--sigma", "0.7", "--scale", "2", "--tau", "0.5", "--dog-threshold", "0.45",
          "--bloom-threshold", "0.5", "--bloom-sigma", "3"]),
    ]
    with tempfile.TemporaryDirectory() as folder:
        folder = pathlib.Path(folder)
        fill_path, edge_path = folder / "fill.png", folder / "edge.png"
        Image.fromarray(fill).save(fill_path)
        Image.fromarray(edge).save(edge_path)
        for name, rgb, options, flags in cases:
            source = folder / f"{name}.png"
            Image.fromarray(rgb).save(source)
            outputs = [folder / f"{name}_{kind}.png" for kind in ("final", "edges", "fill")]
            result = run(source, outputs[0], "--fill-atlas", fill_path, "--edge-atlas", edge_path,
                         "--edges-output", outputs[1], "--fill-output", outputs[2], *flags)
            assert result.returncode == 0, (name, result.stderr)
            expected = oracle(rgb, fill, edge, **options)
            for path, reference in zip(outputs, expected):
                actual = np.asarray(Image.open(path).convert("L"))
                assert actual.shape == reference.shape, (name, actual.shape, reference.shape)
                difference = np.abs(actual.astype(int) - reference.astype(int))
                assert difference.max() <= 2, (name, path.name, difference.max(), difference.mean())
            print(f"PASS: {name} (final, edges, fill)")
        # Constant atlases exercise the zero-range normalization path.
        fill[:] = 128
        Image.fromarray(fill).save(fill_path)
        result = run(source, outputs[0], "--fill-atlas", fill_path,
                     "--edge-atlas", edge_path, "--no-bloom", "--edge-votes", "64")
        assert result.returncode == 0, result.stderr
        assert not np.asarray(Image.open(outputs[0])).any()
        # Non-8 glyph dimensions and a different fill count are supported.
        fill = rng.integers(0, 256, (4, 24), dtype=np.uint8)
        edge = rng.integers(0, 256, (4, 20), dtype=np.uint8)
        edge[:, :4] = 0
        Image.fromarray(fill).save(fill_path)
        Image.fromarray(edge).save(edge_path)
        Image.fromarray(ramp).save(source)
        result = run(source, outputs[0], "--fill-atlas", fill_path, "--edge-atlas", edge_path,
                     "--edge-votes", "0", "--no-bloom")
        assert result.returncode == 0, result.stderr
        difference = np.abs(np.asarray(Image.open(outputs[0])).astype(int) -
                            oracle(ramp, fill, edge, bloom=False, votes=0)[0].astype(int))
        assert difference.max() <= 2
        # JPEG and grayscale decoding are checked against the decoded RGB pixels.
        for name, mode in (("jpeg", "RGB"), ("gray", "L")):
            source = folder / ("input.jpg" if name == "jpeg" else "gray.png")
            Image.fromarray(ramp).convert(mode).save(source)
            decoded = np.asarray(Image.open(source).convert("RGB"))
            result = run(source, outputs[0], "--fill-atlas", fill_path,
                         "--edge-atlas", edge_path, "--no-bloom")
            assert result.returncode == 0, result.stderr
            difference = np.abs(np.asarray(Image.open(outputs[0])).astype(int) -
                                oracle(decoded, fill, edge, bloom=False)[0].astype(int))
            assert difference.max() <= 2, (name, difference.max())
        # Optional profiling must preserve output and produce consistent timing data.
        benchmark = pathlib.Path(EXE).parent / "ascii_benchmark"
        metrics_path = folder / "metrics.json"
        previous = None
        for profile in (0, 1):
            result = subprocess.run([str(benchmark), str(source), str(outputs[0]), str(metrics_path),
                                     "1", "2", "1", str(profile), "0"], cwd=ROOT, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            metrics = json.loads(metrics_path.read_text())
            assert len(metrics["samples"]) == 2
            for sample in metrics["samples"]:
                accounted = sum(sample[name] for name in ("load_ms", "conversion_ms", "write_ms", "overhead_ms"))
                assert abs(accounted - sample["total_ms"]) < 1e-5
                assert sample["total_ms"] >= sample["conversion_ms"] >= sample["pipeline_ms"] > 0
                assert sample["device_buffer_mib"] > 0
                assert len(sample["stage_ms"]) == (10 if profile else 0)
                assert all(value >= 0 for value in sample["stage_ms"].values())
                assert sum(sample["stage_ms"].values()) <= sample["pipeline_ms"] + 0.1
            current = np.asarray(Image.open(outputs[0])).copy()
            if previous is not None:
                assert np.array_equal(previous, current)
            previous = current
    print("GPU reference tests passed")
    return 0


if __name__ == "__main__":
    if sys.argv[3] == "--host":
        host_tests()
    else:
        sys.exit(gpu_tests())
