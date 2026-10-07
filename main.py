"""Command-line wrapper around the original, unmodified AsciiArt algorithm."""
import argparse
import json
import os
from pathlib import Path
import resource
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl-cache"))
from AsciiArt import AsciiArt


def convert(input_path, output_path, bloom=True, edges_output=None, fill_output=None):
    stages = {}

    def timed(name, function, *args):
        start = perf_counter()
        result = function(*args)
        stages[name] = stages.get(name, 0) + (perf_counter() - start) * 1000
        return result

    class MeasuredAsciiArt(AsciiArt):
        def get_luminance(self):
            timed("luminance", super().get_luminance)

    start = perf_counter()
    edges = timed("load_edges_and_luminance", MeasuredAsciiArt, str(input_path), "res/edgesASCII.png")
    timed("dog", edges.difference_of_gaussian, 2.0, 1.6, 1.0, 0.3)
    timed("sobel", edges.sobel)
    timed("magnitude", edges.get_magnitude, 0.2)
    timed("angles", edges.find_angle)
    timed("direction_quantization", edges.edge_quantize)
    timed("edge_voting", edges.controlled_downsample, edges.char_size, 12)
    timed("edge_render", edges.to_ascii_art, True)
    if edges_output:
        timed("write_edges", edges.store_image, str(edges_output))
    fill = timed("load_fill_and_luminance", MeasuredAsciiArt, str(input_path), "res/fillASCII.png")
    input_width, input_height = fill.width, fill.height
    if bloom:
        timed("bloom_blur", fill.get_bloom_data, 0.8, 50)
    timed("fill_downsample", fill.downsample, fill.char_size)
    timed("fill_quantization", fill.quantize, 10)
    timed("fill_render", fill.to_ascii_art, False)
    if fill_output:
        timed("write_fill", fill.store_image, str(fill_output))
    timed("combine", fill.combine, edges)
    if bloom:
        timed("bloom_add", fill.add_bloom_data)
    timed("write_final", fill.store_image, str(output_path))
    width, height = fill.width, fill.height
    del edges, fill
    total_ms = (perf_counter() - start) * 1000
    load_ms = stages.pop("load_edges_and_luminance") + stages.pop("load_fill_and_luminance") - stages["luminance"]
    write_ms = sum(stages.pop(name, 0) for name in ("write_edges", "write_fill", "write_final"))
    pipeline_ms = sum(stages.values())
    return {"total_ms": total_ms, "load_ms": load_ms, "write_ms": write_ms,
            "conversion_ms": pipeline_ms, "pipeline_ms": pipeline_ms,
            "overhead_ms": total_ms - load_ms - write_ms - pipeline_ms,
            "stage_ms": stages, "width": width, "height": height,
            "input_width": input_width, "input_height": input_height,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def main():
    parser = argparse.ArgumentParser(description="Original Python ASCII image converter")
    parser.add_argument("input", nargs="?", default="images/sample_resize.png", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--edges-output", type=Path)
    parser.add_argument("--fill-output", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")
    if args.output is None:
        suffix = "final" if args.no_bloom else "final_with_bloom"
        args.output = Path("output") / f"{args.input.stem}_{suffix}.png"
        if args.edges_output is None:
            args.edges_output = Path("output") / f"{args.input.stem}_edges.png"
    sources = {args.input.resolve(), Path("res/edgesASCII.png").resolve(), Path("res/fillASCII.png").resolve()}
    targets = [path.resolve() for path in (args.output, args.edges_output, args.fill_output, args.metrics_json) if path]
    if len(targets) != len(set(targets)) or sources.intersection(targets):
        parser.error("input, atlas, and output paths must not overlap")
    samples = []
    for index in range(args.warmup + args.repeat):
        sample = convert(args.input, args.output, not args.no_bloom, args.edges_output, args.fill_output)
        if index >= args.warmup:
            samples.append(sample)
    if args.metrics_json:
        args.metrics_json.write_text(json.dumps({"implementation": "python", "warmup": args.warmup,
                                                "samples": samples}, indent=2) + "\n")
    print(f"{args.output}: {sample['width']}x{sample['height']}; conversion {sample['total_ms']:.3f} ms")


if __name__ == "__main__":
    main()
