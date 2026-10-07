"""Reproducible Python/CUDA latency, stage, memory, and fidelity benchmarks."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def command(arguments, env=None):
    result = subprocess.run(list(map(str, arguments)), cwd=ROOT, env=env,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {arguments}\n{result.stderr}\n{result.stdout}")
    return result.stdout.strip()


def optional_command(arguments):
    try:
        return command(arguments)
    except (OSError, RuntimeError) as error:
        return str(error)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = sorted(values)
    position = 0.9 * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return {"n": len(values), "median": statistics.median(values), "mean": statistics.mean(values),
            "min": values[0], "max": values[-1], "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "p90": values[lower] + (values[upper] - values[lower]) * (position - lower)}


def compare(python_path, cuda_path):
    import numpy as np
    from PIL import Image
    expected = np.asarray(Image.open(python_path).convert("L")).astype(np.float64)
    actual = np.asarray(Image.open(cuda_path).convert("L")).astype(np.float64)
    if expected.shape != actual.shape:
        raise RuntimeError(f"Output dimensions differ: {expected.shape}, {actual.shape}")
    difference = abs(actual - expected)
    mse = float(np.mean(difference ** 2))
    return {"shape": list(actual.shape), "max_abs_error": int(difference.max()),
            "mae": float(difference.mean()), "rmse": math.sqrt(mse),
            "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "exact_percent": float(np.mean(difference == 0) * 100),
            "within_2_percent": float(np.mean(difference <= 2) * 100),
            "python_png_sha256": digest(python_path), "cuda_png_sha256": digest(cuda_path)}


def summarize_record(record):
    result = {"fresh_process_ms": summary([s["process_ms"] for s in record["fresh"]])}
    for name in ("total_ms", "load_ms", "conversion_ms", "write_ms", "overhead_ms", "pipeline_ms", "peak_rss_mib"):
        result[name] = summary([s[name] for s in record["steady"]])
    if record["implementation"] == "cuda":
        result["device_buffer_mib"] = summary([s["device_buffer_mib"] for s in record["steady"]])
    result["stage_ms"] = {name: summary([s["stage_ms"][name] for s in record["profile"]])
                          for name in record["profile"][0]["stage_ms"]}
    return result


def write_report(data, destination):
    import numpy as np
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
    import matplotlib.pyplot as plt
    records = data["records"]
    for record in records:
        record["summary"] = summarize_record(record)
    result_dir = destination.parent
    data["report_generator_sha256"] = digest(Path(__file__))
    with (result_dir / "summary.csv").open("w", newline="") as file:
        fields = ["size", "bloom", "implementation", "metric", "n", "median", "mean", "min", "max", "stddev", "p90"]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in records:
            for metric, stats in r["summary"].items():
                if metric == "stage_ms":
                    for stage, stage_stats in stats.items():
                        writer.writerow({"size": r["size"], "bloom": r["bloom"], "implementation": r["implementation"],
                                         "metric": f"stage:{stage}", **stage_stats})
                else:
                    writer.writerow({"size": r["size"], "bloom": r["bloom"], "implementation": r["implementation"],
                                     "metric": metric, **stats})
    destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    env = data["environment"]
    optimized = data.get("optimized_cuda", False)
    pipeline_boundary = ("Optimized CUDA prepares Gaussian coefficients before the CUDA event interval; input/atlas uploads, output downloads, and context setup are excluded. The interval covers GPU processing/normalization and can include host scheduling gaps between one-shot launches." if optimized else "The first CUDA implementation's interval includes Gaussian coefficient generation/uploads and host scheduling gaps; it excludes initial input/atlas upload, output download, and buffer/context setup.")
    lines = ["# Python and CUDA benchmark report", "", f"Measured {data['started_utc']} (UTC).", "",
             "## Environment", "", f"- CPU: {env['cpu']}", f"- GPU/driver/VRAM: `{env['gpu']}`",
             f"- OS: {env['platform']}", f"- Python: {env['python'].splitlines()[0]}",
             f"- Packages: {', '.join(f'{k} {v}' for k, v in env['packages'].items())}",
             f"- CUDA: `{next((line for line in env['nvcc'].splitlines() if 'release' in line), env['nvcc'].splitlines()[-1])}`", f"- Nsight: `{env['nsys']}`",
             f"- Build: Release, architectures `{env['cuda_architectures']}`",
             "- BLAS/OpenMP environment: one thread. No CPU affinity or fixed GPU clocks.",
             "- GPU also drives the desktop. Background activity, power management, thermals, and filesystem cache can affect results.", "",
             "## Method", "",
             f"Each case has {data['settings']['warmup']} discarded warmup conversion(s), {data['settings']['repeat']} measured conversions in one process, and {data['settings']['fresh_repeat']} independent fresh-process runs per implementation. Runs are sequential; implementation order alternates across cases. P90 uses linear interpolation, and standard deviation is the sample standard deviation. These small-sample statistics describe this run, rather than confidence intervals.", "",
             "Inputs are RGB8 PNGs generated by resizing the repository's `images/sample.png` with Pillow Lanczos. The same bytes, 8×8 atlases, and default pipeline settings are passed to both implementations. Bloom is tested both off and on (threshold 0.8, sigma 50). Only the final output is saved. Python calls the original `AsciiArt` methods unchanged, including its otherwise unused magnitude computation and two input decodes. CUDA uses magnitude threshold 0, matching the original edge-selection behavior.", "",
             "**Fresh process** measures subprocess launch through exit, including Python imports or CUDA context startup, loading, processing, encoding, and shutdown. It is process-cold, with normally warm filesystem caches. **Warmed total** measures one conversion inside an already running process: input/atlas decoding, processing, per-call allocations/transfers, PNG encoding/write, and cleanup. Images are decoded again and buffers allocated again on every iteration; there is no persistent image cache. PNG writes are not fsynced. Python writes RGBA PNG, CUDA grayscale PNG, so end-to-end speedup includes their existing I/O differences.", "",
             "**Python pipeline** is the sum of timed processing methods, including both luminance conversions, excluding loading and PNG normalization/encoding. **CUDA pipeline** uses CUDA events. CUDA_BOUNDARY Pipeline ratios are useful but cover different boundaries; warmed total and fresh-process latency are the primary comparisons. `conversion_ms` for CUDA is the full C API call, including allocations, transfers, event queries, and CUDA buffer cleanup.", "",
             "Latency samples disable extra CUDA stage events. CUDA stage profiles are collected separately with stage events enabled; these intervals include launch/scheduling overhead, with coefficient preparation and synchronization inside the first implementation’s Gaussian intervals but outside optimized intervals. Stage medians need not sum to the median pipeline time. Python stage instrumentation times whole methods with `perf_counter`, rather than each pixel. cProfile and Nsight captures are separate diagnostic runs and do not supply headline latency numbers.", "",
             "## Latency and throughput", "",
             "All times are medians in milliseconds. Throughput is input megapixels divided by warmed total; it is single-image throughput, rather than video FPS.", "",
             "| Input | Bloom | Python fresh | CUDA fresh | Fresh speedup | Python warm | CUDA warm | Warm speedup | Python MP/s | CUDA MP/s | Python pipeline | CUDA pipeline |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    cases = []
    for index in range(0, len(records), 2):
        pair = {r["implementation"]: r for r in records[index:index+2]}
        py, cu = pair["python"], pair["cuda"]
        p, c = py["summary"], cu["summary"]
        width, height = map(int, py["size"].split("x"))
        pixels = width * height / 1000
        pf, cf = p["fresh_process_ms"]["median"], c["fresh_process_ms"]["median"]
        pw, cw = p["total_ms"]["median"], c["total_ms"]["median"]
        lines.append(f"| {py['size']} | {'on' if py['bloom'] else 'off'} | {pf:.2f} | {cf:.2f} | {pf/cf:.2f}× | {pw:.2f} | {cw:.2f} | {pw/cw:.2f}× | {pixels/pw:.2f} | {pixels/cw:.2f} | {p['pipeline_ms']['median']:.2f} | {c['pipeline_ms']['median']:.2f} |")
        cases.append((py, cu))
    lines += ["", "## Variability, I/O, and memory", "",
              "Warmed total and P90 are milliseconds; ± is sample standard deviation. RSS is the process high-water mark across warmups and measured conversions, including imports/runtime/context overhead. Device memory uses requested buffer allocations, excluding CUDA context/driver storage. The optimized arena includes coefficients/alignment; the first implementation’s counter omits its small coefficient allocations. Linux ru_maxrss can retain the launcher’s pre-exec memory high-water mark, so RSS is an upper bound rather than isolated converter working memory. Host RSS and device memory are separate measures. CPU/GPU clocks are not locked, so timing variation can occasionally make a fresh-process run faster than a warmed run despite startup costs.", "",
              "| Input | Bloom | Impl. | Warm total ± SD | P90 | Min–max | Load | Conversion/API | Save | Host peak RSS MiB | CUDA image buffers MiB |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        s = r["summary"]
        t = s["total_ms"]
        buffer = f"{s['device_buffer_mib']['median']:.2f}" if "device_buffer_mib" in s else "—"
        lines.append(f"| {r['size']} | {'on' if r['bloom'] else 'off'} | {r['implementation']} | {t['median']:.2f} ± {t['stddev']:.2f} | {t['p90']:.2f} | {t['min']:.2f}–{t['max']:.2f} | {s['load_ms']['median']:.2f} | {s['conversion_ms']['median']:.2f} | {s['write_ms']['median']:.2f} | {s['peak_rss_mib']['max']:.2f} | {buffer} |")
    lines += ["", "## Stage profiles", "", "Stage medians in milliseconds. Every case is included in the CSV/JSON; the following tables show 1080p when present (otherwise the largest tested input).", ""]
    profile_size = "1920x1080" if any(r["size"] == "1920x1080" for r in records) else records[-1]["size"]
    for impl in ("python", "cuda"):
        selected = [r for r in records if r["size"] == profile_size and r["implementation"] == impl]
        lines += [f"### {'CUDA' if impl == 'cuda' else 'Python'} at {profile_size}", "", "| Stage | Bloom off | Bloom on |", "| --- | ---: | ---: |"]
        by_bloom = {r["bloom"]: r for r in selected}
        stages = dict.fromkeys(name for r in selected for name in r["summary"]["stage_ms"])
        for stage in stages:
            values = [by_bloom.get(b, {}).get("summary", {}).get("stage_ms", {}).get(stage, {}).get("median") for b in (False, True)]
            cells = [f"{v:.4f}" if v is not None else "—" for v in values]
            lines.append(f"| {stage} | {cells[0]} | {cells[1]} |")
        lines.append("")
    # Actual profiler outputs supplement event/method intervals; they are not headline timings.
    if not data["settings"]["skip_profilers"] and (result_dir / "python_profiles.json").exists():
        profiles = json.loads((result_dir / "python_profiles.json").read_text())
        lines += ["## cProfile and Nsight Systems", "",
                  f"Separate diagnostic captures use {profile_size}. cProfile includes one process with imports and one conversion. Nsight captures one warmup and one measured conversion, so its totals cover **two conversions**, including the warmup. It aggregates each kernel function across Gaussian sigmas; its Gaussian rows combine edge and bloom blur work. These profiled times include instrumentation and must not replace the unprofiled latency samples above.", "",
                  "### Python cProfile", "", "Cumulative time in milliseconds for the five most expensive original processing methods. Cumulative times are nested and must not be summed.", "",
                  "| Bloom | Function | Calls | Self ms | Cumulative ms |", "| --- | --- | ---: | ---: | ---: |"]
        for mode, stats in profiles.items():
            original = [r for r in stats["functions"] if r["file"].endswith("AsciiArt.py") and r["function"] not in ("<module>", "__init__")]
            for row in original[:5]:
                lines.append(f"| {'on' if mode == 'bloom' else 'off'} | `{row['function']}` | {row['calls']} | {row['self_ms']:.3f} | {row['cumulative_ms']:.3f} |")
        for report, title, name_field, count_field in (
            ("cuda_gpu_kern_sum", "CUDA kernels", "Name", "Instances"),
            ("cuda_api_sum", "CUDA API calls", "Name", "Num Calls"),
            ("cuda_gpu_mem_time_sum", "GPU memory transfers", "Operation", "Count")):
            lines += ["", f"### {title}", "", "| Bloom | Operation | Count | Total ms | Mean ms | Share % |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"]
            for mode in ("no_bloom", "bloom"):
                with (result_dir / f"nsight_{mode}_{report}.csv").open() as file:
                    rows = list(csv.DictReader(file))
                if report == "cuda_api_sum":
                    rows = rows[:5]
                for row in rows:
                    name = row[name_field]
                    if report == "cuda_gpu_kern_sum":
                        name = name.split("(", 1)[0].removeprefix("void ") if "gaussian" not in name else ("gaussian_horizontal" if "<(bool)1>" in name else "gaussian_vertical")
                    lines.append(f"| {'on' if mode == 'bloom' else 'off'} | `{name}` | {row[count_field]} | {float(row['Total Time (ns)'])/1e6:.4f} | {float(row['Avg (ns)'])/1e6:.4f} | {row['Time (%)']} |")
        lines += ["", "A high `cudaFree` API time can include synchronization with outstanding kernels; it does not establish that deallocation alone is expensive. GPU transfer times exclude the CPU-side cost of pageable-memory staging and API scheduling. See the accompanying CSVs for every API row and transferred byte count.", ""]
    warm_speedups = [py["summary"]["total_ms"]["median"] / cu["summary"]["total_ms"]["median"] for py, cu in cases]
    fresh_speedups = [py["summary"]["fresh_process_ms"]["median"] / cu["summary"]["fresh_process_ms"]["median"] for py, cu in cases]
    lines += ["## What the measurements show", "",
              f"Across this input sweep, CUDA reduces warmed end-to-end conversion time by **{min(warm_speedups):.2f}–{max(warm_speedups):.2f}×**. Fresh-process speedup is smaller at **{min(fresh_speedups):.2f}–{max(fresh_speedups):.2f}×**, because imports and CUDA context startup matter most for small inputs. These are comparisons with this original Python baseline on this machine, rather than a general CUDA-versus-CPU speedup or a comparison against optimized native CPU code.", ""]
    py_example = next((r for r in records if r["size"] == profile_size and r["bloom"] and r["implementation"] == "python"), None)
    cu_example = next((r for r in records if r["size"] == profile_size and r["bloom"] and r["implementation"] == "cuda"), None)
    if py_example and cu_example:
        p, c = py_example["summary"], cu_example["summary"]
        direction = p["stage_ms"]["direction_quantization"]["median"]
        io_share = (c["load_ms"]["median"] + c["write_ms"]["median"]) / c["total_ms"]["median"] * 100
        lines += [f"At {profile_size} with bloom, Python's per-pixel direction loop alone takes {direction:.2f} ms ({direction/p['pipeline_ms']['median']*100:.1f}% of its processing time). The CPU bloom blur adds {p['stage_ms']['bloom_blur']['median']:.2f} ms. CUDA's event profile places its largest GPU interval in bloom blur ({c['stage_ms']['bloom_blur']['median']:.2f} ms), while decoding and PNG saving together account for about {io_share:.1f}% of warmed CUDA total time. The current CUDA application is therefore primarily limited by image I/O at this size. These one-shot measurements include repeated decode/encode. Reusable contexts and GPU-resident execution are measured separately in OPTIMIZATION.md.", ""]
    if "validation" in data:
        lines += ["## Validation", "", data["validation"]["description"], ""]
    profiler_artifacts = (["- [Python cProfile summaries](benchmarks/results/python_profiles.json)",
                           "- Nsight kernel/API/transfer CSV summaries: `benchmarks/results/nsight_*_*.csv`."]
                          if not data["settings"]["skip_profilers"] else ["- cProfile/Nsight captures were skipped for this run."])
    lines += ["## Output agreement", "", "Comparison uses decoded grayscale pixels, rather than PNG file bytes. PSNR uses a peak of 255. Exact matches have undefined/infinite PSNR (shown as ∞). Threshold decisions, atlas quantization, and float32 versus float64 intermediate operations can change whole glyphs, so maximum error alone is not a visual-quality assessment.", "",
              "| Input | Bloom | Output W×H | MAE | RMSE | Max error | Exact pixels % | Within 2 levels % | PSNR dB |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for py, cu in cases:
        q = cu["agreement"]
        psnr = f"{q['psnr_db']:.2f}" if q["psnr_db"] is not None else "∞"
        lines.append(f"| {cu['size']} | {'on' if cu['bloom'] else 'off'} | {q['shape'][1]}×{q['shape'][0]} | {q['mae']:.4f} | {q['rmse']:.4f} | {q['max_abs_error']} | {q['exact_percent']:.3f} | {q['within_2_percent']:.3f} | {psnr} |")
    lines += ["", "## Artifacts and reproduction", "",
              "- [Raw timings, metadata, input and source SHA-256 hashes](benchmarks/results/results.json)",
              "- [All summary statistics and every stage/case](benchmarks/results/summary.csv)",
              "- [Latency plot](benchmarks/results/latency.png) · [SVG for publication](benchmarks/results/latency.svg)",
              "- [Exact Python dependency lock](benchmarks/results/python_requirements.txt)",
              *profiler_artifacts, "",
              "```sh", "python3 -m venv .venv", ".venv/bin/python -m pip install -r benchmarks/results/python_requirements.txt",
              f"cmake -S . -B build -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CUDA_ARCHITECTURES={env['cuda_architectures']}'",
              "cmake --build build -j", ".venv/bin/python benchmarks/run.py", "```", "",
              "The script needs GPU access and stops on a failed CUDA run. `--sizes`, `--repeat`, `--fresh-repeat`, `--warmup`, `--device`, and `--skip-profilers` allow a different run; the settings and exact commands are recorded in JSON. Regenerate the tables/plot from existing data with `--report-only`. Binary `.prof`, `.nsys-rep`, `.sqlite`, and scratch images are local artifacts ignored by Git; text summaries and the plot are publishable.", ""]
    if not optimized:
        lines[2:2] = ["These measurements cover the first CUDA implementation, before the latest kernel and I/O optimizations. See [OPTIMIZATION.md](OPTIMIZATION.md) for the current version compared against that committed baseline.", ""]
    if not optimized:
        index = lines.index("## Artifacts and reproduction") + 2
        lines[index:index] = ["Use baseline commit `3f711bb` in a separate checkout before running the commands below. The current checkout contains the optimized implementation.", ""]
    lines = [line.replace("CUDA_BOUNDARY", pipeline_boundary) for line in lines]
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, bloom in zip(axes, (False, True)):
        selected = [(py, cu) for py, cu in cases if py["bloom"] == bloom]
        x = np.arange(len(selected))
        for offset, impl, color in ((-0.18, "python", "#4267ac"), (0.18, "cuda", "#42935b")):
            rs = [pair[0 if impl == "python" else 1] for pair in selected]
            values = [r["summary"]["total_ms"]["median"] for r in rs]
            axis.bar(x + offset, values, width=0.36, label="CUDA" if impl == "cuda" else "Python", color=color)
        axis.set_xticks(x, [pair[0]["size"] for pair in selected], rotation=35, ha="right")
        axis.set_yscale("log")
        axis.set_ylabel("Warmed conversion, median ms (log scale)")
        axis.set_title(f"Bloom {'on' if bloom else 'off'}")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Original Python vs CUDA\n{env['cpu']} · {env['gpu'].split(',')[0]}", fontsize=11)
    fig.savefig(result_dir / "latency.png", dpi=180)
    fig.savefig(result_dir / "latency.svg")
    # Matplotlib emits trailing spaces in SVG paths; keep generated text Git-clean.
    svg_path = result_dir / "latency.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def capture_profiles(args, source, env, work, results):
    profiles = {}
    for bloom in (False, True):
        mode = "bloom" if bloom else "no_bloom"
        profile = results / f"python_{mode}.prof"
        python_cmd = [sys.executable, "-m", "cProfile", "-o", profile, ROOT / "main.py", source, work / "profile_python.png"]
        if not bloom:
            python_cmd.append("--no-bloom")
        command(python_cmd, env)
        stats = pstats.Stats(str(profile))
        rows = []
        for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
            rows.append({"file": str(filename).replace(str(ROOT), "."), "line": line, "function": name,
                         "primitive_calls": primitive, "calls": calls, "self_ms": own*1000, "cumulative_ms": cumulative*1000})
        profiles[mode] = {"total_profile_ms": stats.total_tt*1000,
                          "functions": sorted(rows, key=lambda r: r["cumulative_ms"], reverse=True)}
        nsys_base = results / f"nsight_{mode}"
        cuda_cmd = [ROOT / args.build / "ascii_benchmark", source, work / "profile_cuda.png", work / "nsight_metrics.json",
                    "1", "1", int(bloom), "0", args.device]
        command(["nsys", "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda", "--force-overwrite=true",
                 "-o", nsys_base, *cuda_cmd], env)
        command(["nsys", "stats", "--force-overwrite=true", "--report",
                 "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum",
                 "--format", "csv", "--output", nsys_base, str(nsys_base) + ".nsys-rep"], env)
        print(f"Captured cProfile/Nsight: {mode}", flush=True)
    (results / "python_profiles.json").write_text(json.dumps(profiles, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256x256,512x512,812x812,1000x1250,1920x1080,3840x2160")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fresh-repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build", default="build")
    parser.add_argument("--skip-profilers", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if min(args.repeat, args.fresh_repeat, args.profile_repeat) < 1 or args.warmup < 0 or args.device < 0:
        parser.error("repeat counts must be positive; warmup/device must be nonnegative")
    results, work = ROOT / "benchmarks/results", ROOT / "benchmarks/work"
    results.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    destination = results / "results.json"
    if args.report_only:
        write_report(json.loads(destination.read_text()), destination)
        return
    from PIL import Image
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(ROOT / ".mpl-cache")
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    cache = (ROOT / args.build / "CMakeCache.txt").read_text()
    architectures = next((line.split("=", 1)[1] for line in cache.splitlines() if line.startswith("CMAKE_CUDA_ARCHITECTURES:")), "75;80;86;89;90")
    files = [ROOT / "AsciiArt.py", ROOT / "main.py", ROOT / "CMakeLists.txt", ROOT / "benchmarks/run.py", ROOT / "requirements.txt",
             *sorted((ROOT / "cuda").glob("*"))]
    data = {"schema_version": 1, "optimized_cuda": True, "started_utc": datetime.now(timezone.utc).isoformat(),
            "settings": vars(args), "environment": {"cpu": cpu, "platform": platform.platform(), "python": sys.version,
             "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "Pillow")},
             "package_lock": {package.metadata['Name']: package.version for package in importlib.metadata.distributions()},
             "atlas_hashes": {name: digest(ROOT / name) for name in ("res/fillASCII.png", "res/edgesASCII.png")},
             "io_library_versions": optional_command(["pkg-config", "--modversion", "libpng", "libjpeg"]),
             "host_compiler": optional_command(["cc", "--version"]),
             "gpu": command(["nvidia-smi", "--id", args.device, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
             "gpu_snapshot_before": command(["nvidia-smi"]), "nvcc": command(["nvcc", "--version"]),
             "nsys": optional_command(["nsys", "--version"]), "cuda_architectures": architectures,
             "thread_environment": {name: environment[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
             "git_revision": command(["git", "rev-parse", "HEAD"]), "git_status": command(["git", "status", "--short"])},
            "source_hashes": {str(f.relative_to(ROOT)): digest(f) for f in files}, "inputs": [], "records": []}
    lock = data["environment"]["package_lock"]
    (results / "python_requirements.txt").write_text("\n".join(f"{name}=={version}" for name, version in sorted(lock.items()) if name.lower() not in ("pip", "setuptools")) + "\n")
    source_image = Image.open(ROOT / "images/sample.png").convert("RGB")
    sizes = args.sizes.split(",")
    for case_index, size in enumerate(sizes):
        width, height = map(int, size.split("x"))
        if min(width, height) < 8:
            parser.error("input sizes must be at least 8x8")
        source = work / f"input_{size}.png"
        source_image.resize((width, height), Image.Resampling.LANCZOS).save(source)
        data["inputs"].append({"size": size, "source": "images/sample.png", "source_sha256": digest(ROOT / "images/sample.png"),
                               "generated_sha256": digest(source), "bytes": source.stat().st_size})
        for bloom in (False, True):
            mode = "bloom" if bloom else "no_bloom"
            pair = {}
            order = ("python", "cuda") if (case_index + int(bloom)) % 2 == 0 else ("cuda", "python")
            for impl in order:
                output = work / f"{size}_{mode}_{impl}.png"
                metrics = work / f"{size}_{mode}_{impl}.json"

                def worker(warmup, repeat, profile=False):
                    if impl == "python":
                        cmd = [sys.executable, ROOT / "main.py", source, output, "--metrics-json", metrics,
                               "--warmup", warmup, "--repeat", repeat]
                        if not bloom:
                            cmd.append("--no-bloom")
                    else:
                        cmd = [ROOT / args.build / "ascii_benchmark", source, output, metrics,
                               warmup, repeat, int(bloom), int(profile), args.device]
                    start = perf_counter()
                    command(cmd, environment)
                    process_ms = (perf_counter() - start) * 1000
                    return json.loads(metrics.read_text())["samples"], process_ms, list(map(str, cmd))

                steady, _, invocation = worker(args.warmup, args.repeat)
                fresh = []
                for _ in range(args.fresh_repeat):
                    samples, process_ms, _ = worker(0, 1)
                    fresh.append({"process_ms": process_ms, "sample": samples[0]})
                profile = worker(args.warmup, args.profile_repeat, True)[0] if impl == "cuda" else steady
                pair[impl] = {"size": size, "bloom": bloom, "implementation": impl, "command": invocation,
                              "steady": steady, "fresh": fresh, "profile": profile}
                print(f"{size} {mode} {impl}: warm median {statistics.median(s['total_ms'] for s in steady):.2f} ms", flush=True)
            pair["cuda"]["agreement"] = compare(work / f"{size}_{mode}_python.png", work / f"{size}_{mode}_cuda.png")
            data["records"].extend([pair["python"], pair["cuda"]])
            destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if not args.skip_profilers:
        profile_size = "1920x1080" if "1920x1080" in sizes else sizes[-1]
        capture_profiles(args, work / f"input_{profile_size}.png", environment, work, results)
    data["environment"]["gpu_snapshot_after"] = command(["nvidia-smi"])
    data["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(data, destination)
    print("Wrote BENCHMARKS.md, raw JSON, summary CSV, and latency.png", flush=True)


if __name__ == "__main__":
    main()
