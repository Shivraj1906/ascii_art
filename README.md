# ASCII art generator

- A small scoped and fun project that converts an image into ASCII art. Inspired by [Acerola's](https://www.youtube.com/@Acerola_t) video on turning video games into text using HLSL.
- The original Python implementation uses matplotlib, NumPy, and SciPy. A complete CUDA C implementation is available in [`cuda/`](./cuda/), with a C command-line interface and a reusable C API.

## Python version

Create an isolated environment with Python 3.12 or later and the package versions used for the published benchmarks:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py images/sample_resize.png output/python_final.png
.venv/bin/python main.py images/sample_resize.png output/python_no_bloom.png --no-bloom
```

[`AsciiArt.py`](./AsciiArt.py) retains the original algorithm. [`main.py`](./main.py) supplies a headless CLI with optional `--edges-output`, `--fill-output`, and machine-readable `--metrics-json` timing data. Running it without arguments retains the original sample-image workflow. Both CLIs use the same atlases and default processing settings; output directories must already exist.

## Benchmarks

The [benchmark report](./BENCHMARKS.md) compares the original Python implementation and CUDA on six image sizes from 256×256 through 3840×2160, with bloom off and on. It includes fresh-process latency, warmed conversion latency, variability, throughput, stage profiles, memory, and decoded output agreement. [Raw measurements](./benchmarks/results/results.json), [all summary statistics](./benchmarks/results/summary.csv), and the profiling summaries accompany the report.

The report preserves measurements of the **first CUDA implementation** on an i7-12650H and RTX 3060 Laptop GPU. In that run, warmed end-to-end conversion was **19.19–34.26× faster** with CUDA across this suite. Representative median times include loading, processing, saving, and cleanup:

| Input | Bloom | Python | CUDA | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1920×1080 | off | 1,631.72 ms | 56.85 ms | 28.70× |
| 1920×1080 | on | 2,033.09 ms | 93.20 ms | 21.81× |
| 3840×2160 | off | 6,905.57 ms | 201.57 ms | 34.26× |
| 3840×2160 | on | 7,490.71 ms | 315.60 ms | 23.73× |

Fresh-process speedups are smaller because they include Python imports or CUDA context startup. No-bloom outputs match Python exactly in every measured case; all bloom output pixels differ by at most two intensity levels. The report documents timing boundaries, the desktop GPU workload, and other limits on interpreting these results.

To benchmark the current Python and CUDA versions on a Linux machine with a working GPU:

```sh
.venv/bin/python benchmarks/run.py
```

Use commit `3f711bb` in a separate checkout to reproduce the historical first-version table. The current default suite uses one warmup, five measured conversions, three fresh-process runs, and separate CUDA stage-event profiles. It also captures Python cProfile and NVIDIA Nsight Systems summaries for 1080p. Use `--skip-profilers` when Nsight is unavailable, or `--sizes 512x512,1920x1080` for a shorter suite. The script records environment metadata, source/input hashes, settings, exact worker commands, and every sample, then generates `BENCHMARKS.md` and a standalone plot. Timings are measured on the local GPU; no estimated speedups are used.

![Measured warmed conversion latency](./benchmarks/results/latency.png)

The current version adds tiled/fused kernels, cached contexts, CUDA graphs, exact FFT bloom for large reusable frames, GPU-resident buffers, and faster lossless PNG encoding. [OPTIMIZATION.md](./OPTIMIZATION.md) compares it against the committed first implementation in the same session, with [raw data and all statistics](./benchmarks/optimized_results/). Its timings distinguish single-image calls, reusable contexts, and resident execution.

## Build the CUDA version

Requirements:

- An NVIDIA GPU with a working CUDA driver for execution.
- CUDA Toolkit 12 or later, including cuFFT, with `nvcc`, a compatible C/C++ host compiler, and CMake 3.24 or later. CUDA compiles `.cu` files through its C++ frontend; the host interface and image I/O are C, and the kernels use C-style code.
- libpng and libjpeg development packages (`libpng-dev` and `libjpeg-dev` on Debian/Ubuntu).

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The default build targets compute capabilities 7.5, 8.0, 8.6, 8.9, and 9.0, with PTX for the highest target. To select your GPU architecture or use another toolkit:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j
```

Use an architecture supported by both your GPU and toolkit. Building does not require an attached GPU; running the converter does.

## Run

```sh
# Same settings as main.py, including bloom.
./build/ascii_cuda images/sample_resize.png output/cuda_final_with_bloom.png

# Save the stages and disable bloom.
./build/ascii_cuda images/sample_resize.png output/cuda_final.png \
  --no-bloom \
  --edges-output output/cuda_edges.png \
  --fill-output output/cuda_fill.png

# JPEG input and custom settings.
./build/ascii_cuda input.jpg output/custom.png \
  --sigma 1.5 --edge-votes 8 --bloom-sigma 20

./build/ascii_cuda --help
```

Input can be PNG or RGB/grayscale JPEG, including grayscale PNG and PNG with alpha. The decoder converts to RGB8 and ignores alpha, matching the original converter's use of RGB channels. Output is an 8-bit grayscale PNG, regardless of the output filename extension. Output directories must already exist. Run from the repository root, or provide `--fill-atlas` and `--edge-atlas` paths explicitly.

Output dimensions are `floor(input_width / glyph_size) * glyph_size` by `floor(input_height / glyph_size) * glyph_size`. Incomplete cells at the right and bottom are cropped; an input smaller than one glyph is rejected. For example, the 812×812 sample produces an 808×808 image with the included 8×8 glyphs.

| Option | Default | Effect |
| --- | --- | --- |
| `--fill-atlas PATH` | `res/fillASCII.png` | Horizontal strip of fill glyphs, ordered by brightness |
| `--edge-atlas PATH` | `res/edgesASCII.png` | Edge glyph strip in the existing atlas order |
| `--sigma VALUE` | `2` | Narrow Gaussian standard deviation |
| `--scale VALUE` | `1.6` | Wide Gaussian sigma multiplier |
| `--tau VALUE` | `1` | Weight in `(1 + tau) * narrow - tau * wide` |
| `--dog-threshold VALUE` | `0.3` | Binarization threshold before Sobel |
| `--magnitude-threshold VALUE` | `0` | Minimum accepted Sobel magnitude |
| `--edge-votes INTEGER` | `12` | A direction needs strictly more than this many votes |
| `--bloom-threshold VALUE` | `0.8` | Bright-pass cutoff, applied before blur |
| `--bloom-sigma VALUE` | `50` | Bloom Gaussian standard deviation |
| `--bloom-method METHOD` | `auto` | `auto`, `direct`, or `fft`; single-image auto uses direct filtering |
| `--png-compression INTEGER` | `1` | Lossless level 0–9; levels 6–9 enable filter trials for smaller files |
| `--no-bloom` | Bloom enabled | Disable the bright pass, blur, and addition |
| `--fixed-range` | Per-image min/max normalization | Clamp intensities to `[0, 1]` instead |
| `--device INTEGER` | `0` | CUDA device ordinal |
| `--edges-output PATH` | Disabled | Save rendered edge glyphs |
| `--fill-output PATH` | Disabled | Save rendered fill glyphs before composition |

Gaussian sigmas, including `sigma * scale`, must be in `(0, 1024]`. Scale must be positive, tau in `[0, 1000000]`, magnitude and vote thresholds nonnegative, and bloom threshold in `[0, 1]`. Non-finite numeric values are rejected. Very wide blurs are supported but require more work.

## CUDA pipeline and compatibility

All image processing runs on the GPU: luminance conversion, both separable Gaussian blurs, DoG thresholding, Sobel direction classification, per-cell voting, luminance quantization, glyph rendering, edge/fill composition, bloom, and output normalization. The CPU handles command-line parsing, PNG/JPEG decoding and encoding, Gaussian coefficient generation, and atlas metadata. Image buffers stay on the GPU between stages; only inputs/atlases are uploaded and requested finished images are downloaded.

Gaussian kernels use warp-aligned shared-memory tiles, shared coefficients, symmetric tap pairs, and several output accumulators per thread. Default edge filters have specialized radii; wide bloom uses larger vertical tiles when shared memory permits. DoG thresholding is fused into the wide filter, and bright thresholding is fused into bloom loading. Very wide filters retain shared/global fallbacks. All paths use the original finite truncation radius `int(4 * sigma + 0.5)` and half-sample symmetric reflection, including kernels wider than the image.

Binary Sobel gradients use integer comparisons equivalent to the original angle bins. For 8×8 glyphs, four warps share a Sobel tile and vote using ballots; other glyph sizes use the generic path. Rendering reduces the normalization range in the same pass. Binary glyph strips containing zero in every glyph can encode directly without a range reduction when bloom is absent. Three full-size float buffers suffice for default settings because buffers are reused after their earlier contents are consumed.

The reusable API captures kernel execution in a CUDA graph. Automatic bloom selection uses exact FFT convolution for frames with at least 262,144 pixels and a bloom radius of at least 64; smaller filters/frames use direct convolution. FFT signals are padded with the required reflected halo to a smooth transform length, multiplied by the finite filter spectrum, and cropped. Bloom remains full resolution. FFT plan creation and extra workspace are paid once per context; single-image automatic mode uses the direct filter. `bloom_method = 1` or `2` overrides selection.
The default pipeline preserves the original luminance weights, top-left fill sampling, DoG settings, Sobel axis conventions, direction bins, strict edge vote threshold, and first-occurrence tie breaking. A nonempty edge glyph replaces its entire fill cell. Bloom is computed at the original resolution and cropped before addition. Min/max normalization and a 256-entry gray mapping reproduce the Python output-saving behavior, including mapping constant images to black. `--fixed-range` is useful when brightness should stay consistent between images.

There are a few deliberate differences and fixes:

- `main.py` computes a magnitude threshold of `0.2`, but never uses the thresholded magnitude to select edges. CUDA defaults to `0` to preserve the resulting behavior; `--magnitude-threshold 0.2` actually enables filtering.
- Glyph size comes from atlas height, and the number of fill levels comes from atlas width. Both atlases must have the same height and contain square glyphs arranged horizontally. The edge atlas must contain at least the original five glyphs: blank, direction 0, direction 1, direction 2, direction 3. Atlas red channels supply glyph intensities, as in Python.
- Fill indices are clamped to the last glyph rather than silently losing a cell on an out-of-range atlas lookup. Invalid inputs and CUDA errors are reported, and allocated buffers are released.
- GPU processing uses float32. Values close to a threshold can differ from Python's float64 calculations, so output is not guaranteed to be byte-identical. PNG is written as grayscale rather than matplotlib's RGBA output. Higher-bit-depth input is decoded to RGB8.

The executable reports CUDA-event pipeline time and total time including image I/O, setup, and transfers. Pipeline time includes processing and normalization, and excludes coefficient preparation/upload, input/atlas transfers, output download, and context setup. One-shot launches can include host scheduling gaps; graph replay reduces them. The CLI creates one context per image. Reusable contexts retain buffers and plans; the resident API also keeps inputs and outputs on the GPU.

PNG output defaults to lossless compression level 1 without per-row filter trials. Decoded pixels retain their quality, but files can be larger than with the previous writer. Use `--png-compression 6` for a smaller output. The optimization report includes encoding times and file sizes.
## Tests

Point CMake at the virtual environment to enable the Python pipeline tests alongside the CUDA reference tests:

```sh
cmake -S . -B build -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CTest checks PNG round trips, alpha handling, grayscale JPEG decoding, corrupt-input error recovery, CLI parsing, and pre-CUDA validation. CUDA tests also exercise graph replay, independent contexts, resident buffers, all attainable binary-gradient bins, and FFT/direct agreement. If a CUDA device and Python with NumPy/Pillow are available, the oracle suite compares final, edge, and fill images against an independent CPU reference. Those tests cover random images, directional edges, cropped dimensions, dark/white inputs, constant output including FFT autoscaling, both normalization modes, threshold changes, transparent PNG, JPEG, custom glyph sizes, and Gaussian kernels wider than the input. A tolerance of two intensity levels allows float32 rounding; large glyph or composition differences fail. GPU reference tests are explicitly skipped when the driver/device or optional Python packages are unavailable.

To check device memory access on a GPU machine:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/ascii_cuda images/sample_resize.png output/cuda_checked.png
```

The implementation has been built with CUDA 13.4 and validated against the CPU reference on an NVIDIA GeForce RTX 3060 Laptop GPU. All host and CUDA reference tests pass. Run CTest with access to the NVIDIA driver: a restricted environment without GPU access reports the CUDA reference suite as skipped.

## Use from C

[`cuda/ascii_cuda.h`](./cuda/ascii_cuda.h) exposes `ascii_options_default`, `ascii_convert`, and `ascii_output_free`. Pass decoded RGB8 input and both RGB8 atlases, configure an `AsciiOptions`, and choose whether edge/fill buffers should also be returned. A successful call returns the output dimensions, allocated grayscale byte buffers, device name, and pipeline time in `AsciiOutput`. On failure it returns zero and writes an error message to the supplied buffer. Release successful output with `ascii_output_free`.

The CMake targets `ascii_gpu` and `ascii_image` can be linked into another application. The command-line implementation in [`cuda/main.c`](./cuda/main.c) demonstrates input loading, conversion, output saving, and cleanup. The converter requires CUDA and reports an error when no device is available.

For repeated images of the same dimensions, call `ascii_context_create` once, `ascii_context_convert` for each RGB8 frame, then `ascii_context_destroy`. Options and atlas contents are copied at creation; source host buffers need not stay alive. Free every returned `AsciiOutput` before reusing it. Calls on one context must be serialized. Independent contexts own separate streams, coefficients, plans, and device buffers.

```c
AsciiContext *context = ascii_context_create(
    input.width, input.height, fill.rgb, fill.width, fill.height,
    edge.rgb, edge.width, edge.height, &options, 0, 0, error, sizeof(error));
/* Check context != NULL, then for each same-size RGB8 frame: */
AsciiOutput output = {0};
if (ascii_context_convert(context, frame_rgb, &output, error, sizeof(error))) {
    /* Consume output.final_pixels. */
    ascii_output_free(&output);
}
ascii_context_destroy(context);
```

For an existing GPU pipeline, `ascii_context_device_rgb` returns the context's writable RGB8 input allocation. Finish producer work before `ascii_context_run_device`, which executes and synchronizes the captured pipeline. `ascii_context_device_pixels(context, mode)` returns borrowed GPU output (`0` final, `1` edges, `2` fill), or NULL for a disabled output. The resident call returns dimensions/timing metadata with NULL host pixel pointers, avoiding CPU copies and output allocations. These device pointers remain valid until context destruction; each execution overwrites the outputs. The context uses `options.device`.

Set `AsciiOptions.profile = 1` to populate ten optional CUDA stage timings; `ascii_gpu_stage_name` supplies their names. Work fused into another stage reports zero in its former slot. Profiling is disabled by default. `device_buffer_bytes` counts the aligned arena, coefficients, and explicit FFT workspace, excluding CUDA driver context and opaque cuFFT plan storage. `AsciiOptions` gained a `bloom_method` field; recompile C API consumers against the current header.

[`ascii_benchmark`](./cuda/benchmark.c) measures full conversions; its optional `REUSE` and `BLOOM_METHOD` arguments select cached graph execution. [`ascii_resident_benchmark`](./cuda/resident_benchmark.c) decodes/uploads once and measures resident execution, then downloads/saves once outside the interval. Their exact invocation is documented by running them without arguments. [`benchmarks/compare_cuda.py`](./benchmarks/compare_cuda.py) records both paths alongside the baseline; [OPTIMIZATION.md](./OPTIMIZATION.md) gives reproduction commands.
## This project uses following methods for implementation

- Image downsampling and upsampling.
- Image quantization.
- Sobel filter for edge detection using atan2 to find angles and mapping it to corresponding slash character.
- Difference of gaussian to enhance edge detection that acts as a preprocessor to sobel filter to reduce noise.
- Image thresholding and gaussian blur to achieve bloom effect.

## Output

### Processing stages

![pipeline](./images/readme_image.png)

### Optional bloom

![bloom](./images/bloom_readme.png)

## References

- [Acerola's video](https://youtu.be/gg40RWiaHRY?si=Ht_5jxvlYIw1IgLJ)
- [Paper on difference of gaussian](https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf)
