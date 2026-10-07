# CUDA optimization benchmark

Measured 2026-10-07T11:04:16.229185+00:00 (UTC).

This report compares the committed CUDA implementation before optimization with the optimized code in the same session. The original Python comparison remains in [BENCHMARKS.md](BENCHMARKS.md).

Optimized single-image conversions are **1.77–2.48× faster end to end** across this suite. The GPU interval and resident execution tables below show the processing improvements separately from PNG I/O.

## What changed

- Warp-aligned shared-memory Gaussian tiles process several outputs per thread, share coefficients, exploit coefficient symmetry, skip interior border arithmetic, and specialize the default short filters.
- The wide Gaussian writes binary DoG directly. Bright thresholding happens during bloom tile loading. Four warps share Sobel input and vote on four 8×8 glyph cells with ballots, preserving first-pixel tie breaking.
- Rendering also reduces its min/max range with warp shuffles. Binary atlases use a proven direct-encoding path when normalization cannot change the result.
- One aligned arena replaces per-buffer allocations. Three full-size float buffers suffice for default settings; gray becomes bloom and narrow becomes rendering storage.
- Reusable contexts cache atlases, coefficients, buffers, streams, and captured CUDA graphs. Large reusable frames use cuFFT for exact finite convolution with reflected halos; single-image automatic mode uses direct filtering to avoid FFT plan setup.
- RGB PNG decoding avoids RGBA expansion. PNG output defaults to lossless level 1 without filter trials; levels 6–9 trade encoding time for smaller files.

## Environment and method

- CPU/OS: 12th Gen Intel(R) Core(TM) i7-12650H / Linux-7.2.6-arch2-1-x86_64-with-glibc2.44
- GPU/driver/memory: `NVIDIA GeForce RTX 3060 Laptop GPU, 615.71.09, 6144 MiB`
- CUDA: `Cuda compilation tools, release 13.4, V13.4.59`
- Baseline commit: `3f711bbcee657d37810d060543c39bc59bac3ac4`
- Both builds: Release, architecture `86`; same toolchain and atlas/input bytes.
- Host compiler: `cc (GCC) 16.2.1 20260810`; libpng/libjpeg: `1.6.58, 3.2.0`
- Desktop GPU, no fixed clocks, CPU affinity, cache flushing, or fsync. Results include power-management, thermal and background variation.

Each of the 12 cases runs 3 independent warmed trials per path, each with 2 discarded conversions and 5 measured conversions. Trial order rotates across implementations. The 3 fresh-process samples per one-shot path include launch through exit. Stage events are enabled only in separate diagnostic runs. Statistics are descriptive; P90 is interpolated, and standard deviation is the sample standard deviation.

Inputs are RGB8 Lanczos resizes of `images/sample.png`. Default 8×8 atlases, Gaussian radii, thresholds, full-resolution bloom and normalization are unchanged. All outputs are decoded before comparison. The optimized PNG compression is lossless, but its default level changes file size and encoding time; end-to-end gains include that tradeoff.

**Baseline / optimized** allocate and release device buffers on every conversion. **Graph direct / graph auto** retain a fixed-size context after warmup, but still decode input/atlases, upload RGB, download pixels, encode PNG, and free host outputs each iteration. `conversion_ms` includes the entire host API call; in graph modes it excludes context creation, which is paid during warmup. Atlas decoding remains in graph totals for a controlled comparison.

**Resident** decodes/uploads once, then measures sequential synchronous GPU graph replays with input and output already on the GPU. `execution_ms` includes launch, synchronization, event queries and host bookkeeping in the API, excluding decode, transfers, context creation and encoding. One final download/save verifies output outside the interval. These numbers describe execution throughput for the supplied frame, rather than disk-to-disk throughput or a concurrent video pipeline.

**GPU pipeline** is the CUDA event interval excluding RGB transfers. The baseline interval also contains per-blur coefficient generation/upload, frees and scheduling gaps. Optimized coefficients are prepared before this interval; one-shot launches can still have scheduling gaps, while graph replay reduces them. Separate Nsight summaries distinguish kernel time and API costs. Stage medians need not sum to pipeline medians.

## Warmed end-to-end latency

Medians in milliseconds. Speedup is committed baseline / optimized one-shot total.

| Input | Bloom | Baseline | Optimized | Speedup | Graph direct | Graph auto |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 | off | 2.502 | 1.387 | 1.80× | 1.200 | 1.225 |
| 256x256 | on | 4.154 | 1.745 | 2.38× | 1.537 | 1.594 |
| 512x512 | off | 9.653 | 4.792 | 2.01× | 4.801 | 4.869 |
| 512x512 | on | 15.473 | 6.829 | 2.27× | 6.380 | 5.721 |
| 812x812 | off | 21.808 | 11.423 | 1.91× | 10.714 | 10.606 |
| 812x812 | on | 35.176 | 14.199 | 2.48× | 14.117 | 13.162 |
| 1000x1250 | off | 38.540 | 20.285 | 1.90× | 19.807 | 19.735 |
| 1000x1250 | on | 61.043 | 24.833 | 2.46× | 24.549 | 25.040 |
| 1920x1080 | off | 56.988 | 31.687 | 1.80× | 31.835 | 32.761 |
| 1920x1080 | on | 92.242 | 40.099 | 2.30× | 40.164 | 38.961 |
| 3840x2160 | off | 199.405 | 112.808 | 1.77× | 112.301 | 116.490 |
| 3840x2160 | on | 286.623 | 142.426 | 2.01× | 142.358 | 120.613 |

## GPU interval and resident throughput

GPU medians in milliseconds; resident FPS is 1000 / median execution time. Setup covers decode/context/plan creation and upload in one resident process, excluded from replay timing (one sample). Resident replay has a different GPU duty cycle from the I/O-heavy runs, so power management can also affect these intervals.

| Input | Bloom | Baseline GPU | Optimized GPU | Graph direct GPU | Graph auto GPU | Resident GPU | Resident execution | Resident FPS | Setup ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256 | off | 0.0963 | 0.0527 | 0.0452 | 0.0469 | 0.0417 | 0.0463 | 21605.5 | 161.19 |
| 256x256 | on | 0.3062 | 0.2226 | 0.2198 | 0.2191 | 0.2150 | 0.2199 | 4547.2 | 119.90 |
| 512x512 | off | 0.2263 | 0.0972 | 0.0981 | 0.1740 | 0.0891 | 0.0940 | 10640.1 | 129.37 |
| 512x512 | on | 0.9286 | 0.5400 | 0.6142 | 0.2176 | 0.2128 | 0.2178 | 4590.5 | 139.41 |
| 812x812 | off | 0.5231 | 0.2196 | 0.2908 | 0.2855 | 0.2056 | 0.2107 | 4747.1 | 132.73 |
| 812x812 | on | 2.5497 | 1.4234 | 1.4121 | 0.8541 | 0.7997 | 0.8070 | 1239.1 | 151.11 |
| 1000x1250 | off | 0.8385 | 0.3601 | 0.4274 | 0.4273 | 0.3441 | 0.3494 | 2862.3 | 140.59 |
| 1000x1250 | on | 4.1418 | 1.9012 | 1.9678 | 1.2072 | 1.1781 | 1.1845 | 844.2 | 151.33 |
| 1920x1080 | off | 1.4509 | 0.5456 | 0.6236 | 0.6259 | 0.6287 | 0.6348 | 1575.2 | 179.99 |
| 1920x1080 | on | 7.9841 | 3.7794 | 3.7769 | 2.2829 | 2.1498 | 2.1554 | 464.0 | 191.44 |
| 3840x2160 | off | 5.6924 | 2.3100 | 2.3975 | 2.4105 | 2.3049 | 2.3116 | 432.6 | 231.73 |
| 3840x2160 | on | 34.5929 | 14.6629 | 15.0051 | 8.1714 | 7.9596 | 7.9926 | 125.1 | 265.59 |

## Conversion API, I/O, memory and PNG size

API, decode, and write medians are milliseconds. Device MiB counts the aligned image/coefficient arena and explicit FFT workspace, excluding driver context and opaque cuFFT plan storage. RSS uses Linux ru_maxrss, which can retain the launcher’s memory high-water mark from before exec. It is an upper bound, rather than isolated converter working memory; identical 4K peaks across modes do not establish equal memory use. A local exec probe confirmed this inheritance. Device allocation counters are independent of that effect.

| Input | Bloom | Path | API | Decode | Write | Device MiB | Peak RSS MiB | PNG KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256 | off | baseline | 0.298 | 1.015 | 1.172 | 1.822 | 179.250 | 3.05 |
| 256x256 | off | optimized | 0.208 | 0.907 | 0.243 | 1.073 | 189.379 | 4.71 |
| 256x256 | off | graph_direct | 0.095 | 0.866 | 0.214 | 1.073 | 189.730 | 4.71 |
| 256x256 | off | graph_auto | 0.094 | 0.854 | 0.228 | 1.073 | 189.750 | 4.71 |
| 256x256 | on | baseline | 0.615 | 0.999 | 2.461 | 2.072 | 179.332 | 7.89 |
| 256x256 | on | optimized | 0.400 | 0.919 | 0.414 | 1.074 | 189.418 | 11.41 |
| 256x256 | on | graph_direct | 0.266 | 0.891 | 0.373 | 1.074 | 189.840 | 11.41 |
| 256x256 | on | graph_auto | 0.266 | 0.872 | 0.380 | 1.074 | 189.852 | 11.41 |
| 512x512 | off | baseline | 1.037 | 4.051 | 4.664 | 7.280 | 180.852 | 9.99 |
| 512x512 | off | optimized | 0.561 | 3.323 | 0.885 | 4.281 | 189.727 | 16.13 |
| 512x512 | off | graph_direct | 0.263 | 3.284 | 0.866 | 4.281 | 190.531 | 16.13 |
| 512x512 | off | graph_auto | 0.368 | 3.566 | 0.939 | 4.281 | 190.113 | 16.13 |
| 512x512 | on | baseline | 1.812 | 3.689 | 9.707 | 8.280 | 180.902 | 28.97 |
| 512x512 | on | optimized | 1.011 | 3.900 | 1.801 | 4.282 | 191.227 | 45.69 |
| 512x512 | on | graph_direct | 0.794 | 3.839 | 1.802 | 4.282 | 190.172 | 45.69 |
| 512x512 | on | graph_auto | 0.384 | 3.635 | 1.665 | 6.173 | 198.062 | 45.69 |
| 812x812 | off | baseline | 2.292 | 9.321 | 10.209 | 18.275 | 184.637 | 21.61 |
| 812x812 | off | optimized | 0.823 | 8.508 | 1.998 | 10.756 | 191.516 | 35.74 |
| 812x812 | off | graph_direct | 0.667 | 8.150 | 1.944 | 10.756 | 191.781 | 35.74 |
| 812x812 | off | graph_auto | 0.655 | 8.057 | 1.935 | 10.756 | 191.781 | 35.74 |
| 812x812 | on | baseline | 4.565 | 9.264 | 21.430 | 20.790 | 184.363 | 65.36 |
| 812x812 | on | optimized | 2.097 | 8.467 | 3.682 | 10.757 | 191.543 | 105.25 |
| 812x812 | on | graph_direct | 1.768 | 8.455 | 3.742 | 10.757 | 191.816 | 105.25 |
| 812x812 | on | graph_auto | 1.224 | 8.364 | 3.689 | 18.532 | 200.316 | 105.10 |
| 1000x1250 | off | baseline | 3.168 | 16.846 | 18.555 | 34.688 | 189.105 | 38.48 |
| 1000x1250 | off | optimized | 1.203 | 15.348 | 3.638 | 20.392 | 193.852 | 64.24 |
| 1000x1250 | off | graph_direct | 1.005 | 15.048 | 3.654 | 20.392 | 194.211 | 64.24 |
| 1000x1250 | off | graph_auto | 1.003 | 15.108 | 3.616 | 20.392 | 194.137 | 64.24 |
| 1000x1250 | on | baseline | 6.805 | 16.680 | 37.344 | 39.457 | 189.082 | 113.32 |
| 1000x1250 | on | optimized | 2.838 | 15.235 | 6.648 | 20.394 | 193.559 | 183.87 |
| 1000x1250 | on | graph_direct | 2.559 | 15.356 | 6.765 | 20.394 | 194.277 | 183.87 |
| 1000x1250 | on | graph_auto | 1.792 | 16.106 | 6.991 | 34.174 | 202.680 | 183.82 |
| 1920x1080 | off | baseline | 4.805 | 27.124 | 25.378 | 57.537 | 195.301 | 51.24 |
| 1920x1080 | off | optimized | 1.813 | 24.709 | 5.267 | 33.807 | 198.793 | 85.83 |
| 1920x1080 | off | graph_direct | 1.479 | 25.161 | 5.188 | 33.807 | 198.797 | 85.83 |
| 1920x1080 | off | graph_auto | 1.498 | 25.852 | 5.426 | 33.807 | 199.098 | 85.83 |
| 1920x1080 | on | baseline | 11.679 | 26.674 | 54.206 | 65.447 | 195.148 | 165.45 |
| 1920x1080 | on | optimized | 5.022 | 25.187 | 10.245 | 33.809 | 198.742 | 284.80 |
| 1920x1080 | on | graph_direct | 4.669 | 25.239 | 10.151 | 33.809 | 199.125 | 284.80 |
| 1920x1080 | on | graph_auto | 3.142 | 25.124 | 10.728 | 55.835 | 207.641 | 285.06 |
| 3840x2160 | off | baseline | 16.570 | 98.515 | 84.101 | 230.047 | 242.750 | 157.25 |
| 3840x2160 | off | optimized | 5.483 | 90.017 | 17.401 | 135.125 | 220.902 | 270.64 |
| 3840x2160 | off | graph_direct | 5.180 | 89.749 | 17.380 | 135.125 | 220.965 | 270.64 |
| 3840x2160 | off | graph_auto | 5.269 | 92.577 | 17.995 | 135.125 | 221.211 | 270.64 |
| 3840x2160 | on | baseline | 46.521 | 87.114 | 152.976 | 261.687 | 332.508 | 524.93 |
| 3840x2160 | on | optimized | 17.894 | 88.548 | 35.869 | 135.127 | 332.508 | 958.08 |
| 3840x2160 | on | graph_direct | 17.839 | 88.273 | 35.717 | 135.127 | 332.508 | 958.08 |
| 3840x2160 | on | graph_auto | 10.964 | 78.897 | 30.726 | 210.200 | 332.508 | 958.44 |

## Fresh process latency

One-shot process medians include driver startup and executable shutdown. Cached filesystem pages are not flushed.

| Input | Bloom | Baseline ms | Optimized ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 256x256 | off | 204.858 | 202.797 | 1.01× |
| 256x256 | on | 198.588 | 206.108 | 0.96× |
| 512x512 | off | 219.151 | 222.571 | 0.98× |
| 512x512 | on | 229.429 | 227.483 | 1.01× |
| 812x812 | off | 251.630 | 245.429 | 1.03× |
| 812x812 | on | 244.730 | 238.037 | 1.03× |
| 1000x1250 | off | 238.677 | 223.191 | 1.07× |
| 1000x1250 | on | 288.624 | 250.081 | 1.15× |
| 1920x1080 | off | 265.492 | 252.877 | 1.05× |
| 1920x1080 | on | 311.432 | 261.299 | 1.19× |
| 3840x2160 | off | 417.454 | 356.733 | 1.17× |
| 3840x2160 | on | 511.279 | 362.339 | 1.41× |

## Stage profiles

Separately measured with stage events enabled. Fused stages report zero in their former slot; their work belongs to the containing stage. Sobel includes cell voting for the default glyphs.

| Input | Bloom | Path | Stage | Median ms |
| --- | --- | --- | --- | ---: |
| 1920x1080 | off | baseline | luminance | 0.07168 |
| 1920x1080 | off | baseline | gaussian_narrow | 0.36352 |
| 1920x1080 | off | baseline | gaussian_wide | 0.46899 |
| 1920x1080 | off | baseline | dog_threshold | 0.09728 |
| 1920x1080 | off | baseline | sobel_directions | 0.16179 |
| 1920x1080 | off | baseline | cell_voting_and_fill | 0.02662 |
| 1920x1080 | off | baseline | bloom_bright_pass | 0.00000 |
| 1920x1080 | off | baseline | bloom_blur | 0.00000 |
| 1920x1080 | off | baseline | render_and_combine | 0.08806 |
| 1920x1080 | off | baseline | normalize_and_encode | 0.12800 |
| 1920x1080 | off | optimized | luminance | 0.06394 |
| 1920x1080 | off | optimized | gaussian_narrow | 0.14848 |
| 1920x1080 | off | optimized | gaussian_wide_and_dog | 0.18739 |
| 1920x1080 | off | optimized | dog_threshold_fused | 0.00000 |
| 1920x1080 | off | optimized | sobel_and_cell_voting | 0.10138 |
| 1920x1080 | off | optimized | generic_cell_voting_and_fill | 0.00000 |
| 1920x1080 | off | optimized | bright_pass_fused | 0.00000 |
| 1920x1080 | off | optimized | bloom_blur | 0.00000 |
| 1920x1080 | off | optimized | render_and_combine | 0.08909 |
| 1920x1080 | off | optimized | normalize_and_encode | 0.00102 |
| 1920x1080 | off | graph_direct | luminance | 0.15379 |
| 1920x1080 | off | graph_direct | gaussian_narrow | 0.14950 |
| 1920x1080 | off | graph_direct | gaussian_wide_and_dog | 0.18637 |
| 1920x1080 | off | graph_direct | dog_threshold_fused | 0.00000 |
| 1920x1080 | off | graph_direct | sobel_and_cell_voting | 0.10054 |
| 1920x1080 | off | graph_direct | generic_cell_voting_and_fill | 0.00000 |
| 1920x1080 | off | graph_direct | bright_pass_fused | 0.00000 |
| 1920x1080 | off | graph_direct | bloom_blur | 0.00000 |
| 1920x1080 | off | graph_direct | render_and_combine | 0.08909 |
| 1920x1080 | off | graph_direct | normalize_and_encode | 0.00102 |
| 1920x1080 | off | graph_auto | luminance | 0.16365 |
| 1920x1080 | off | graph_auto | gaussian_narrow | 0.15770 |
| 1920x1080 | off | graph_auto | gaussian_wide_and_dog | 0.20582 |
| 1920x1080 | off | graph_auto | dog_threshold_fused | 0.00000 |
| 1920x1080 | off | graph_auto | sobel_and_cell_voting | 0.11366 |
| 1920x1080 | off | graph_auto | generic_cell_voting_and_fill | 0.00000 |
| 1920x1080 | off | graph_auto | bright_pass_fused | 0.00000 |
| 1920x1080 | off | graph_auto | bloom_blur | 0.00000 |
| 1920x1080 | off | graph_auto | render_and_combine | 0.10138 |
| 1920x1080 | off | graph_auto | normalize_and_encode | 0.00102 |
| 1920x1080 | on | baseline | luminance | 0.07066 |
| 1920x1080 | on | baseline | gaussian_narrow | 0.36070 |
| 1920x1080 | on | baseline | gaussian_wide | 0.46490 |
| 1920x1080 | on | baseline | dog_threshold | 0.09830 |
| 1920x1080 | on | baseline | sobel_directions | 0.15974 |
| 1920x1080 | on | baseline | cell_voting_and_fill | 0.02662 |
| 1920x1080 | on | baseline | bloom_bright_pass | 0.06349 |
| 1920x1080 | on | baseline | bloom_blur | 5.74157 |
| 1920x1080 | on | baseline | render_and_combine | 0.09523 |
| 1920x1080 | on | baseline | normalize_and_encode | 0.12390 |
| 1920x1080 | on | optimized | luminance | 0.06320 |
| 1920x1080 | on | optimized | gaussian_narrow | 0.14848 |
| 1920x1080 | on | optimized | gaussian_wide_and_dog | 0.18637 |
| 1920x1080 | on | optimized | dog_threshold_fused | 0.00000 |
| 1920x1080 | on | optimized | sobel_and_cell_voting | 0.10035 |
| 1920x1080 | on | optimized | generic_cell_voting_and_fill | 0.00000 |
| 1920x1080 | on | optimized | bright_pass_fused | 0.00000 |
| 1920x1080 | on | optimized | bloom_blur | 2.76378 |
| 1920x1080 | on | optimized | render_and_combine | 0.10752 |
| 1920x1080 | on | optimized | normalize_and_encode | 0.04915 |
| 1920x1080 | on | graph_direct | luminance | 0.16371 |
| 1920x1080 | on | graph_direct | gaussian_narrow | 0.15872 |
| 1920x1080 | on | graph_direct | gaussian_wide_and_dog | 0.20582 |
| 1920x1080 | on | graph_direct | dog_threshold_fused | 0.00000 |
| 1920x1080 | on | graph_direct | sobel_and_cell_voting | 0.11366 |
| 1920x1080 | on | graph_direct | generic_cell_voting_and_fill | 0.00000 |
| 1920x1080 | on | graph_direct | bright_pass_fused | 0.00000 |
| 1920x1080 | on | graph_direct | bloom_blur | 3.17645 |
| 1920x1080 | on | graph_direct | render_and_combine | 0.11981 |
| 1920x1080 | on | graph_direct | normalize_and_encode | 0.04915 |
| 1920x1080 | on | graph_auto | luminance | 0.16448 |
| 1920x1080 | on | graph_auto | gaussian_narrow | 0.15872 |
| 1920x1080 | on | graph_auto | gaussian_wide_and_dog | 0.20685 |
| 1920x1080 | on | graph_auto | dog_threshold_fused | 0.00000 |
| 1920x1080 | on | graph_auto | sobel_and_cell_voting | 0.11469 |
| 1920x1080 | on | graph_auto | generic_cell_voting_and_fill | 0.00000 |
| 1920x1080 | on | graph_auto | bright_pass_fused | 0.00000 |
| 1920x1080 | on | graph_auto | bloom_blur | 1.50221 |
| 1920x1080 | on | graph_auto | render_and_combine | 0.12083 |
| 1920x1080 | on | graph_auto | normalize_and_encode | 0.04915 |
| 3840x2160 | off | baseline | luminance | 0.23450 |
| 3840x2160 | off | baseline | gaussian_narrow | 1.58003 |
| 3840x2160 | off | baseline | gaussian_wide | 2.07462 |
| 3840x2160 | off | baseline | dog_threshold | 0.37683 |
| 3840x2160 | off | baseline | sobel_directions | 0.69325 |
| 3840x2160 | off | baseline | cell_voting_and_fill | 0.07782 |
| 3840x2160 | off | baseline | bloom_bright_pass | 0.00000 |
| 3840x2160 | off | baseline | bloom_blur | 0.00000 |
| 3840x2160 | off | baseline | render_and_combine | 0.34918 |
| 3840x2160 | off | baseline | normalize_and_encode | 0.31642 |
| 3840x2160 | off | optimized | luminance | 0.23229 |
| 3840x2160 | off | optimized | gaussian_narrow | 0.55398 |
| 3840x2160 | off | optimized | gaussian_wide_and_dog | 0.73626 |
| 3840x2160 | off | optimized | dog_threshold_fused | 0.00000 |
| 3840x2160 | off | optimized | sobel_and_cell_voting | 0.43520 |
| 3840x2160 | off | optimized | generic_cell_voting_and_fill | 0.00000 |
| 3840x2160 | off | optimized | bright_pass_fused | 0.00000 |
| 3840x2160 | off | optimized | bloom_blur | 0.00000 |
| 3840x2160 | off | optimized | render_and_combine | 1.41005 |
| 3840x2160 | off | optimized | normalize_and_encode | 0.00205 |
| 3840x2160 | off | graph_direct | luminance | 0.33424 |
| 3840x2160 | off | graph_direct | gaussian_narrow | 0.55501 |
| 3840x2160 | off | graph_direct | gaussian_wide_and_dog | 0.72909 |
| 3840x2160 | off | graph_direct | dog_threshold_fused | 0.00000 |
| 3840x2160 | off | graph_direct | sobel_and_cell_voting | 0.43520 |
| 3840x2160 | off | graph_direct | generic_cell_voting_and_fill | 0.00000 |
| 3840x2160 | off | graph_direct | bright_pass_fused | 0.00000 |
| 3840x2160 | off | graph_direct | bloom_blur | 0.00000 |
| 3840x2160 | off | graph_direct | render_and_combine | 0.35965 |
| 3840x2160 | off | graph_direct | normalize_and_encode | 0.00205 |
| 3840x2160 | off | graph_auto | luminance | 0.33302 |
| 3840x2160 | off | graph_auto | gaussian_narrow | 0.55501 |
| 3840x2160 | off | graph_auto | gaussian_wide_and_dog | 0.72909 |
| 3840x2160 | off | graph_auto | dog_threshold_fused | 0.00000 |
| 3840x2160 | off | graph_auto | sobel_and_cell_voting | 0.43520 |
| 3840x2160 | off | graph_auto | generic_cell_voting_and_fill | 0.00000 |
| 3840x2160 | off | graph_auto | bright_pass_fused | 0.00000 |
| 3840x2160 | off | graph_auto | bloom_blur | 0.00000 |
| 3840x2160 | off | graph_auto | render_and_combine | 0.35942 |
| 3840x2160 | off | graph_auto | normalize_and_encode | 0.00102 |
| 3840x2160 | on | baseline | luminance | 0.23962 |
| 3840x2160 | on | baseline | gaussian_narrow | 1.57696 |
| 3840x2160 | on | baseline | gaussian_wide | 2.06336 |
| 3840x2160 | on | baseline | dog_threshold | 0.37683 |
| 3840x2160 | on | baseline | sobel_directions | 0.69427 |
| 3840x2160 | on | baseline | cell_voting_and_fill | 0.07680 |
| 3840x2160 | on | baseline | bloom_bright_pass | 0.25088 |
| 3840x2160 | on | baseline | bloom_blur | 27.23123 |
| 3840x2160 | on | baseline | render_and_combine | 0.37069 |
| 3840x2160 | on | baseline | normalize_and_encode | 0.30208 |
| 3840x2160 | on | optimized | luminance | 0.23274 |
| 3840x2160 | on | optimized | gaussian_narrow | 0.55501 |
| 3840x2160 | on | optimized | gaussian_wide_and_dog | 0.72909 |
| 3840x2160 | on | optimized | dog_threshold_fused | 0.00000 |
| 3840x2160 | on | optimized | sobel_and_cell_voting | 0.43520 |
| 3840x2160 | on | optimized | generic_cell_voting_and_fill | 0.00000 |
| 3840x2160 | on | optimized | bright_pass_fused | 0.00000 |
| 3840x2160 | on | optimized | bloom_blur | 12.06886 |
| 3840x2160 | on | optimized | render_and_combine | 0.38605 |
| 3840x2160 | on | optimized | normalize_and_encode | 0.16794 |
| 3840x2160 | on | graph_direct | luminance | 0.33216 |
| 3840x2160 | on | graph_direct | gaussian_narrow | 0.56525 |
| 3840x2160 | on | graph_direct | gaussian_wide_and_dog | 0.72909 |
| 3840x2160 | on | graph_direct | dog_threshold_fused | 0.00000 |
| 3840x2160 | on | graph_direct | sobel_and_cell_voting | 0.43520 |
| 3840x2160 | on | graph_direct | generic_cell_voting_and_fill | 0.00000 |
| 3840x2160 | on | graph_direct | bright_pass_fused | 0.00000 |
| 3840x2160 | on | graph_direct | bloom_blur | 12.79898 |
| 3840x2160 | on | graph_direct | render_and_combine | 0.38605 |
| 3840x2160 | on | graph_direct | normalize_and_encode | 0.16896 |
| 3840x2160 | on | graph_auto | luminance | 0.33491 |
| 3840x2160 | on | graph_auto | gaussian_narrow | 0.55910 |
| 3840x2160 | on | graph_auto | gaussian_wide_and_dog | 0.72909 |
| 3840x2160 | on | graph_auto | dog_threshold_fused | 0.00000 |
| 3840x2160 | on | graph_auto | sobel_and_cell_voting | 0.43622 |
| 3840x2160 | on | graph_auto | generic_cell_voting_and_fill | 0.00000 |
| 3840x2160 | on | graph_auto | bright_pass_fused | 0.00000 |
| 3840x2160 | on | graph_auto | bloom_blur | 5.34016 |
| 3840x2160 | on | graph_auto | render_and_combine | 0.38605 |
| 3840x2160 | on | graph_auto | normalize_and_encode | 0.16691 |

## Nsight kernel and API diagnostics

These separate 1080p captures contain one warmup and one measured conversion. Kernel totals cover both conversions; API totals also include process initialization, context/graph creation and cleanup. Kernel names are grouped below; every original row remains in the linked CSV files. These instrumentation runs do not provide headline latency.

| Bloom | Path | Kernel group | Launches | Total ms | Mean ms |
| --- | --- | --- | ---: | ---: | ---: |
| off | baseline | Gaussian horizontal | 4 | 0.8144 | 0.2036 |
| off | baseline | Gaussian vertical | 4 | 0.6690 | 0.1673 |
| off | baseline | direction_kernel | 2 | 0.2917 | 0.1459 |
| off | baseline | dog_kernel | 2 | 0.1918 | 0.0959 |
| off | baseline | render_kernel | 2 | 0.1592 | 0.0796 |
| off | baseline | range_kernel | 2 | 0.1445 | 0.0723 |
| off | baseline | luminance_kernel | 2 | 0.1132 | 0.0566 |
| off | baseline | encode_kernel | 2 | 0.0844 | 0.0422 |
| off | baseline | cells_kernel | 2 | 0.0451 | 0.0225 |
| off | baseline | range_finish_kernel | 2 | 0.0059 | 0.0030 |
| off | optimized | Gaussian horizontal | 4 | 0.4190 | 0.1048 |
| off | optimized | Gaussian vertical | 4 | 0.3394 | 0.0848 |
| off | optimized | sobel_cells_kernel | 2 | 0.2490 | 0.1245 |
| off | optimized | render_kernel | 2 | 0.2058 | 0.1029 |
| off | optimized | luminance_kernel | 2 | 0.1384 | 0.0692 |
| off | graph_auto | Gaussian horizontal | 4 | 0.4183 | 0.1046 |
| off | graph_auto | Gaussian vertical | 4 | 0.3041 | 0.0760 |
| off | graph_auto | sobel_cells_kernel | 2 | 0.2291 | 0.1146 |
| off | graph_auto | render_kernel | 2 | 0.2049 | 0.1024 |
| off | graph_auto | luminance_kernel | 2 | 0.1159 | 0.0580 |
| on | baseline | Gaussian horizontal | 6 | 8.6617 | 1.4436 |
| on | baseline | Gaussian vertical | 6 | 7.2434 | 1.2072 |
| on | baseline | direction_kernel | 2 | 0.3688 | 0.1844 |
| on | baseline | render_kernel | 2 | 0.2171 | 0.1085 |
| on | baseline | dog_kernel | 2 | 0.1933 | 0.0967 |
| on | baseline | range_kernel | 2 | 0.1827 | 0.0913 |
| on | baseline | bright_kernel | 2 | 0.1221 | 0.0611 |
| on | baseline | luminance_kernel | 2 | 0.1166 | 0.0583 |
| on | baseline | encode_kernel | 2 | 0.0834 | 0.0417 |
| on | baseline | cells_kernel | 2 | 0.0564 | 0.0282 |
| on | baseline | range_finish_kernel | 2 | 0.0074 | 0.0037 |
| on | optimized | Gaussian vertical | 6 | 4.1187 | 0.6865 |
| on | optimized | Gaussian horizontal | 6 | 3.2834 | 0.5472 |
| on | optimized | render_kernel | 2 | 0.2570 | 0.1285 |
| on | optimized | sobel_cells_kernel | 2 | 0.2298 | 0.1149 |
| on | optimized | luminance_kernel | 2 | 0.1166 | 0.0583 |
| on | optimized | encode_kernel | 2 | 0.0912 | 0.0456 |
| on | optimized | range_finish_kernel | 2 | 0.0068 | 0.0034 |
| on | graph_auto | cuFFT transforms / packing | 24 | 2.0487 | 0.0854 |
| on | graph_auto | Gaussian horizontal | 4 | 0.4175 | 0.1044 |
| on | graph_auto | fft_multiply | 4 | 0.3465 | 0.0866 |
| on | graph_auto | Gaussian vertical | 4 | 0.3030 | 0.0758 |
| on | graph_auto | render_kernel | 2 | 0.2395 | 0.1198 |
| on | graph_auto | sobel_cells_kernel | 2 | 0.2286 | 0.1143 |
| on | graph_auto | fft_vertical_pad | 2 | 0.1795 | 0.0898 |
| on | graph_auto | fft_horizontal_pad | 2 | 0.1587 | 0.0793 |
| on | graph_auto | fft_vertical_crop | 2 | 0.1310 | 0.0655 |
| on | graph_auto | fft_horizontal_crop | 2 | 0.1295 | 0.0648 |
| on | graph_auto | luminance_kernel | 2 | 0.1275 | 0.0638 |
| on | graph_auto | encode_kernel | 2 | 0.0887 | 0.0444 |
| on | graph_auto | range_finish_kernel | 4 | 0.0114 | 0.0029 |

Largest five CUDA API intervals per capture. Waiting inside a synchronization or free call includes outstanding GPU work.

| Bloom | Path | API | Calls | Total ms | Mean ms |
| --- | --- | --- | ---: | ---: | ---: |
| off | baseline | cudaFree | 38 | 4.9581 | 0.1305 |
| off | baseline | cudaMemcpy | 12 | 2.5182 | 0.2098 |
| off | baseline | cudaEventSynchronize | 2 | 0.8588 | 0.4294 |
| off | baseline | cudaMalloc | 32 | 0.8072 | 0.0252 |
| off | baseline | cudaLaunchKernel | 24 | 0.2971 | 0.0124 |
| off | optimized | cudaMemcpyAsync | 10 | 4.2092 | 0.4209 |
| off | optimized | cudaLaunchKernel | 14 | 0.3052 | 0.0218 |
| off | optimized | cudaMalloc | 2 | 0.2810 | 0.1405 |
| off | optimized | cudaFree | 2 | 0.2051 | 0.1025 |
| off | optimized | cudaStreamSynchronize | 6 | 0.1117 | 0.0186 |
| off | graph_auto | cudaGraphInstantiate | 1 | 5.2902 | 5.2902 |
| off | graph_auto | cudaMemcpyAsync | 7 | 4.1419 | 0.5917 |
| off | graph_auto | cudaLaunchKernel | 7 | 0.2029 | 0.0290 |
| off | graph_auto | cudaFree | 1 | 0.1186 | 0.1186 |
| off | graph_auto | cudaMalloc | 1 | 0.1184 | 0.1184 |
| on | baseline | cudaFree | 40 | 20.8456 | 0.5211 |
| on | baseline | cudaMemcpy | 14 | 3.9971 | 0.2855 |
| on | baseline | cudaMalloc | 36 | 0.8628 | 0.0240 |
| on | baseline | cudaEventSynchronize | 2 | 0.6989 | 0.3495 |
| on | baseline | cudaLaunchKernel | 30 | 0.3625 | 0.0121 |
| on | optimized | cudaMemcpyAsync | 10 | 10.6426 | 1.0643 |
| on | optimized | cudaLaunchKernel | 22 | 0.2273 | 0.0103 |
| on | optimized | cudaMalloc | 2 | 0.2005 | 0.1002 |
| on | optimized | cudaFree | 2 | 0.1621 | 0.0810 |
| on | optimized | cudaStreamSynchronize | 6 | 0.1105 | 0.0184 |
| on | graph_auto | cudaMemcpyAsync | 8 | 7.8068 | 0.9759 |
| on | graph_auto | cuModuleLoadData | 28 | 2.0468 | 0.0731 |
| on | graph_auto | cuMemFree_v2 | 1 | 1.0447 | 1.0447 |
| on | graph_auto | cuMemcpyHtoD_v2 | 9 | 0.7308 | 0.0812 |
| on | graph_auto | cuModuleUnload | 28 | 0.3234 | 0.0115 |

## Decoded output agreement

All optimized paths are compared against the committed CUDA baseline. Finite FFT convolution changes floating-point accumulation, and values near thresholds can differ. The independent oracle suite covers custom glyphs, edge bins, borders, constant signals, normalization, direct/FFT paths and graph replay.

| Input | Bloom | Path | Max error | MAE | Exact % | Within 2 levels % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 256x256 | off | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | off | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | off | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | off | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | on | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | on | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | on | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 256x256 | on | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | off | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | off | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | off | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | off | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | on | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | on | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | on | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 512x512 | on | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | off | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | off | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | off | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | off | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | on | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | on | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 812x812 | on | graph_auto | 1 | 0.000003 | 99.9997 | 100.0000 |
| 812x812 | on | resident | 1 | 0.000003 | 99.9997 | 100.0000 |
| 1000x1250 | off | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1000x1250 | off | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1000x1250 | off | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1000x1250 | off | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1000x1250 | on | optimized | 1 | 0.000002 | 99.9998 | 100.0000 |
| 1000x1250 | on | graph_direct | 1 | 0.000002 | 99.9998 | 100.0000 |
| 1000x1250 | on | graph_auto | 1 | 0.000008 | 99.9992 | 100.0000 |
| 1000x1250 | on | resident | 1 | 0.000008 | 99.9992 | 100.0000 |
| 1920x1080 | off | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1920x1080 | off | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1920x1080 | off | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1920x1080 | off | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 1920x1080 | on | optimized | 1 | 0.000002 | 99.9998 | 100.0000 |
| 1920x1080 | on | graph_direct | 1 | 0.000002 | 99.9998 | 100.0000 |
| 1920x1080 | on | graph_auto | 1 | 0.000013 | 99.9987 | 100.0000 |
| 1920x1080 | on | resident | 1 | 0.000013 | 99.9987 | 100.0000 |
| 3840x2160 | off | optimized | 0 | 0.000000 | 100.0000 | 100.0000 |
| 3840x2160 | off | graph_direct | 0 | 0.000000 | 100.0000 | 100.0000 |
| 3840x2160 | off | graph_auto | 0 | 0.000000 | 100.0000 | 100.0000 |
| 3840x2160 | off | resident | 0 | 0.000000 | 100.0000 | 100.0000 |
| 3840x2160 | on | optimized | 1 | 0.000003 | 99.9997 | 100.0000 |
| 3840x2160 | on | graph_direct | 1 | 0.000003 | 99.9997 | 100.0000 |
| 3840x2160 | on | graph_auto | 1 | 0.000009 | 99.9991 | 100.0000 |
| 3840x2160 | on | resident | 1 | 0.000009 | 99.9991 | 100.0000 |

## Validation

The normal multi-architecture release build and the architecture-86 benchmark build pass all six CTests. Compute Sanitizer reports zero memory errors, race hazards, and synchronization errors on direct/FFT graph and resident paths. Host image I/O passes AddressSanitizer/UndefinedBehaviorSanitizer (leak detection disabled in this environment). No-bloom benchmark outputs match exactly; bloom outputs differ by at most one gray level in this run.


## Artifacts and reproduction

- [All raw samples, commands, hashes and metadata](benchmarks/optimized_results/results.json)
- [Every metric: median, mean, min, max, standard deviation, P90](benchmarks/optimized_results/summary.csv)
- [Latency plot](benchmarks/optimized_results/latency.png) · [Publication SVG](benchmarks/optimized_results/latency.svg)
- Nsight kernel/API/transfer summaries: `benchmarks/optimized_results/nsight_*_*.csv` (separate two-conversion captures).

```sh
mkdir -p benchmarks/work/baseline-source
git archive 3f711bbcee657d37810d060543c39bc59bac3ac4 | tar -x -C benchmarks/work/baseline-source
cmake -S benchmarks/work/baseline-source -B benchmarks/work/baseline-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF
cmake --build benchmarks/work/baseline-build -j
cmake -S . -B build-opt -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-opt -j
.venv/bin/python benchmarks/compare_cuda.py
```

Use `--sizes`, `--trials`, `--repeat`, `--fresh-repeat`, `--resident-repeat`, `--device`, or `--skip-nsight` to change the run. `--report-only` regenerates tables and plots without the GPU. Input generation, hashes, exact commands and raw samples are retained. Binary profiler files and temporary PNGs stay in ignored `benchmarks/work/`.
