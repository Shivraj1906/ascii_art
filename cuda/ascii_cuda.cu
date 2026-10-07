#include "ascii_cuda.h"
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* C-style host code and CUDA kernels; no framework or C++ container dependency. */
static const int TILE = 16;
static const int THREADS = 256;

__device__ int reflect_index(int index, int length) {
    /* SciPy's half-sample symmetric 'reflect', including kernels wider than input. */
    long long period = 2LL * length;
    long long wrapped = index % period;
    if (wrapped < 0) wrapped += period;
    return (int)(wrapped < length ? wrapped : period - wrapped - 1);
}

__global__ void luminance_kernel(const uint8_t *rgb, float *gray, size_t count) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x)
        gray[i] = (0.2989f * rgb[3*i] + 0.5870f * rgb[3*i+1] +
                   0.1140f * rgb[3*i+2]) / 255.0f;
}

template<bool horizontal>
__global__ void gaussian_tiled(const float *input, float *output, int width,
                              int height, const float *weights, int radius) {
    extern __shared__ float tile[];
    int shared_width = horizontal ? TILE + 2 * radius : TILE;
    int shared_height = horizontal ? TILE : TILE + 2 * radius;
    int lane = threadIdx.y * TILE + threadIdx.x;
    for (int i = lane; i < shared_width * shared_height; i += TILE * TILE) {
        int x = blockIdx.x * TILE + i % shared_width - (horizontal ? radius : 0);
        int y = blockIdx.y * TILE + i / shared_width - (horizontal ? 0 : radius);
        tile[i] = input[(size_t)reflect_index(y, height) * width + reflect_index(x, width)];
    }
    __syncthreads();
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x >= width || y >= height) return;
    int center = (threadIdx.y + (horizontal ? 0 : radius)) * shared_width +
                 threadIdx.x + (horizontal ? radius : 0);
    float sum = 0;
    for (int k = -radius; k <= radius; ++k)
        sum += weights[k + radius] * tile[center + k * (horizontal ? 1 : shared_width)];
    output[(size_t)y * width + x] = sum;
}

template<bool horizontal>
__global__ void gaussian_global(const float *input, float *output, int width,
                               int height, const float *weights, int radius) {
    size_t count = (size_t)width * height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % width), y = (int)(i / width);
        float sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            int sx = horizontal ? reflect_index(x + k, width) : x;
            int sy = horizontal ? y : reflect_index(y + k, height);
            sum += weights[k + radius] * input[(size_t)sy * width + sx];
        }
        output[i] = sum;
    }
}

static int blocks_for(size_t count) {
    size_t blocks = (count + THREADS - 1) / THREADS;
    return (int)(blocks > 4096 ? 4096 : blocks);
}

static cudaError_t gaussian(const float *input, float *scratch, float *output,
                            int width, int height, float sigma, size_t shared_limit) {
    int radius = (int)(4.0 * sigma + 0.5);
    int length = 2 * radius + 1;
    float *weights = (float *)malloc((size_t)length * sizeof(float));
    float *device_weights = NULL;
    if (!weights) return cudaErrorMemoryAllocation;
    double total = 0;
    for (int k = -radius; k <= radius; ++k)
        total += exp(-0.5 * ((double)k / sigma) * ((double)k / sigma));
    for (int k = -radius; k <= radius; ++k)
        weights[k + radius] = (float)(exp(-0.5 * ((double)k / sigma) *
                                            ((double)k / sigma)) / total);
    cudaError_t status = cudaMalloc((void **)&device_weights, (size_t)length * sizeof(float));
    if (status == cudaSuccess)
        status = cudaMemcpy(device_weights, weights, (size_t)length * sizeof(float), cudaMemcpyHostToDevice);
    free(weights);
    if (status == cudaSuccess) {
        size_t shared_bytes = (size_t)TILE * (TILE + 2 * radius) * sizeof(float);
        /* Match ndimage.gaussian_filter's axis order: vertical, then horizontal. */
        if (shared_bytes <= shared_limit) {
            dim3 block(TILE, TILE), grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
            gaussian_tiled<false><<<grid, block, shared_bytes>>>(input, scratch, width, height, device_weights, radius);
            status = cudaGetLastError();
            if (status == cudaSuccess) {
                gaussian_tiled<true><<<grid, block, shared_bytes>>>(scratch, output, width, height, device_weights, radius);
                status = cudaGetLastError();
            }
        } else {
            int blocks = blocks_for((size_t)width * height);
            gaussian_global<false><<<blocks, THREADS>>>(input, scratch, width, height, device_weights, radius);
            status = cudaGetLastError();
            if (status == cudaSuccess) {
                gaussian_global<true><<<blocks, THREADS>>>(scratch, output, width, height, device_weights, radius);
                status = cudaGetLastError();
            }
        }
    }
    /* cudaFree also waits for kernels that are using these weights. */
    cudaError_t release = cudaFree(device_weights);
    return status == cudaSuccess ? release : status;
}

__global__ void dog_kernel(const float *narrow, const float *wide, float *dog,
                          size_t count, float tau, float threshold) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x)
        dog[i] = (1.0f + tau) * narrow[i] - tau * wide[i] >= threshold ? 1.0f : 0.0f;
}

__device__ float sample(const float *image, int x, int y, int width, int height) {
    return image[(size_t)reflect_index(y, height) * width + reflect_index(x, width)];
}

__global__ void direction_kernel(const float *dog, signed char *directions,
                                int width, int height, float threshold) {
    size_t count = (size_t)width * height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % width), y = (int)(i / width);
        float gx = 0, gy = 0;
        /* Original names: gx differentiates rows, gy differentiates columns. */
        for (int k = -1; k <= 1; ++k) {
            float weight = k == 0 ? 2.0f : 1.0f;
            gx += weight * (sample(dog, x+k, y+1, width, height) - sample(dog, x+k, y-1, width, height));
            gy += weight * (sample(dog, x+1, y+k, width, height) - sample(dog, x-1, y+k, width, height));
        }
        int direction = -1;
        if ((gx != 0 || gy != 0) && hypotf(gx, gy) >= threshold) {
            float theta = atan2f(gy, gx);
            float angle = fabsf(theta) / 3.14159265358979323846f;
            if (angle <= 0.2f) { angle = 0; theta = 0; }
            if (angle < 0.05f || angle > 0.9f) direction = 1;
            else if (angle > 0.45f && angle < 0.55f) direction = 0;
            else if (angle > 0.05f && angle < 0.45f) direction = theta > 0 ? 2 : 3;
            else if (angle > 0.55f && angle < 0.9f) direction = theta > 0 ? 3 : 2;
        }
        directions[i] = (signed char)direction;
    }
}

__global__ void cells_kernel(const float *gray, const signed char *directions,
                            int *fills, signed char *edges, int width, int cell_size,
                            int columns, size_t count, int glyphs, int vote_threshold) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % columns) * cell_size, y = (int)(i / columns) * cell_size;
        int votes[4] = {0, 0, 0, 0};
        int first[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
        for (int row = 0; row < cell_size; ++row)
            for (int col = 0; col < cell_size; ++col) {
                int direction = directions[(size_t)(y + row) * width + x + col];
                if (direction >= 0) {
                    ++votes[direction];
                    first[direction] = min(first[direction], row * cell_size + col);
                }
            }
        int best = -1, best_count = 0, best_first = INT_MAX;
        for (int d = 0; d < 4; ++d)
            if (votes[d] > best_count || (votes[d] == best_count && first[d] < best_first)) {
                best = d; best_count = votes[d]; best_first = first[d];
            }
        edges[i] = (signed char)(best_count > vote_threshold ? best : -1);
        fills[i] = min((int)floorf(gray[(size_t)y * width + x] * glyphs), glyphs - 1);
    }
}

__global__ void bright_kernel(const float *gray, float *bright, size_t count, float threshold) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x)
        bright[i] = gray[i] > threshold ? gray[i] : 0;
}

__global__ void render_kernel(const int *fills, const signed char *edges,
                             const uint8_t *fill_atlas, const uint8_t *edge_atlas,
                             int fill_width, int edge_width, unsigned active_mask,
                             const float *bloom, int source_width, float *output,
                             int width, int height, int cell_size, int mode) {
    size_t count = (size_t)width * height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % width), y = (int)(i / width);
        size_t cell = (size_t)(y / cell_size) * (width / cell_size) + x / cell_size;
        int edge = edges[cell] + 1;
        int lx = x % cell_size, ly = y % cell_size;
        uint8_t pixel;
        if (mode == 1 || (mode == 0 && (active_mask & (1u << edge))))
            pixel = edge_atlas[3 * ((size_t)ly * edge_width + edge * cell_size + lx)];
        else
            pixel = fill_atlas[3 * ((size_t)ly * fill_width + fills[cell] * cell_size + lx)];
        float value = pixel / 255.0f;
        if (mode == 0 && bloom) value += bloom[(size_t)y * source_width + x];
        output[i] = value;
    }
}

__global__ void range_kernel(const float *image, size_t count, float2 *ranges) {
    __shared__ float lo[THREADS], hi[THREADS];
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        minimum = fminf(minimum, image[i]); maximum = fmaxf(maximum, image[i]);
    }
    lo[threadIdx.x] = minimum; hi[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            lo[threadIdx.x] = fminf(lo[threadIdx.x], lo[threadIdx.x + stride]);
            hi[threadIdx.x] = fmaxf(hi[threadIdx.x], hi[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ranges[blockIdx.x] = make_float2(lo[0], hi[0]);
}

__global__ void range_finish_kernel(float2 *ranges, int count) {
    __shared__ float lo[THREADS], hi[THREADS];
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    for (int i = threadIdx.x; i < count; i += THREADS) {
        minimum = fminf(minimum, ranges[i].x); maximum = fmaxf(maximum, ranges[i].y);
    }
    lo[threadIdx.x] = minimum; hi[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            lo[threadIdx.x] = fminf(lo[threadIdx.x], lo[threadIdx.x + stride]);
            hi[threadIdx.x] = fmaxf(hi[threadIdx.x], hi[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ranges[0] = make_float2(lo[0], hi[0]);
}

__global__ void encode_kernel(const float *image, uint8_t *bytes, size_t count,
                             const float2 *range, int normalize) {
    float lo = normalize ? range[0].x : 0, hi = normalize ? range[0].y : 1;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        float value = hi > lo ? (image[i] - lo) / (hi - lo) : 0;
        /* Matplotlib's gray colormap uses a 256-entry LUT. */
        bytes[i] = (uint8_t)fminf(255, fmaxf(0, floorf(value * 256)));
    }
}

extern "C" void ascii_options_default(AsciiOptions *o) {
    o->sigma = 2; o->scale = 1.6f; o->tau = 1; o->dog_threshold = 0.3f;
    o->magnitude_threshold = 0; o->edge_votes = 12;
    o->bloom_threshold = 0.8f; o->bloom_sigma = 50;
    o->bloom = 1; o->normalize = 1; o->device = 0;
    o->profile = 0;
}

extern "C" const char *ascii_gpu_stage_name(int stage) {
    static const char *names[ASCII_GPU_STAGES] = {
        "luminance", "gaussian_narrow", "gaussian_wide", "dog_threshold", "sobel_directions",
        "cell_voting_and_fill", "bloom_bright_pass", "bloom_blur", "render_and_combine", "normalize_and_encode"
    };
    return stage >= 0 && stage < ASCII_GPU_STAGES ? names[stage] : "unknown";
}

extern "C" void ascii_output_free(AsciiOutput *out) {
    free(out->final_pixels); free(out->edge_pixels); free(out->fill_pixels);
    memset(out, 0, sizeof(*out));
}

extern "C" int ascii_convert(const uint8_t *rgb, int width, int height,
                             const uint8_t *fill_rgb, int fill_width, int fill_height,
                             const uint8_t *edge_rgb, int edge_width, int edge_height,
                             const AsciiOptions *o, int want_edges, int want_fill,
                             AsciiOutput *out, char *error, size_t error_size) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
#define INVALID(message) do { snprintf(error, error_size, "%s", message); return 0; } while (0)
    if (!rgb || !fill_rgb || !edge_rgb || !o) INVALID("missing image, atlas, or options");
    if (width <= 0 || height <= 0 || width > INT_MAX - 8192 || height > INT_MAX - 8192 ||
        (size_t)width > SIZE_MAX / sizeof(float) / (size_t)height)
        INVALID("invalid or oversized image dimensions");
    if (fill_height <= 0 || fill_width < fill_height || fill_width % fill_height ||
        edge_height != fill_height || (long long)edge_width < 5LL * edge_height ||
        edge_width % edge_height || fill_height > 1024 ||
        (size_t)fill_width > SIZE_MAX / 3 / (size_t)fill_height ||
        (size_t)edge_width > SIZE_MAX / 3 / (size_t)edge_height)
        INVALID("atlases must be horizontal square glyph strips of equal height; edges need 5 glyphs");
    if (width < fill_height || height < fill_height) INVALID("input is smaller than one glyph");
    if (!isfinite(o->sigma) || !isfinite(o->scale) || o->sigma <= 0 || o->scale <= 0 ||
        o->sigma > 1024 || !isfinite(o->sigma * o->scale) || o->sigma * o->scale <= 0 || o->sigma * o->scale > 1024 ||
        !isfinite(o->bloom_sigma) || o->bloom_sigma <= 0 || o->bloom_sigma > 1024 ||
        !isfinite(o->tau) || o->tau < 0 || o->tau > 1000000 ||
        !isfinite(o->dog_threshold) || !isfinite(o->magnitude_threshold) || o->magnitude_threshold < 0 ||
        !isfinite(o->bloom_threshold) || o->bloom_threshold < 0 || o->bloom_threshold > 1 ||
        o->edge_votes < 0 || o->device < 0)
        INVALID("invalid numeric options (Gaussian sigmas must be in (0, 1024])");
#undef INVALID
    const int cell_size = fill_height, columns = width / cell_size, rows = height / cell_size;
    const size_t count = (size_t)width * height, cell_count = (size_t)columns * rows;
    const int output_width = columns * cell_size, output_height = rows * cell_size;
    const size_t output_count = (size_t)output_width * output_height;
    const int blocks = blocks_for(count), output_blocks = blocks_for(output_count);
    uint8_t *d_rgb = NULL, *d_fill = NULL, *d_edge = NULL, *d_bytes[3] = {NULL, NULL, NULL};
    float *gray = NULL, *scratch = NULL, *narrow = NULL, *wide = NULL, *dog = NULL;
    float *bloom = NULL, *rendered = NULL;
    signed char *directions = NULL, *edge_cells = NULL;
    int *fill_cells = NULL;
    float2 *ranges = NULL;
    cudaEvent_t start = NULL, stop = NULL;
    cudaEvent_t stage_events[2 * ASCII_GPU_STAGES] = {};
    int stage_active[ASCII_GPU_STAGES] = {};
    cudaDeviceProp properties;
    cudaError_t status;
    unsigned active_mask = 0;
    int success = 0;
    out->width = output_width; out->height = output_height;
#define CUDA(call) do { status = (call); if (status != cudaSuccess) { \
    snprintf(error, error_size, "%s: %s", #call, cudaGetErrorString(status)); goto cleanup; } } while (0)
#define ALLOC(pointer, bytes) CUDA(cudaMalloc((void **)&(pointer), (bytes)))
#define LAUNCH(...) do { __VA_ARGS__; CUDA(cudaGetLastError()); } while (0)
#define PROFILE_BEGIN(stage) do { if (o->profile) { stage_active[stage] = 1; \
    CUDA(cudaEventRecord(stage_events[2 * (stage)])); } } while (0)
#define PROFILE_END(stage) do { if (o->profile) CUDA(cudaEventRecord(stage_events[2 * (stage) + 1])); } while (0)
    CUDA(cudaSetDevice(o->device));
    CUDA(cudaGetDeviceProperties(&properties, o->device));
    snprintf(out->device_name, sizeof(out->device_name), "%s", properties.name);
    if ((height + TILE - 1) / TILE > properties.maxGridSize[1]) {
        snprintf(error, error_size, "image height exceeds CUDA tiled grid limit"); goto cleanup;
    }
    ALLOC(d_rgb, count * 3);
    ALLOC(d_fill, (size_t)fill_width * fill_height * 3);
    ALLOC(d_edge, (size_t)edge_width * edge_height * 3);
    ALLOC(gray, count * sizeof(float)); ALLOC(scratch, count * sizeof(float));
    ALLOC(narrow, count * sizeof(float)); ALLOC(wide, count * sizeof(float));
    ALLOC(dog, count * sizeof(float)); ALLOC(directions, count);
    ALLOC(fill_cells, cell_count * sizeof(int)); ALLOC(edge_cells, cell_count);
    ALLOC(rendered, output_count * sizeof(float));
    ALLOC(ranges, output_blocks * sizeof(float2));
    if (o->bloom) ALLOC(bloom, count * sizeof(float));
    out->final_pixels = (uint8_t *)malloc(output_count);
    if (want_edges) out->edge_pixels = (uint8_t *)malloc(output_count);
    if (want_fill) out->fill_pixels = (uint8_t *)malloc(output_count);
    if (!out->final_pixels || (want_edges && !out->edge_pixels) || (want_fill && !out->fill_pixels)) {
        snprintf(error, error_size, "out of host memory allocating output"); goto cleanup;
    }
    ALLOC(d_bytes[0], output_count);
    if (want_edges) ALLOC(d_bytes[1], output_count);
    if (want_fill) ALLOC(d_bytes[2], output_count);
    out->device_buffer_bytes = count * (24 + (o->bloom ? 4 : 0)) +
        (size_t)fill_width * fill_height * 3 + (size_t)edge_width * edge_height * 3 +
        cell_count * 5 + output_count * (5 + (want_edges ? 1 : 0) + (want_fill ? 1 : 0)) +
        output_blocks * sizeof(float2);
    CUDA(cudaMemcpy(d_rgb, rgb, count * 3, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_fill, fill_rgb, (size_t)fill_width * fill_height * 3, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_edge, edge_rgb, (size_t)edge_width * edge_height * 3, cudaMemcpyHostToDevice));
    for (int glyph = 0; glyph < 5; ++glyph)
        for (int y = 0; y < cell_size; ++y)
            for (int x = 0; x < cell_size; ++x)
                if (edge_rgb[3 * ((size_t)y * edge_width + glyph * cell_size + x)])
                    active_mask |= 1u << glyph;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    if (o->profile)
        for (int i = 0; i < 2 * ASCII_GPU_STAGES; ++i) CUDA(cudaEventCreate(&stage_events[i]));
    CUDA(cudaEventRecord(start));
    PROFILE_BEGIN(0);
    LAUNCH(luminance_kernel<<<blocks, THREADS>>>(d_rgb, gray, count));
    PROFILE_END(0);
    PROFILE_BEGIN(1);
    CUDA(gaussian(gray, scratch, narrow, width, height, o->sigma, properties.sharedMemPerBlock));
    PROFILE_END(1);
    PROFILE_BEGIN(2);
    CUDA(gaussian(gray, scratch, wide, width, height, o->sigma * o->scale, properties.sharedMemPerBlock));
    PROFILE_END(2);
    PROFILE_BEGIN(3);
    LAUNCH(dog_kernel<<<blocks, THREADS>>>(narrow, wide, dog, count, o->tau, o->dog_threshold));
    PROFILE_END(3);
    PROFILE_BEGIN(4);
    LAUNCH(direction_kernel<<<blocks, THREADS>>>(dog, directions, width, height, o->magnitude_threshold));
    PROFILE_END(4);
    PROFILE_BEGIN(5);
    LAUNCH(cells_kernel<<<blocks_for(cell_count), THREADS>>>(gray, directions, fill_cells, edge_cells,
        width, cell_size, columns, cell_count, fill_width / cell_size, o->edge_votes));
    PROFILE_END(5);
    if (o->bloom) {
        PROFILE_BEGIN(6);
        LAUNCH(bright_kernel<<<blocks, THREADS>>>(gray, narrow, count, o->bloom_threshold));
        PROFILE_END(6);
        PROFILE_BEGIN(7);
        CUDA(gaussian(narrow, scratch, bloom, width, height, o->bloom_sigma, properties.sharedMemPerBlock));
        PROFILE_END(7);
    }
    for (int mode = 0; mode < 3; ++mode) {
        if (!d_bytes[mode]) continue;
        if (mode == 0) PROFILE_BEGIN(8);
        LAUNCH(render_kernel<<<output_blocks, THREADS>>>(fill_cells, edge_cells, d_fill, d_edge,
            fill_width, edge_width, active_mask, bloom, width, rendered,
            output_width, output_height, cell_size, mode));
        if (mode == 0) { PROFILE_END(8); PROFILE_BEGIN(9); }
        if (o->normalize) {
            LAUNCH(range_kernel<<<output_blocks, THREADS>>>(rendered, output_count, ranges));
            LAUNCH(range_finish_kernel<<<1, THREADS>>>(ranges, output_blocks));
        }
        LAUNCH(encode_kernel<<<output_blocks, THREADS>>>(rendered, d_bytes[mode], output_count, ranges, o->normalize));
        if (mode == 0) PROFILE_END(9);
    }
    CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
    CUDA(cudaEventElapsedTime(&out->gpu_milliseconds, start, stop));
    for (int i = 0; i < ASCII_GPU_STAGES; ++i)
        if (stage_active[i]) CUDA(cudaEventElapsedTime(&out->gpu_stage_milliseconds[i], stage_events[2*i], stage_events[2*i+1]));
    CUDA(cudaMemcpy(out->final_pixels, d_bytes[0], output_count, cudaMemcpyDeviceToHost));
    if (want_edges) CUDA(cudaMemcpy(out->edge_pixels, d_bytes[1], output_count, cudaMemcpyDeviceToHost));
    if (want_fill) CUDA(cudaMemcpy(out->fill_pixels, d_bytes[2], output_count, cudaMemcpyDeviceToHost));
    success = 1;
cleanup:
    for (int i = 0; i < 2 * ASCII_GPU_STAGES; ++i)
        if (stage_events[i]) cudaEventDestroy(stage_events[i]);
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    cudaFree(d_rgb); cudaFree(d_fill); cudaFree(d_edge);
    for (int mode = 0; mode < 3; ++mode) cudaFree(d_bytes[mode]);
    cudaFree(gray); cudaFree(scratch); cudaFree(narrow); cudaFree(wide); cudaFree(dog);
    cudaFree(bloom); cudaFree(rendered); cudaFree(directions);
    cudaFree(edge_cells); cudaFree(fill_cells); cudaFree(ranges);
    if (!success) ascii_output_free(out);
    return success;
#undef LAUNCH
#undef PROFILE_BEGIN
#undef PROFILE_END
#undef ALLOC
#undef CUDA
}
