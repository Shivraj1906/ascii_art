#include "ascii_cuda.h"
#include <cuda_runtime.h>
#include <cufft.h>
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

/* One warp spans one row. Each thread has independent accumulators, amortizing
 * the halo and exposing instruction-level parallelism in the 401-tap bloom. */
template<bool horizontal, int fixed_radius, int steps, int block_rows, bool bright = false, bool binary = false>
__global__ void gaussian_coarsened(const float *__restrict__ input,
                                   float *__restrict__ output, int width, int height,
                                   const float *__restrict__ weights, int dynamic_radius, float bright_threshold = 0,
                                   const float *narrow = NULL, uint8_t *dog = NULL, float tau = 0, float threshold = 0) {
    extern __shared__ float tile[];
    const int radius = fixed_radius >= 0 ? fixed_radius : dynamic_radius;
    const int output_columns = horizontal ? 32 * steps : 32;
    const int output_rows = horizontal ? block_rows : block_rows * steps;
    const int shared_columns = output_columns + (horizontal ? 2 * radius : 0);
    const int shared_rows = output_rows + (horizontal ? 0 : 2 * radius);
    const int lane = threadIdx.y * 32 + threadIdx.x;
    const int base_x = blockIdx.x * output_columns;
    const int base_y = blockIdx.y * output_rows;
    float *shared_weights = tile + shared_columns * shared_rows;
    for (int k = lane; k <= radius; k += 32 * block_rows)
        shared_weights[k] = weights[radius + k];
    const bool interior = base_x >= (horizontal ? radius : 0) &&
        base_y >= (horizontal ? 0 : radius) &&
        base_x + output_columns + (horizontal ? radius : 0) <= width &&
        base_y + output_rows + (horizontal ? 0 : radius) <= height;
    for (int i = lane; i < shared_columns * shared_rows; i += 32 * block_rows) {
        int x = base_x + i % shared_columns - (horizontal ? radius : 0);
        int y = base_y + i / shared_columns - (horizontal ? 0 : radius);
        if (!interior) { x = reflect_index(x, width); y = reflect_index(y, height); }
        float value = input[(size_t)y * width + x];
        tile[i] = bright && value <= bright_threshold ? 0 : value;
    }
    __syncthreads();
    float sums[steps];
    const int center = (threadIdx.y + (horizontal ? 0 : radius)) * shared_columns +
                       threadIdx.x + (horizontal ? radius : 0);
    const int step_stride = horizontal ? 32 : block_rows * shared_columns;
#pragma unroll
    for (int step = 0; step < steps; ++step)
        sums[step] = tile[center + step * step_stride] * shared_weights[0];
    /* Symmetric coefficients halve the number of multiplies. */
#pragma unroll
    for (int k = 1; k <= radius; ++k) {
        const float weight = shared_weights[k];
        const int offset = horizontal ? k : k * shared_columns;
#pragma unroll
        for (int step = 0; step < steps; ++step) {
            const int index = center + step * step_stride;
            sums[step] = fmaf(tile[index - offset] + tile[index + offset], weight, sums[step]);
        }
    }
#pragma unroll
    for (int step = 0; step < steps; ++step) {
        const int x = base_x + threadIdx.x + (horizontal ? step * 32 : 0);
        const int y = base_y + threadIdx.y + (horizontal ? 0 : step * block_rows);
        if (x < width && y < height) {
            size_t i = (size_t)y*width+x;
            if (binary) dog[i] = (1.0f+tau)*narrow[i]-tau*sums[step] >= threshold;
            else output[i] = sums[step];
        }
    }
}

__device__ float2 block_range(float minimum, float maximum);

__global__ void luminance_kernel(const uint8_t *rgb, float *gray, size_t count,
                                float2 *bright_ranges, float threshold) {
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        float value = (0.2989f * rgb[3*i] + 0.5870f * rgb[3*i+1] +
                   0.1140f * rgb[3*i+2]) / 255.0f;
        gray[i] = value;
        if (bright_ranges) {
            float bright = value > threshold ? value : 0;
            minimum = fminf(minimum, bright); maximum = fmaxf(maximum, bright);
        }
    }
    if (bright_ranges) {
        float2 range = block_range(minimum, maximum);
        if (!threadIdx.x) bright_ranges[blockIdx.x] = range;
    }
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

static void gaussian_weights(float *weights, float sigma, int radius) {
    double total = 0;
    for (int k = -radius; k <= radius; ++k)
        total += exp(-0.5 * ((double)k / sigma) * ((double)k / sigma));
    for (int k = -radius; k <= radius; ++k)
        weights[k + radius] = (float)(exp(-0.5 * ((double)k / sigma) *
                                            ((double)k / sigma)) / total);
}

__global__ void dog_kernel(const float *, const float *, uint8_t *, size_t, float, float);
__global__ void bright_kernel(const float *, float *, size_t, float);

template<int radius, int steps, int rows>
static cudaError_t gaussian_columns(const float *input, float *scratch, int width,
        int height, const float *weights, int dynamic_radius, cudaStream_t stream,
        float bright_threshold) {
    dim3 block(32,rows), grid((width+31)/32,(height+rows*steps-1)/(rows*steps));
    size_t shared = ((size_t)32*(rows*steps+2*dynamic_radius)+dynamic_radius+1)*sizeof(float);
    if (bright_threshold >= 0)
        gaussian_coarsened<false,radius,steps,rows,true><<<grid,block,shared,stream>>>(input,scratch,width,height,weights,dynamic_radius,bright_threshold);
    else
        gaussian_coarsened<false,radius,steps,rows><<<grid,block,shared,stream>>>(input,scratch,width,height,weights,dynamic_radius);
    return cudaGetLastError();
}

template<int radius>
static cudaError_t gaussian_rows(const float *scratch, float *output, int width,
        int height, const float *weights, int dynamic_radius, cudaStream_t stream,
        const float *narrow, uint8_t *dog, float tau, float threshold) {
    dim3 block(32,4), grid((width+255)/256,(height+3)/4);
    size_t shared = ((size_t)4*(256+2*dynamic_radius)+dynamic_radius+1)*sizeof(float);
    if (dog)
        gaussian_coarsened<true,radius,8,4,false,true><<<grid,block,shared,stream>>>(scratch,output,width,height,weights,dynamic_radius,0,narrow,dog,tau,threshold);
    else
        gaussian_coarsened<true,radius,8,4><<<grid,block,shared,stream>>>(scratch,output,width,height,weights,dynamic_radius);
    return cudaGetLastError();
}

static cudaError_t gaussian(const float *input, float *scratch, float *output,
        int width, int height, const float *weights, int radius, size_t shared_limit,
        cudaStream_t stream, float bright_threshold = -1, const float *narrow = NULL,
        uint8_t *dog = NULL, float tau = 0, float threshold = 0) {
    cudaError_t status;
    if (radius <= 256 && ((size_t)32*(32+2*radius)+radius+1)*sizeof(float) <= shared_limit) {
        size_t wide_shared = ((size_t)32*(256+2*radius)+radius+1)*sizeof(float);
        if (radius >= 64 && width >= 512 && height >= 512 && wide_shared <= shared_limit)
            status = gaussian_columns<-1,16,16>(input,scratch,width,height,weights,radius,stream,bright_threshold);
        else if (radius == 8)
            status = gaussian_columns<8,4,8>(input,scratch,width,height,weights,radius,stream,bright_threshold);
        else if (radius == 13)
            status = gaussian_columns<13,4,8>(input,scratch,width,height,weights,radius,stream,bright_threshold);
        else
            status = gaussian_columns<-1,4,8>(input,scratch,width,height,weights,radius,stream,bright_threshold);
        if (status != cudaSuccess) return status;
        if (radius == 8) return gaussian_rows<8>(scratch,output,width,height,weights,radius,stream,narrow,dog,tau,threshold);
        if (radius == 13) return gaussian_rows<13>(scratch,output,width,height,weights,radius,stream,narrow,dog,tau,threshold);
        return gaussian_rows<-1>(scratch,output,width,height,weights,radius,stream,narrow,dog,tau,threshold);
    }
    /* Arbitrary sigmas and devices with less shared memory retain exact borders. */
    if (bright_threshold >= 0) {
        bright_kernel<<<blocks_for((size_t)width*height),THREADS,0,stream>>>(input,output,(size_t)width*height,bright_threshold);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        input = output;
    }
    size_t shared_bytes = (size_t)TILE*(TILE+2*radius)*sizeof(float);
    if (shared_bytes <= min(shared_limit,(size_t)48*1024)) {
        dim3 block(TILE,TILE), grid((width+TILE-1)/TILE,(height+TILE-1)/TILE);
        gaussian_tiled<false><<<grid,block,shared_bytes,stream>>>(input,scratch,width,height,weights,radius);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        gaussian_tiled<true><<<grid,block,shared_bytes,stream>>>(scratch,output,width,height,weights,radius);
    } else {
        int blocks = blocks_for((size_t)width*height);
        gaussian_global<false><<<blocks,THREADS,0,stream>>>(input,scratch,width,height,weights,radius);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        gaussian_global<true><<<blocks,THREADS,0,stream>>>(scratch,output,width,height,weights,radius);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (dog) {
        dog_kernel<<<blocks_for((size_t)width*height),THREADS,0,stream>>>(narrow,output,dog,(size_t)width*height,tau,threshold);
        status = cudaGetLastError();
    }
    return status;
}

__global__ void dog_kernel(const float *narrow, const float *wide, uint8_t *dog,
                          size_t count, float tau, float threshold) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x)
        dog[i] = (1.0f + tau) * narrow[i] - tau * wide[i] >= threshold;
}

__device__ int sample(const uint8_t *image, int x, int y, int width, int height) {
    return image[(size_t)reflect_index(y, height) * width + reflect_index(x, width)];
}

/* Binary Sobel gradients are integers in [-4,4]. These rational boundaries
 * classify every possible pair exactly like the original atan2 bins. */
__device__ int classify_gradient(int gx, int gy, float threshold) {
    if ((!gx && !gy) || (threshold > 0 && sqrtf((float)(gx*gx + gy*gy)) < threshold)) return -1;
    if (!gx) return 0;
    if (!gy) return 1;
    int ax = abs(gx), ay = abs(gy);
    if ((gx > 0 && 3*ay <= 2*ax) || (gx < 0 && 4*ay <= ax)) return 1;
    return gx > 0 ? (gy > 0 ? 2 : 3) : (gy > 0 ? 3 : 2);
}

__global__ void direction_kernel(const uint8_t *dog, signed char *directions,
                                int width, int height, float threshold) {
    size_t count = (size_t)width * height;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % width), y = (int)(i / width);
        int gx = 0, gy = 0;
        /* Original names: gx differentiates rows, gy differentiates columns. */
        for (int k = -1; k <= 1; ++k) {
            int weight = k == 0 ? 2 : 1;
            gx += weight * (sample(dog, x+k, y+1, width, height) - sample(dog, x+k, y-1, width, height));
            gy += weight * (sample(dog, x+1, y+k, width, height) - sample(dog, x-1, y+k, width, height));
        }
        directions[i] = (signed char)classify_gradient(gx, gy, threshold);
    }
}

/* Four warps vote on four 8x8 cells, sharing a 32x8 binary tile plus halo.
 * Ballots count directions and preserve the first-pixel tie-breaking rule. */
__global__ void sobel_cells_kernel(const uint8_t *dog, const float *gray,
        int *fills, signed char *edges, int width, int height, int columns,
        int glyphs, int vote_threshold, float magnitude_threshold) {
    __shared__ uint8_t tile[10][34];
    int lane = threadIdx.x, warp = threadIdx.y;
    int base_x = blockIdx.x * 32, base_y = blockIdx.y * 8;
    for (int i = warp*32 + lane; i < 340; i += 128) {
        int x = reflect_index(base_x + i%34 - 1, width);
        int y = reflect_index(base_y + i/34 - 1, height);
        tile[i/34][i%34] = dog[(size_t)y*width+x];
    }
    __syncthreads();
    int votes[4] = {}, first[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
#pragma unroll
    for (int step = 0; step < 2; ++step) {
        int pixel = lane + step*32;
        int x = warp*8 + pixel%8 + 1, y = pixel/8 + 1;
        int gx = tile[y+1][x-1] + 2*tile[y+1][x] + tile[y+1][x+1]
               - tile[y-1][x-1] - 2*tile[y-1][x] - tile[y-1][x+1];
        int gy = tile[y-1][x+1] + 2*tile[y][x+1] + tile[y+1][x+1]
               - tile[y-1][x-1] - 2*tile[y][x-1] - tile[y+1][x-1];
        int direction = classify_gradient(gx, gy, magnitude_threshold);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            unsigned mask = __ballot_sync(0xffffffff, direction == d);
            votes[d] += __popc(mask);
            if (mask) first[d] = min(first[d], step*32 + __ffs(mask) - 1);
        }
    }
    int column = blockIdx.x*4 + warp;
    if (lane == 0 && column < columns) {
        int best = -1, best_count = 0, best_first = INT_MAX;
#pragma unroll
        for (int d = 0; d < 4; ++d)
            if (votes[d] > best_count || (votes[d] == best_count && first[d] < best_first)) {
                best = d; best_count = votes[d]; best_first = first[d];
            }
        size_t cell = (size_t)blockIdx.y*columns + column;
        edges[cell] = best_count > vote_threshold ? best : -1;
        fills[cell] = min((int)floorf(gray[(size_t)base_y*width+column*8]*glyphs), glyphs-1);
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

__device__ uint8_t encode_value(float value) {
    return (uint8_t)fminf(255, fmaxf(0, floorf(value * 256)));
}

__device__ float2 block_range(float minimum, float maximum) {
    __shared__ float2 warps[8];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int delta = 16; delta; delta /= 2) {
        minimum = fminf(minimum, __shfl_down_sync(0xffffffff, minimum, delta));
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    }
    if (!lane) warps[warp] = make_float2(minimum, maximum);
    __syncthreads();
    if (!warp) {
        minimum = lane < 8 ? warps[lane].x : FLT_MAX;
        maximum = lane < 8 ? warps[lane].y : -FLT_MAX;
        for (int delta = 16; delta; delta /= 2) {
            minimum = fminf(minimum, __shfl_down_sync(0xffffffff, minimum, delta));
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
        }
    }
    return make_float2(minimum, maximum);
}

template<bool direct>
__global__ void render_kernel(const int *fills, const signed char *edges,
                             const uint8_t *fill_atlas, const uint8_t *edge_atlas,
                             int fill_width, int edge_width, unsigned active_mask,
                             const float *bloom, int source_width, float *output,
                             int width, int height, int cell_size, int mode,
                             uint8_t *bytes, float2 *ranges) {
    size_t count = (size_t)width * height;
    float minimum = FLT_MAX, maximum = -FLT_MAX;
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
        if (direct) bytes[i] = encode_value(value);
        else {
            output[i] = value;
            minimum = fminf(minimum, value); maximum = fmaxf(maximum, value);
        }
    }
    if (!direct) {
        float2 range = block_range(minimum, maximum);
        if (!threadIdx.x) ranges[blockIdx.x] = range;
    }
}

__global__ void range_finish_kernel(float2 *ranges, int count) {
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    for (int i = threadIdx.x; i < count; i += THREADS) {
        minimum = fminf(minimum, ranges[i].x); maximum = fmaxf(maximum, ranges[i].y);
    }
    float2 range = block_range(minimum, maximum);
    if (threadIdx.x == 0) ranges[0] = range;
}

__global__ void encode_kernel(const float *image, uint8_t *bytes, size_t count,
                             const float2 *range, int normalize) {
    float lo = normalize ? range[0].x : 0, hi = normalize ? range[0].y : 1;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)gridDim.x * blockDim.x) {
        float value = hi > lo ? (image[i] - lo) / (hi - lo) : 0;
        /* Matplotlib's gray colormap uses a 256-entry LUT. */
        bytes[i] = encode_value(value);
    }
}

extern "C" void ascii_options_default(AsciiOptions *o) {
    o->sigma = 2; o->scale = 1.6f; o->tau = 1; o->dog_threshold = 0.3f;
    o->magnitude_threshold = 0; o->edge_votes = 12;
    o->bloom_threshold = 0.8f; o->bloom_sigma = 50;
    o->bloom = 1; o->normalize = 1; o->device = 0;
    o->profile = 0;
    o->bloom_method = 0;
}

extern "C" const char *ascii_gpu_stage_name(int stage) {
    static const char *names[ASCII_GPU_STAGES] = {
        "luminance", "gaussian_narrow", "gaussian_wide_and_dog", "dog_threshold_fused", "sobel_and_cell_voting",
        "generic_cell_voting_and_fill", "bright_pass_fused", "bloom_blur", "render_and_combine", "normalize_and_encode"
    };
    return stage >= 0 && stage < ASCII_GPU_STAGES ? names[stage] : "unknown";
}

extern "C" void ascii_output_free(AsciiOutput *out) {
    free(out->final_pixels); free(out->edge_pixels); free(out->fill_pixels);
    memset(out, 0, sizeof(*out));
}

struct AsciiContext {
    AsciiOptions options;
    int width, height, columns, rows, cell_size, output_width, output_height;
    int fill_width, edge_width, glyphs, want_edges, want_fill, binary_atlases, capturing;
    size_t count, cell_count, output_count, arena_bytes, shared_limit;
    int blocks, output_blocks, radii[3], offsets[3];
    unsigned active_mask;
    char device_name[256];
    void *arena;
    uint8_t *rgb, *fill, *edge, *dog, *bytes[3];
    float *gray, *scratch, *narrow, *wide, *weights;
    signed char *directions, *edge_cells;
    int *fill_cells;
    float2 *ranges;
    cudaStream_t stream;
    cudaEvent_t start, stop, stage_events[2 * ASCII_GPU_STAGES];
    cudaGraphExec_t graph;
    int fft_bloom, fft_lengths[2];
    cufftHandle fft_plans[4];
    float *fft_buffer, *fft_frequencies;
    uint8_t *fft_workspace;
    float2 *bright_ranges;
};

/* Pad each signal with the exact reflected halo and transform a smooth length
 * at least N+2R. Cropping the halo makes circular convolution equal the finite
 * spatial filter, without downsampling or a Gaussian approximation. */
__global__ void fft_vertical_pad(const float *gray, float *buffer, int width,
                                int height, int length, int radius, float threshold) {
    __shared__ float tile[32][33];
    int x = blockIdx.x*32+threadIdx.x;
    for (int step = 0; step < 32; step += 8) {
        int y = blockIdx.y*32+threadIdx.y+step;
        float value = x < width && y < height+2*radius ? gray[(size_t)reflect_index(y-radius,height)*width+x] : 0;
        tile[threadIdx.y+step][threadIdx.x] = value > threshold ? value : 0;
    }
    __syncthreads();
    int y = blockIdx.y*32+threadIdx.x;
    for (int step = 0; step < 32; step += 8) {
        int column = blockIdx.x*32+threadIdx.y+step;
        if (column < width && y < length)
            buffer[(size_t)column*(length+2)+y] = tile[threadIdx.x][threadIdx.y+step];
    }
}

__global__ void fft_vertical_crop(const float *buffer, float *output, int width, int height, int length, int radius) {
    __shared__ float tile[32][33];
    int y = blockIdx.y*32+threadIdx.x;
    for (int step = 0; step < 32; step += 8) {
        int x = blockIdx.x*32+threadIdx.y+step;
        tile[threadIdx.y+step][threadIdx.x] = x < width && y < height ? buffer[(size_t)x*(length+2)+y+radius] : 0;
    }
    __syncthreads();
    int x = blockIdx.x*32+threadIdx.x;
    for (int step = 0; step < 32; step += 8) {
        int row = blockIdx.y*32+threadIdx.y+step;
        if (x < width && row < height) output[(size_t)row*width+x] = tile[threadIdx.x][threadIdx.y+step];
    }
}

__global__ void fft_horizontal_pad(const float *input, float *buffer, int width, int height, int length, int radius) {
    size_t count = (size_t)length*height;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < count; i += (size_t)gridDim.x*blockDim.x) {
        int x = i%length, y = i/length;
        buffer[(size_t)y*(length+2)+x] = x < width+2*radius ? input[(size_t)y*width+reflect_index(x-radius,width)] : 0;
    }
}

__global__ void fft_horizontal_crop(const float *buffer, float *output, int width, int height,
                                   int length, int radius, const float2 *bright_range) {
    size_t count = (size_t)width*height;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < count; i += (size_t)gridDim.x*blockDim.x)
        /* Preserve constant signals exactly: autoscaling must not amplify FFT
         * roundoff into visible contrast on an otherwise constant image. */
        output[i] = bright_range[0].x == bright_range[0].y ? bright_range[0].x :
                    buffer[(i/width)*(length+2)+i%width+radius];
}

__global__ void fft_multiply(cufftComplex *buffer, const float *frequencies,
                           int dimension, int batch) {
    size_t count = (size_t)(dimension+1)*batch;
    for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < count; i += (size_t)gridDim.x*blockDim.x) {
        float scale = frequencies[i%(dimension+1)]/(2*dimension);
        cufftComplex value = buffer[i];
        buffer[i] = make_float2(value.x*scale, value.y*scale);
    }
}

static int fft_length(int minimum) {
    long long best = INT_MAX;
    for (long long a = 2; a < INT_MAX; a *= 2)
        for (long long b = a; b < INT_MAX; b *= 3)
            for (long long d = b; d < INT_MAX; d *= 5)
                if (d >= minimum && d < best) best = d;
    return best < INT_MAX-32 ? (int)best : 0;
}

static void filter_frequencies(float *output, const float *weights, int radius, int dimension) {
    for (int f = 0; f <= dimension; ++f) {
        double cosine = cos(3.14159265358979323846*f/dimension);
        double previous = 1, current = cosine, value = weights[radius];
        for (int k = 1; k <= radius; ++k) {
            value += 2.0*weights[radius+k]*current;
            double next = 2*cosine*current-previous;
            previous = current; current = next;
        }
        output[f] = (float)value;
    }
}

static cudaError_t fft_blur(AsciiContext *c) {
    int column_length = c->fft_lengths[0], row_length = c->fft_lengths[1], radius = c->radii[2];
    dim3 block(32,8), pad_grid((c->width+31)/32, (column_length+31)/32);
    dim3 crop_grid((c->width+31)/32, (c->height+31)/32);
    cudaStream_t stream = c->stream;
    range_finish_kernel<<<1,THREADS,0,stream>>>(c->bright_ranges,c->blocks);
    fft_vertical_pad<<<pad_grid,block,0,stream>>>(c->gray,c->fft_buffer,c->width,c->height,column_length,radius,c->options.bloom_threshold);
    if (cufftExecR2C(c->fft_plans[0],c->fft_buffer,(cufftComplex *)c->fft_buffer) != CUFFT_SUCCESS) return cudaErrorUnknown;
    fft_multiply<<<c->blocks,THREADS,0,stream>>>((cufftComplex *)c->fft_buffer,c->fft_frequencies,column_length/2,c->width);
    if (cufftExecC2R(c->fft_plans[1],(cufftComplex *)c->fft_buffer,c->fft_buffer) != CUFFT_SUCCESS) return cudaErrorUnknown;
    fft_vertical_crop<<<crop_grid,block,0,stream>>>(c->fft_buffer,c->scratch,c->width,c->height,column_length,radius);
    fft_horizontal_pad<<<c->blocks,THREADS,0,stream>>>(c->scratch,c->fft_buffer,c->width,c->height,row_length,radius);
    if (cufftExecR2C(c->fft_plans[2],c->fft_buffer,(cufftComplex *)c->fft_buffer) != CUFFT_SUCCESS) return cudaErrorUnknown;
    fft_multiply<<<c->blocks,THREADS,0,stream>>>((cufftComplex *)c->fft_buffer,c->fft_frequencies+column_length/2+1,row_length/2,c->height);
    if (cufftExecC2R(c->fft_plans[3],(cufftComplex *)c->fft_buffer,c->fft_buffer) != CUFFT_SUCCESS) return cudaErrorUnknown;
    fft_horizontal_crop<<<c->blocks,THREADS,0,stream>>>(c->fft_buffer,c->wide,c->width,c->height,row_length,radius,c->bright_ranges);
    return cudaGetLastError();
}

static int validate(int width, int height,
                    const uint8_t *fill_rgb, int fill_width, int fill_height,
                    const uint8_t *edge_rgb, int edge_width, int edge_height,
                    const AsciiOptions *o, char *error, size_t error_size) {
#define INVALID(message) do { snprintf(error, error_size, "%s", message); return 0; } while (0)
    if (!fill_rgb || !edge_rgb || !o) INVALID("missing atlas or options");
    if (width <= 0 || height <= 0 || width > INT_MAX - 8192 || height > INT_MAX - 8192 ||
        (size_t)width > SIZE_MAX / 64 / (size_t)height)
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
        o->edge_votes < 0 || o->device < 0 || o->bloom_method < 0 || o->bloom_method > 2)
        INVALID("invalid numeric options (Gaussian sigmas must be in (0, 1024])");
#undef INVALID
    return 1;
}

/* A nonempty binary glyph containing both values guarantees the global range
 * [0,1]; a completely blank image encodes to zero under either path. */
static int binary_strip(const uint8_t *rgb, int width, int cell_size, int glyphs) {
    for (int g = 0; g < glyphs; ++g) {
        int has_zero = 0;
        for (int y = 0; y < cell_size; ++y)
            for (int x = 0; x < cell_size; ++x) {
                int value = rgb[3*((size_t)y*width+g*cell_size+x)];
                if (value != 0 && value != 255) return 0;
                has_zero |= value == 0;
            }
        if (!has_zero) return 0;
    }
    return 1;
}

static cudaError_t enqueue_pipeline(AsciiContext *c) {
    const AsciiOptions *o = &c->options;
    cudaError_t status;
    cudaStream_t stream = c->stream;
#define CHECK(call) do { status = (call); if (status != cudaSuccess) return status; } while (0)
#define LAUNCH(...) do { __VA_ARGS__; CHECK(cudaGetLastError()); } while (0)
/* External event nodes preserve timing when this sequence is captured. */
#define BEGIN(stage) do { if (o->profile) CHECK(c->capturing ? cudaEventRecordWithFlags(c->stage_events[2*(stage)], stream, cudaEventRecordExternal) : cudaEventRecord(c->stage_events[2*(stage)], stream)); } while (0)
#define END(stage) do { if (o->profile) CHECK(c->capturing ? cudaEventRecordWithFlags(c->stage_events[2*(stage)+1], stream, cudaEventRecordExternal) : cudaEventRecord(c->stage_events[2*(stage)+1], stream)); } while (0)
    BEGIN(0);
    LAUNCH(luminance_kernel<<<c->blocks, THREADS, 0, stream>>>(c->rgb, c->gray, c->count,
        c->fft_bloom ? c->bright_ranges : NULL, o->bloom_threshold));
    END(0); BEGIN(1);
    CHECK(gaussian(c->gray, c->scratch, c->narrow, c->width, c->height,
                   c->weights+c->offsets[0], c->radii[0], c->shared_limit, stream));
    END(1); BEGIN(2);
    CHECK(gaussian(c->gray, c->scratch, c->wide, c->width, c->height,
                   c->weights+c->offsets[1], c->radii[1], c->shared_limit, stream, -1, c->narrow, c->dog, o->tau, o->dog_threshold));
    END(2); BEGIN(4);
    if (c->cell_size == 8) {
        dim3 block(32, 4), grid((c->columns+3)/4, c->rows);
        LAUNCH(sobel_cells_kernel<<<grid, block, 0, stream>>>(c->dog, c->gray, c->fill_cells, c->edge_cells,
            c->width, c->height, c->columns, c->glyphs, o->edge_votes, o->magnitude_threshold));
        END(4);
    } else {
        LAUNCH(direction_kernel<<<c->blocks, THREADS, 0, stream>>>(c->dog, c->directions, c->width, c->height, o->magnitude_threshold));
        END(4); BEGIN(5);
        LAUNCH(cells_kernel<<<blocks_for(c->cell_count), THREADS, 0, stream>>>(c->gray, c->directions, c->fill_cells, c->edge_cells,
            c->width, c->cell_size, c->columns, c->cell_count, c->glyphs, o->edge_votes));
        END(5);
    }
    if (o->bloom) {
        if (c->fft_bloom) {
            BEGIN(7); CHECK(fft_blur(c)); END(7);
        } else {
            BEGIN(7);
            CHECK(gaussian(c->gray, c->scratch, c->wide, c->width, c->height,
                           c->weights+c->offsets[2], c->radii[2], c->shared_limit, stream, o->bloom_threshold));
            END(7);
        }
    }
    for (int mode = 0; mode < 3; ++mode) {
        if (!c->bytes[mode]) continue;
        const float *bloom = o->bloom && mode == 0 ? c->wide : NULL;
        int direct = !o->normalize || (!bloom && c->binary_atlases);
        if (mode == 0) BEGIN(8);
        if (direct) {
            LAUNCH(render_kernel<true><<<c->output_blocks, THREADS, 0, stream>>>(c->fill_cells, c->edge_cells,
                c->fill, c->edge, c->fill_width, c->edge_width, c->active_mask, bloom, c->width,
                c->narrow, c->output_width, c->output_height, c->cell_size, mode, c->bytes[mode], c->ranges));
        } else {
            LAUNCH(render_kernel<false><<<c->output_blocks, THREADS, 0, stream>>>(c->fill_cells, c->edge_cells,
                c->fill, c->edge, c->fill_width, c->edge_width, c->active_mask, bloom, c->width,
                c->narrow, c->output_width, c->output_height, c->cell_size, mode, c->bytes[mode], c->ranges));
        }
        if (mode == 0) { END(8); BEGIN(9); }
        if (!direct) {
            LAUNCH(range_finish_kernel<<<1, THREADS, 0, stream>>>(c->ranges, c->output_blocks));
            LAUNCH(encode_kernel<<<c->output_blocks, THREADS, 0, stream>>>(c->narrow, c->bytes[mode], c->output_count, c->ranges, 1));
        }
        if (mode == 0) END(9);
    }
    return cudaSuccess;
#undef CHECK
#undef LAUNCH
#undef BEGIN
#undef END
}

extern "C" void ascii_context_destroy(AsciiContext *c) {
    if (!c) return;
    cudaSetDevice(c->options.device);
    if (c->stream) cudaStreamSynchronize(c->stream);
    if (c->graph) cudaGraphExecDestroy(c->graph);
    for (int i = 0; i < 4; ++i) if (c->fft_plans[i]) cufftDestroy(c->fft_plans[i]);
    if (c->start) cudaEventDestroy(c->start);
    if (c->stop) cudaEventDestroy(c->stop);
    for (int i = 0; i < 2*ASCII_GPU_STAGES; ++i)
        if (c->stage_events[i]) cudaEventDestroy(c->stage_events[i]);
    if (c->arena) cudaFree(c->arena);
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

static AsciiContext *context_create(int width, int height,
                  const uint8_t *fill_rgb, int fill_width, int fill_height,
                  const uint8_t *edge_rgb, int edge_width, int edge_height,
                  const AsciiOptions *o, int want_edges, int want_fill, int capture,
                  char *error, size_t error_size) {
    if (!validate(width, height, fill_rgb, fill_width, fill_height,
                  edge_rgb, edge_width, edge_height, o, error, error_size)) return NULL;
    AsciiContext *c = (AsciiContext *)calloc(1, sizeof(*c));
    if (!c) { snprintf(error, error_size, "out of memory creating CUDA context"); return NULL; }
    c->options = *o;
    c->width = width; c->height = height; c->cell_size = fill_height;
    c->columns = width/fill_height; c->rows = height/fill_height;
    c->output_width = c->columns*fill_height; c->output_height = c->rows*fill_height;
    c->fill_width = fill_width; c->edge_width = edge_width; c->glyphs = fill_width/fill_height;
    c->count = (size_t)width*height; c->cell_count = (size_t)c->columns*c->rows;
    c->output_count = (size_t)c->output_width*c->output_height;
    c->blocks = blocks_for(c->count); c->output_blocks = blocks_for(c->output_count);
    c->want_edges = !!want_edges; c->want_fill = !!want_fill;
    c->binary_atlases = binary_strip(fill_rgb, fill_width, fill_height, c->glyphs) &&
                       binary_strip(edge_rgb, edge_width, edge_height, 5);
    c->radii[0] = (int)(4.0*o->sigma+0.5);
    c->radii[1] = (int)(4.0*(o->sigma*o->scale)+0.5);
    c->radii[2] = (int)(4.0*o->bloom_sigma+0.5);
    c->offsets[1] = 2*c->radii[0]+1;
    c->offsets[2] = c->offsets[1]+2*c->radii[1]+1;
    int weight_count = c->offsets[2]+(o->bloom ? 2*c->radii[2]+1 : 0);
    float *host_weights = NULL;
    float *host_frequencies = NULL;
    size_t fft_workspace_bytes = 0;
    int allocate_wide = 1;
    cudaDeviceProp properties;
    cudaGraph_t graph = NULL;
    cudaError_t status;
#define CUDA(call) do { status = (call); if (status != cudaSuccess) { \
    snprintf(error, error_size, "%s: %s", #call, cudaGetErrorString(status)); goto failed; } } while (0)
#define FFT(call) do { cufftResult result = (call); if (result != CUFFT_SUCCESS) { \
    snprintf(error, error_size, "%s: cuFFT error %d", #call, (int)result); goto failed; } } while (0)
    CUDA(cudaSetDevice(o->device));
    CUDA(cudaGetDeviceProperties(&properties, o->device));
    snprintf(c->device_name, sizeof(c->device_name), "%s", properties.name);
    c->shared_limit = properties.sharedMemPerBlockOptin ? properties.sharedMemPerBlockOptin : properties.sharedMemPerBlock;
    allocate_wide = c->radii[1] > 256 ||
        ((size_t)32*(32+2*c->radii[1])+c->radii[1]+1)*sizeof(float) > c->shared_limit;
    c->fft_bloom = o->bloom && (o->bloom_method == 2 ||
        (o->bloom_method == 0 && capture && c->radii[2] >= 64 && c->count >= 262144));
    if (c->fft_bloom) {
        if (width > (INT_MAX-32)/2 || height > (INT_MAX-32)/2) {
            snprintf(error, error_size, "image too large for reflected FFT dimensions"); goto failed;
        }
        c->fft_lengths[0] = fft_length(height+2*c->radii[2]);
        c->fft_lengths[1] = fft_length(width+2*c->radii[2]);
        if (!c->fft_lengths[0] || !c->fft_lengths[1]) {
            snprintf(error,error_size,"FFT dimensions exceed the supported range"); goto failed;
        }
        for (int i = 0; i < 4; ++i) {
            int dimension = c->fft_lengths[i/2]/2, batch = i < 2 ? width : height;
            int n[] = {2*dimension}, real_embed[] = {2*dimension+2}, complex_embed[] = {dimension+1};
            size_t workspace = 0;
            FFT(cufftCreate(&c->fft_plans[i]));
            FFT(cufftSetAutoAllocation(c->fft_plans[i], 0));
            if (i%2 == 0)
                FFT(cufftMakePlanMany(c->fft_plans[i],1,n,real_embed,1,2*dimension+2,complex_embed,1,dimension+1,CUFFT_R2C,batch,&workspace));
            else
                FFT(cufftMakePlanMany(c->fft_plans[i],1,n,complex_embed,1,dimension+1,real_embed,1,2*dimension+2,CUFFT_C2R,batch,&workspace));
            if (workspace > fft_workspace_bytes) fft_workspace_bytes = workspace;
        }
    }
    if ((height+3)/4 > properties.maxGridSize[1] || c->rows > properties.maxGridSize[1]) {
        snprintf(error, error_size, "image height exceeds CUDA tiled grid limit"); goto failed;
    }
    /* One aligned arena avoids many allocator synchronizations. Dead Gaussian
     * arrays become bloom and rendering storage instead of staying allocated. */
#define SLOT(pointer, bytes) do { \
    c->arena_bytes = (c->arena_bytes+255)&~(size_t)255; \
    c->pointer = (decltype(c->pointer+0))(uintptr_t)c->arena_bytes; \
    c->arena_bytes += (bytes); } while (0)
    SLOT(rgb, c->count*3); SLOT(fill, (size_t)fill_width*fill_height*3);
    SLOT(edge, (size_t)edge_width*edge_height*3);
    SLOT(gray, c->count*4); SLOT(scratch, c->count*4);
    SLOT(narrow, c->count*4);
    if (allocate_wide) SLOT(wide, c->count*4);
    SLOT(dog, c->count);
    if (c->cell_size != 8) SLOT(directions, c->count);
    SLOT(fill_cells, c->cell_count*sizeof(int)); SLOT(edge_cells, c->cell_count);
    SLOT(ranges, c->output_blocks*sizeof(float2)); SLOT(weights, (size_t)weight_count*4);
    SLOT(bytes[0], c->output_count);
    if (want_edges) SLOT(bytes[1], c->output_count);
    if (want_fill) SLOT(bytes[2], c->output_count);
    if (c->fft_bloom) {
        SLOT(fft_buffer, max((size_t)width*(c->fft_lengths[0]+2),(size_t)height*(c->fft_lengths[1]+2))*sizeof(float));
        SLOT(fft_frequencies, ((size_t)c->fft_lengths[0]/2+c->fft_lengths[1]/2+2)*sizeof(float));
        SLOT(fft_workspace, fft_workspace_bytes);
        SLOT(bright_ranges, (size_t)c->blocks*sizeof(float2));
    }
#undef SLOT
    CUDA(cudaMalloc(&c->arena, c->arena_bytes));
#define FIX(pointer) c->pointer = (decltype(c->pointer+0))((uint8_t *)c->arena+(uintptr_t)c->pointer)
    FIX(rgb); FIX(fill); FIX(edge); FIX(gray); FIX(scratch); FIX(narrow); FIX(dog);
    if (allocate_wide) { FIX(wide); }
    else c->wide = c->gray; /* Fused DoG has no wide image; gray dies after voting. */
    if (c->cell_size != 8) { FIX(directions); }
    FIX(fill_cells); FIX(edge_cells); FIX(ranges); FIX(weights); FIX(bytes[0]);
    if (want_edges) { FIX(bytes[1]); }
    if (want_fill) { FIX(bytes[2]); }
    if (c->fft_bloom) { FIX(fft_buffer); FIX(fft_frequencies); FIX(fft_workspace); FIX(bright_ranges); }
#undef FIX
    CUDA(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    if (c->fft_bloom)
        for (int i = 0; i < 4; ++i) {
            FFT(cufftSetWorkArea(c->fft_plans[i],c->fft_workspace));
            FFT(cufftSetStream(c->fft_plans[i],c->stream));
        }
    host_weights = (float *)malloc((size_t)weight_count*sizeof(float));
    if (!host_weights) { snprintf(error, error_size, "out of memory preparing coefficients"); goto failed; }
    gaussian_weights(host_weights, o->sigma, c->radii[0]);
    gaussian_weights(host_weights+c->offsets[1], o->sigma*o->scale, c->radii[1]);
    if (o->bloom) gaussian_weights(host_weights+c->offsets[2], o->bloom_sigma, c->radii[2]);
    if (c->fft_bloom) {
        host_frequencies = (float *)malloc(((size_t)c->fft_lengths[0]/2+c->fft_lengths[1]/2+2)*sizeof(float));
        if (!host_frequencies) { snprintf(error,error_size,"out of memory preparing FFT filter"); goto failed; }
        filter_frequencies(host_frequencies,host_weights+c->offsets[2],c->radii[2],c->fft_lengths[0]/2);
        filter_frequencies(host_frequencies+c->fft_lengths[0]/2+1,host_weights+c->offsets[2],c->radii[2],c->fft_lengths[1]/2);
        CUDA(cudaMemcpyAsync(c->fft_frequencies,host_frequencies,((size_t)c->fft_lengths[0]/2+c->fft_lengths[1]/2+2)*sizeof(float),cudaMemcpyHostToDevice,c->stream));
    }
    CUDA(cudaMemcpyAsync(c->weights, host_weights, (size_t)weight_count*4, cudaMemcpyHostToDevice, c->stream));
    CUDA(cudaMemcpyAsync(c->fill, fill_rgb, (size_t)fill_width*fill_height*3, cudaMemcpyHostToDevice, c->stream));
    CUDA(cudaMemcpyAsync(c->edge, edge_rgb, (size_t)edge_width*edge_height*3, cudaMemcpyHostToDevice, c->stream));
    CUDA(cudaStreamSynchronize(c->stream));
    free(host_weights); host_weights = NULL;
    free(host_frequencies); host_frequencies = NULL;
    for (int g = 0; g < 5; ++g)
        for (int y = 0; y < fill_height; ++y)
            for (int x = 0; x < fill_height; ++x)
                if (edge_rgb[3*((size_t)y*edge_width+g*fill_height+x)]) c->active_mask |= 1u<<g;
    CUDA(cudaEventCreate(&c->start)); CUDA(cudaEventCreate(&c->stop));
    if (o->profile)
        for (int i = 0; i < 2*ASCII_GPU_STAGES; ++i) CUDA(cudaEventCreate(&c->stage_events[i]));
    if (c->shared_limit > properties.sharedMemPerBlock) {
        CUDA(cudaFuncSetAttribute(gaussian_coarsened<false, -1, 4, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c->shared_limit));
        CUDA(cudaFuncSetAttribute(gaussian_coarsened<false, -1, 16, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c->shared_limit));
        CUDA(cudaFuncSetAttribute(gaussian_coarsened<false, -1, 4, 8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c->shared_limit));
        CUDA(cudaFuncSetAttribute(gaussian_coarsened<false, -1, 16, 16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c->shared_limit));
    }
    if (capture) {
        CUDA(cudaStreamBeginCapture(c->stream, cudaStreamCaptureModeThreadLocal));
        c->capturing = 1;
        status = enqueue_pipeline(c);
        c->capturing = 0;
        if (status != cudaSuccess) {
            cudaStreamEndCapture(c->stream, &graph);
            snprintf(error, error_size, "capturing CUDA pipeline: %s", cudaGetErrorString(status)); goto failed;
        }
        CUDA(cudaStreamEndCapture(c->stream, &graph));
        CUDA(cudaGraphInstantiate(&c->graph, graph, 0));
        CUDA(cudaGraphDestroy(graph)); graph = NULL;
    }
    return c;
failed:
    if (c->stream) cudaStreamSynchronize(c->stream);
    free(host_weights);
    free(host_frequencies);
    if (graph) cudaGraphDestroy(graph);
    ascii_context_destroy(c);
    return NULL;
#undef CUDA
#undef FFT
}

extern "C" AsciiContext *ascii_context_create(int width, int height,
                  const uint8_t *fill_rgb, int fill_width, int fill_height,
                  const uint8_t *edge_rgb, int edge_width, int edge_height,
                  const AsciiOptions *options, int want_edges, int want_fill,
                  char *error, size_t error_size) {
    return context_create(width, height, fill_rgb, fill_width, fill_height,
        edge_rgb, edge_width, edge_height, options, want_edges, want_fill, 1, error, error_size);
}

extern "C" uint8_t *ascii_context_device_rgb(AsciiContext *c) {
    return c ? c->rgb : NULL;
}

extern "C" const uint8_t *ascii_context_device_pixels(AsciiContext *c, int mode) {
    return c && mode >= 0 && mode < 3 ? c->bytes[mode] : NULL;
}

static cudaError_t run_pipeline(AsciiContext *c) {
    cudaError_t status = cudaEventRecord(c->start,c->stream);
    if (status != cudaSuccess) return status;
    status = c->graph ? cudaGraphLaunch(c->graph,c->stream) : enqueue_pipeline(c);
    if (status != cudaSuccess) return status;
    return cudaEventRecord(c->stop,c->stream);
}

static cudaError_t read_metrics(AsciiContext *c, AsciiOutput *out) {
    out->width = c->output_width; out->height = c->output_height;
    out->device_buffer_bytes = c->arena_bytes;
    snprintf(out->device_name,sizeof(out->device_name),"%s",c->device_name);
    cudaError_t status = cudaEventElapsedTime(&out->gpu_milliseconds,c->start,c->stop);
    if (status != cudaSuccess) return status;
    if (c->options.profile)
        for (int i = 0; i < ASCII_GPU_STAGES; ++i) {
            if (i == 3 || i == 6 || (i == 5 && c->cell_size == 8) || (i == 7 && !c->options.bloom)) continue;
            status = cudaEventElapsedTime(&out->gpu_stage_milliseconds[i],c->stage_events[2*i],c->stage_events[2*i+1]);
            if (status != cudaSuccess) return status;
        }
    return cudaSuccess;
}

extern "C" int ascii_context_run_device(AsciiContext *c, AsciiOutput *out,
                                        char *error, size_t error_size) {
    if (!out) return 0;
    memset(out,0,sizeof(*out));
    if (!c) { snprintf(error,error_size,"missing context"); return 0; }
    cudaError_t status = cudaSetDevice(c->options.device);
    if (status == cudaSuccess) status = run_pipeline(c);
    if (status == cudaSuccess) status = cudaStreamSynchronize(c->stream);
    if (status == cudaSuccess) status = read_metrics(c,out);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(c->stream);
        snprintf(error,error_size,"GPU-resident conversion: %s",cudaGetErrorString(status));
        memset(out,0,sizeof(*out)); return 0;
    }
    return 1;
}

extern "C" int ascii_context_convert(AsciiContext *c, const uint8_t *rgb,
                                    AsciiOutput *out, char *error, size_t error_size) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!c || !rgb) { snprintf(error, error_size, "missing context or image"); return 0; }
    cudaError_t status;
#define CUDA(call) do { status = (call); if (status != cudaSuccess) { \
    snprintf(error, error_size, "%s: %s", #call, cudaGetErrorString(status)); goto failed; } } while (0)
    out->width = c->output_width; out->height = c->output_height;
    out->device_buffer_bytes = c->arena_bytes;
    snprintf(out->device_name, sizeof(out->device_name), "%s", c->device_name);
    out->final_pixels = (uint8_t *)malloc(c->output_count);
    if (c->want_edges) out->edge_pixels = (uint8_t *)malloc(c->output_count);
    if (c->want_fill) out->fill_pixels = (uint8_t *)malloc(c->output_count);
    if (!out->final_pixels || (c->want_edges && !out->edge_pixels) || (c->want_fill && !out->fill_pixels)) {
        snprintf(error, error_size, "out of host memory allocating output"); goto failed;
    }
    CUDA(cudaSetDevice(c->options.device));
    CUDA(cudaMemcpyAsync(c->rgb, rgb, c->count*3, cudaMemcpyHostToDevice, c->stream));
    CUDA(run_pipeline(c));
    CUDA(cudaMemcpyAsync(out->final_pixels, c->bytes[0], c->output_count, cudaMemcpyDeviceToHost, c->stream));
    if (c->want_edges) CUDA(cudaMemcpyAsync(out->edge_pixels, c->bytes[1], c->output_count, cudaMemcpyDeviceToHost, c->stream));
    if (c->want_fill) CUDA(cudaMemcpyAsync(out->fill_pixels, c->bytes[2], c->output_count, cudaMemcpyDeviceToHost, c->stream));
    CUDA(cudaStreamSynchronize(c->stream));
    CUDA(read_metrics(c,out));
    return 1;
failed:
    cudaStreamSynchronize(c->stream);
    ascii_output_free(out);
    return 0;
#undef CUDA
}

extern "C" int ascii_convert(const uint8_t *rgb, int width, int height,
                             const uint8_t *fill_rgb, int fill_width, int fill_height,
                             const uint8_t *edge_rgb, int edge_width, int edge_height,
                             const AsciiOptions *o, int want_edges, int want_fill,
                             AsciiOutput *out, char *error, size_t error_size) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!rgb) { snprintf(error, error_size, "missing image"); return 0; }
    AsciiContext *c = context_create(width, height, fill_rgb, fill_width, fill_height,
        edge_rgb, edge_width, edge_height, o, want_edges, want_fill, 0, error, error_size);
    if (!c) return 0;
    int result = ascii_context_convert(c, rgb, out, error, error_size);
    ascii_context_destroy(c);
    return result;
}
