#ifndef ASCII_CUDA_H
#define ASCII_CUDA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { ASCII_GPU_STAGES = 10 };

typedef struct {
    float sigma, scale, tau, dog_threshold;
    float magnitude_threshold, bloom_threshold, bloom_sigma;
    int edge_votes, bloom, normalize, device;
    int profile; /* Optional per-stage CUDA events; disabled by default. */
} AsciiOptions;

typedef struct {
    int width, height;
    uint8_t *final_pixels, *edge_pixels, *fill_pixels;
    float gpu_milliseconds;
    float gpu_stage_milliseconds[ASCII_GPU_STAGES];
    size_t device_buffer_bytes; /* Image buffers, excluding the CUDA context. */
    char device_name[256];
} AsciiOutput;

void ascii_options_default(AsciiOptions *options);
const char *ascii_gpu_stage_name(int stage);
/* Atlases are horizontal strips of square glyphs; RGB8 buffers are borrowed.
 * Output buffers are allocated by this call and released with ascii_output_free.
 * Returns 0 on failure with a human-readable error. */
int ascii_convert(const uint8_t *rgb, int width, int height,
                  const uint8_t *fill_rgb, int fill_width, int fill_height,
                  const uint8_t *edge_rgb, int edge_width, int edge_height,
                  const AsciiOptions *options, int want_edges, int want_fill,
                  AsciiOutput *output, char *error, size_t error_size);
void ascii_output_free(AsciiOutput *output);
#ifdef __cplusplus
}
#endif
#endif
