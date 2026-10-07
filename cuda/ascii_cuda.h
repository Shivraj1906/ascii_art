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
    int bloom_method; /* 0=auto (FFT for large reusable frames), 1=direct, 2=FFT. */
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
typedef struct AsciiContext AsciiContext;
/* Reuse one arena and a CUDA graph for fixed-size frames. Atlases/options are
 * copied at creation. Calls on one context must be serialized; independent
 * contexts own separate streams and buffers. Output ownership is unchanged. */
AsciiContext *ascii_context_create(int width, int height,
                  const uint8_t *fill_rgb, int fill_width, int fill_height,
                  const uint8_t *edge_rgb, int edge_width, int edge_height,
                  const AsciiOptions *options, int want_edges, int want_fill,
                  char *error, size_t error_size);
int ascii_context_convert(AsciiContext *context, const uint8_t *rgb,
                  AsciiOutput *output, char *error, size_t error_size);
void ascii_context_destroy(AsciiContext *context);
/* GPU-resident use: fill device_rgb and finish producer work before run_device.
 * Device buffers remain owned by the context. run_device synchronizes its
 * pipeline and returns timing/metadata with NULL host pixel pointers. */
uint8_t *ascii_context_device_rgb(AsciiContext *context);
const uint8_t *ascii_context_device_pixels(AsciiContext *context, int mode);
int ascii_context_run_device(AsciiContext *context, AsciiOutput *metrics,
                  char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
