#include "ascii_cuda.h"
#include "image_io.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(FILE *file) {
    fprintf(file,
        "Usage: ascii_cuda INPUT OUTPUT.png [options]\n"
        "\nPNG/JPEG input, grayscale PNG output. Incomplete glyph cells are cropped.\n"
        "Run from the repository root or pass explicit atlas paths.\n\n"
        "  --fill-atlas PATH          Fill glyph strip (default res/fillASCII.png)\n"
        "  --edge-atlas PATH          Edge glyph strip (default res/edgesASCII.png)\n"
        "  --edges-output PATH       Also save the rendered edge image\n"
        "  --fill-output PATH        Also save the fill image before edges/bloom\n"
        "  --no-bloom                Disable bloom (enabled by default)\n"
        "  --fixed-range             Clamp to [0,1] instead of min/max normalization\n"
        "  --sigma VALUE             Narrow Gaussian sigma (default 2)\n"
        "  --scale VALUE             Wide sigma multiplier (default 1.6)\n"
        "  --tau VALUE               DoG weight (default 1)\n"
        "  --dog-threshold VALUE     DoG binarization threshold (default 0.3)\n"
        "  --magnitude-threshold V   Suppress weak Sobel gradients (default 0)\n"
        "  --edge-votes INTEGER      Require strictly more votes (default 12)\n"
        "  --bloom-threshold VALUE   Bright-pass threshold (default 0.8)\n"
        "  --bloom-sigma VALUE       Bloom Gaussian sigma (default 50)\n"
        "  --device INTEGER          CUDA device ordinal (default 0)\n"
        "  --help                    Show this help\n");
}

static int parse_float(const char *text, float *value) {
    char *end;
    errno = 0;
    *value = strtof(text, &end);
    return text[0] && end != text && !*end && !errno && isfinite(*value);
}

static int parse_int(const char *text, int *value) {
    char *end;
    errno = 0;
    long number = strtol(text, &end, 10);
    if (!text[0] || end == text || *end || errno || number < 0 || number > INT_MAX) return 0;
    *value = (int)number;
    return 1;
}

int main(int argc, char **argv) {
    AsciiOptions options;
    ascii_options_default(&options);
    const char *input_path = NULL, *output_path = NULL;
    const char *fill_path = "res/fillASCII.png", *edge_path = "res/edgesASCII.png";
    const char *edges_output = NULL, *fill_output = NULL;
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (!strcmp(key, "--help")) { usage(stdout); return 0; }
        if (!strcmp(key, "--no-bloom")) { options.bloom = 0; continue; }
        if (!strcmp(key, "--fixed-range")) { options.normalize = 0; continue; }
        if (key[0] != '-') {
            if (!input_path) input_path = key;
            else if (!output_path) output_path = key;
            else { fprintf(stderr, "Unexpected argument: %s\n", key); return 2; }
            continue;
        }
        if (++i == argc) { fprintf(stderr, "Missing value for %s\n", key); return 2; }
        const char *value = argv[i];
        int valid = 1;
        if (!strcmp(key, "--fill-atlas")) fill_path = value;
        else if (!strcmp(key, "--edge-atlas")) edge_path = value;
        else if (!strcmp(key, "--edges-output")) edges_output = value;
        else if (!strcmp(key, "--fill-output")) fill_output = value;
        else if (!strcmp(key, "--sigma")) valid = parse_float(value, &options.sigma);
        else if (!strcmp(key, "--scale")) valid = parse_float(value, &options.scale);
        else if (!strcmp(key, "--tau")) valid = parse_float(value, &options.tau);
        else if (!strcmp(key, "--dog-threshold")) valid = parse_float(value, &options.dog_threshold);
        else if (!strcmp(key, "--magnitude-threshold")) valid = parse_float(value, &options.magnitude_threshold);
        else if (!strcmp(key, "--edge-votes")) valid = parse_int(value, &options.edge_votes);
        else if (!strcmp(key, "--bloom-threshold")) valid = parse_float(value, &options.bloom_threshold);
        else if (!strcmp(key, "--bloom-sigma")) valid = parse_float(value, &options.bloom_sigma);
        else if (!strcmp(key, "--device")) valid = parse_int(value, &options.device);
        else { fprintf(stderr, "Unknown option: %s\n", key); return 2; }
        if (!valid) { fprintf(stderr, "Invalid value for %s: %s\n", key, value); return 2; }
    }
    if (!input_path || !output_path) { usage(stderr); return 2; }
    if (!strcmp(input_path, output_path) || !strcmp(fill_path, output_path) ||
        !strcmp(edge_path, output_path) ||
        (edges_output && (!strcmp(edges_output, input_path) || !strcmp(edges_output, fill_path) ||
                          !strcmp(edges_output, edge_path) || !strcmp(edges_output, output_path))) ||
        (fill_output && (!strcmp(fill_output, input_path) || !strcmp(fill_output, fill_path) ||
                         !strcmp(fill_output, edge_path) || !strcmp(fill_output, output_path) ||
                         (edges_output && !strcmp(fill_output, edges_output))))) {
        fprintf(stderr, "Input, atlas, and output paths must not overlap\n"); return 2;
    }
    Image input = {0}, fill = {0}, edge = {0};
    AsciiOutput output = {0};
    char error[512];
    int result = 1;
    struct timespec start, stop;
    timespec_get(&start, TIME_UTC);
    if (!image_load(input_path, &input, error, sizeof(error)) ||
        !image_load(fill_path, &fill, error, sizeof(error)) ||
        !image_load(edge_path, &edge, error, sizeof(error))) {
        fprintf(stderr, "Image load failed: %s\n", error); goto cleanup;
    }
    if (!ascii_convert(input.rgb, input.width, input.height,
                       fill.rgb, fill.width, fill.height, edge.rgb, edge.width, edge.height,
                       &options, edges_output != NULL, fill_output != NULL,
                       &output, error, sizeof(error))) {
        fprintf(stderr, "CUDA conversion failed: %s\n", error); goto cleanup;
    }
    if (!image_write_gray(output_path, output.width, output.height, output.final_pixels, error, sizeof(error)) ||
        (edges_output && !image_write_gray(edges_output, output.width, output.height, output.edge_pixels, error, sizeof(error))) ||
        (fill_output && !image_write_gray(fill_output, output.width, output.height, output.fill_pixels, error, sizeof(error)))) {
        fprintf(stderr, "Image write failed: %s\n", error); goto cleanup;
    }
    timespec_get(&stop, TIME_UTC);
    double elapsed_ms = (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("%s: %dx%d -> %dx%d on %s\n", output_path, input.width, input.height,
           output.width, output.height, output.device_name);
    printf("GPU pipeline: %.3f ms; total including I/O and setup: %.3f ms\n", output.gpu_milliseconds, elapsed_ms);
    result = 0;
cleanup:
    ascii_output_free(&output);
    image_free(&input); image_free(&fill); image_free(&edge);
    return result;
}
