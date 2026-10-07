/* Independent conversions in one process: the context stays warm, buffers do not. */
#define _POSIX_C_SOURCE 200809L
#include "ascii_cuda.h"
#include "image_io.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000.0 + time.tv_nsec / 1000000.0;
}

static int integer(const char *text, int minimum, int maximum) {
    char *end;
    errno = 0;
    long number = strtol(text, &end, 10);
    if (!text[0] || *end || errno || number < minimum || number > maximum) return -1;
    return (int)number;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "Usage: ascii_benchmark INPUT OUTPUT.png METRICS.json WARMUP REPEAT BLOOM PROFILE DEVICE\n");
        return 2;
    }
    int warmup = integer(argv[4], 0, 10000), repeat = integer(argv[5], 1, 10000);
    int bloom = integer(argv[6], 0, 1), profile = integer(argv[7], 0, 1), device = integer(argv[8], 0, 10000);
    if (warmup < 0 || repeat < 0 || bloom < 0 || profile < 0 || device < 0) {
        fprintf(stderr, "Invalid benchmark argument\n"); return 2;
    }
    if (!strcmp(argv[1], argv[2]) || !strcmp(argv[1], argv[3]) || !strcmp(argv[2], argv[3])) {
        fprintf(stderr, "Input/output/metrics paths must differ\n"); return 2;
    }
    FILE *metrics = fopen(argv[3], "w");
    if (!metrics) { perror("metrics file"); return 1; }
    fprintf(metrics, "{\"implementation\":\"cuda\",\"warmup\":%d,\"profile\":%d,\"samples\":[\n", warmup, profile);
    AsciiOptions options;
    ascii_options_default(&options);
    options.bloom = bloom; options.profile = profile; options.device = device;
    for (int iteration = 0; iteration < warmup + repeat; ++iteration) {
        Image input = {0}, fill = {0}, edge = {0};
        AsciiOutput output = {0};
        char error[512];
        double start = milliseconds();
        if (!image_load(argv[1], &input, error, sizeof(error)) ||
            !image_load("res/fillASCII.png", &fill, error, sizeof(error)) ||
            !image_load("res/edgesASCII.png", &edge, error, sizeof(error))) goto failed;
        double loaded = milliseconds();
        if (!ascii_convert(input.rgb, input.width, input.height, fill.rgb, fill.width, fill.height,
                           edge.rgb, edge.width, edge.height, &options, 0, 0, &output, error, sizeof(error))) goto failed;
        double converted = milliseconds();
        if (!image_write_gray(argv[2], output.width, output.height, output.final_pixels, error, sizeof(error))) goto failed;
        double written = milliseconds();
        int width = output.width, height = output.height;
        float gpu_ms = output.gpu_milliseconds;
        float stage_ms[ASCII_GPU_STAGES];
        memcpy(stage_ms, output.gpu_stage_milliseconds, sizeof(stage_ms));
        size_t device_bytes = output.device_buffer_bytes;
        ascii_output_free(&output);
        image_free(&input); image_free(&fill); image_free(&edge);
        double stop = milliseconds();
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        if (iteration >= warmup) {
            if (iteration > warmup) fprintf(metrics, ",\n");
            fprintf(metrics, "{\"total_ms\":%.9f,\"load_ms\":%.9f,\"conversion_ms\":%.9f,"
                    "\"write_ms\":%.9f,\"overhead_ms\":%.9f,\"pipeline_ms\":%.9f,"
                    "\"peak_rss_mib\":%.9f,\"device_buffer_mib\":%.9f,\"width\":%d,\"height\":%d,\"stage_ms\":{",
                    stop-start, loaded-start, converted-loaded, written-converted, stop-written,
                    gpu_ms, usage.ru_maxrss / 1024.0, device_bytes / 1048576.0, width, height);
            if (profile)
                for (int stage = 0; stage < ASCII_GPU_STAGES; ++stage)
                    fprintf(metrics, "%s\"%s\":%.9f", stage ? "," : "", ascii_gpu_stage_name(stage), stage_ms[stage]);
            fprintf(metrics, "}}");
            fflush(metrics);
        }
        continue;
failed:
        fprintf(stderr, "Benchmark failed: %s\n", error);
        ascii_output_free(&output); image_free(&input); image_free(&fill); image_free(&edge);
        fclose(metrics); return 1;
    }
    fprintf(metrics, "\n]}\n");
    if (fclose(metrics)) { perror("metrics write"); return 1; }
    return 0;
}
