/* Execution-only benchmark: decode/upload once, replay, then download/save once. */
#define _POSIX_C_SOURCE 200809L
#include "ascii_cuda.h"
#include "image_io.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC,&time);
    return time.tv_sec*1000.0+time.tv_nsec/1000000.0;
}
static int integer(const char *text, int minimum, int maximum) {
    char *end;
    errno = 0;
    long value = strtol(text,&end,10);
    return !text[0] || *end || errno || value < minimum || value > maximum ? -1 : (int)value;
}
typedef struct { double wall_ms; AsciiOutput output; } Sample;
int main(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr,"Usage: ascii_resident_benchmark INPUT OUTPUT.png METRICS.json WARMUP REPEAT BLOOM METHOD PROFILE DEVICE\n"); return 2;
    }
    int warmup = integer(argv[4],0,10000), repeat = integer(argv[5],1,10000);
    int bloom = integer(argv[6],0,1), method = integer(argv[7],0,2);
    int profile = integer(argv[8],0,1), device = integer(argv[9],0,10000);
    if (warmup < 0 || repeat < 0 || bloom < 0 || method < 0 || profile < 0 || device < 0) {
        fprintf(stderr,"Invalid benchmark argument\n"); return 2;
    }
    for (int i = 1; i <= 3; ++i) {
        for (int j = i+1; j <= 3; ++j)
            if (!strcmp(argv[i],argv[j])) { fprintf(stderr,"Input/output/metrics paths must differ\n"); return 2; }
        if (i > 1 && (!strcmp(argv[i],"res/fillASCII.png") || !strcmp(argv[i],"res/edgesASCII.png"))) {
            fprintf(stderr,"Outputs must differ from atlas paths\n"); return 2;
        }
    }
    Image input = {0}, fill = {0}, edge = {0};
    AsciiContext *context = NULL;
    Sample *samples = NULL;
    uint8_t *pixels = NULL;
    FILE *file = NULL;
    char error[512] = "";
    int result = 1;
    double start = milliseconds(), ready;
    AsciiOptions options;
    ascii_options_default(&options);
    options.bloom = bloom; options.bloom_method = method; options.profile = profile; options.device = device;
    if (!image_load(argv[1],&input,error,sizeof(error)) || !image_load("res/fillASCII.png",&fill,error,sizeof(error)) ||
        !image_load("res/edgesASCII.png",&edge,error,sizeof(error))) goto failed;
    context = ascii_context_create(input.width,input.height,fill.rgb,fill.width,fill.height,
        edge.rgb,edge.width,edge.height,&options,0,0,error,sizeof(error));
    if (!context) goto failed;
    samples = calloc(repeat,sizeof(*samples));
    if (!samples) { snprintf(error,sizeof(error),"out of host memory"); goto failed; }
    cudaError_t status = cudaMemcpy(ascii_context_device_rgb(context),input.rgb,
        (size_t)input.width*input.height*3,cudaMemcpyHostToDevice);
    if (status != cudaSuccess) { snprintf(error,sizeof(error),"upload: %s",cudaGetErrorString(status)); goto failed; }
    ready = milliseconds();
    for (int i = 0; i < warmup+repeat; ++i) {
        AsciiOutput output = {0};
        double begin = milliseconds();
        if (!ascii_context_run_device(context,&output,error,sizeof(error))) goto failed;
        double stop = milliseconds();
        if (i >= warmup) { samples[i-warmup].wall_ms = stop-begin; samples[i-warmup].output = output; }
    }
    size_t count = (size_t)samples[0].output.width*samples[0].output.height;
    pixels = malloc(count);
    if (!pixels) { snprintf(error,sizeof(error),"out of host memory"); goto failed; }
    status = cudaMemcpy(pixels,ascii_context_device_pixels(context,0),count,cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { snprintf(error,sizeof(error),"download: %s",cudaGetErrorString(status)); goto failed; }
    if (!image_write_gray(argv[2],samples[0].output.width,samples[0].output.height,pixels,error,sizeof(error))) goto failed;
    file = fopen(argv[3],"w");
    if (!file) { snprintf(error,sizeof(error),"cannot open metrics"); goto failed; }
    fprintf(file,"{\"implementation\":\"cuda_resident\",\"warmup\":%d,\"profile\":%d,\"setup_ms\":%.9f,\"samples\":[\n",warmup,profile,ready-start);
    for (int i = 0; i < repeat; ++i) {
        AsciiOutput *output = &samples[i].output;
        fprintf(file,"%s{\"execution_ms\":%.9f,\"pipeline_ms\":%.9f,\"device_buffer_mib\":%.9f,\"width\":%d,\"height\":%d,\"stage_ms\":{",
            i ? ",\n" : "",samples[i].wall_ms,output->gpu_milliseconds,output->device_buffer_bytes/1048576.0,output->width,output->height);
        if (profile) for (int stage = 0; stage < ASCII_GPU_STAGES; ++stage)
            fprintf(file,"%s\"%s\":%.9f",stage ? "," : "",ascii_gpu_stage_name(stage),output->gpu_stage_milliseconds[stage]);
        fprintf(file,"}}");
    }
    fprintf(file,"\n]}\n");
    if (fclose(file)) { file = NULL; snprintf(error,sizeof(error),"metrics write failed"); goto failed; }
    file = NULL; result = 0; goto cleanup;
failed:
    fprintf(stderr,"Resident benchmark failed: %s\n",error);
cleanup:
    if (file) fclose(file);
    free(samples); free(pixels); ascii_context_destroy(context);
    image_free(&input); image_free(&fill); image_free(&edge);
    return result;
}
