// Include private kernels so the exhaustive check exercises production device code.
#include "../cuda/ascii_cuda.cu"
extern "C" {
#include "image_io.h"
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(test) do { if (!(test)) { fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #test, error); return 1; } } while (0)
__global__ void gradient_bins(int *bins) {
    int i = threadIdx.x;
    if (i < 81) bins[i] = classify_gradient(i/9-4,i%9-4,0);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    char error[512] = "";
    Image fill = {}, edge = {};
    REQUIRE(image_load("res/fillASCII.png", &fill, error, sizeof(error)));
    REQUIRE(image_load("res/edgesASCII.png", &edge, error, sizeof(error)));
    uint8_t rgb[67*51*3];
    for (int profile = 0; profile <= 1; ++profile)
        for (int bloom = 0; bloom <= 1; ++bloom) {
            AsciiOptions options;
            ascii_options_default(&options);
            options.profile = profile; options.bloom = bloom;
            AsciiContext *a = ascii_context_create(67, 51, fill.rgb, fill.width, fill.height,
                edge.rgb, edge.width, edge.height, &options, 1, 1, error, sizeof(error));
            REQUIRE(a);
            AsciiOptions alternate = options;
            alternate.bloom_sigma = 3; alternate.sigma = 0.7f; alternate.dog_threshold = 0.5f;
            AsciiContext *b = ascii_context_create(67, 51, fill.rgb, fill.width, fill.height,
                edge.rgb, edge.width, edge.height, &alternate, 1, 1, error, sizeof(error));
            REQUIRE(b);
            for (int frame = 0; frame < 4; ++frame)
                for (int which = 0; which < 2; ++which) {
                    for (size_t i = 0; i < sizeof(rgb); ++i) rgb[i] = (uint8_t)(i*37+frame*71);
                    AsciiOutput graph = {}, direct = {};
                    REQUIRE(ascii_context_convert(which ? b : a, rgb, &graph, error, sizeof(error)));
                    REQUIRE(ascii_convert(rgb, 67, 51, fill.rgb, fill.width, fill.height,
                        edge.rgb, edge.width, edge.height, which ? &alternate : &options, 1, 1,
                        &direct, error, sizeof(error)));
                    REQUIRE(graph.width == 64 && graph.height == 48);
                    REQUIRE(!memcmp(graph.final_pixels, direct.final_pixels, 64*48));
                    REQUIRE(!memcmp(graph.edge_pixels, direct.edge_pixels, 64*48));
                    REQUIRE(!memcmp(graph.fill_pixels, direct.fill_pixels, 64*48));
                    REQUIRE(graph.device_buffer_bytes == direct.device_buffer_bytes);
                    REQUIRE(graph.gpu_milliseconds >= 0);
                    AsciiOutput resident = {};
                    REQUIRE(ascii_context_device_rgb(which ? b : a));
                    REQUIRE(ascii_context_run_device(which ? b : a,&resident,error,sizeof(error)));
                    REQUIRE(resident.width == graph.width && resident.height == graph.height);
                    REQUIRE(!resident.final_pixels && !resident.edge_pixels && !resident.fill_pixels);
                    uint8_t pixels[64*48];
                    REQUIRE(cudaMemcpy(pixels,ascii_context_device_pixels(which ? b : a,0),sizeof(pixels),cudaMemcpyDeviceToHost) == cudaSuccess);
                    REQUIRE(!memcmp(pixels,graph.final_pixels,sizeof(pixels)));
                    REQUIRE(!ascii_context_device_pixels(which ? b : a,3));
                    float sum = 0;
                    for (int stage = 0; stage < ASCII_GPU_STAGES; ++stage) {
                        REQUIRE(graph.gpu_stage_milliseconds[stage] >= 0);
                        if (!profile) REQUIRE(graph.gpu_stage_milliseconds[stage] == 0);
                        sum += graph.gpu_stage_milliseconds[stage];
                    }
                    REQUIRE(sum <= graph.gpu_milliseconds+0.1f);
                    ascii_output_free(&graph); ascii_output_free(&direct);
                }
            AsciiOutput invalid = {};
            REQUIRE(!ascii_context_convert(a, NULL, &invalid, error, sizeof(error)));
            REQUIRE(!invalid.final_pixels);
            ascii_context_destroy(a); ascii_context_destroy(b);
        }
    /* Exercise FFT capture even on these small frames, with separate filter plans. */
    AsciiOptions fft_options;
    ascii_options_default(&fft_options);
    fft_options.bloom_method = 2; fft_options.profile = 1;
    AsciiContext *fft = ascii_context_create(67,51,fill.rgb,fill.width,fill.height,
        edge.rgb,edge.width,edge.height,&fft_options,1,1,error,sizeof(error));
    REQUIRE(fft);
    for (int frame = 0; frame < 3; ++frame) {
        for (size_t i = 0; i < sizeof(rgb); ++i) rgb[i] = frame == 2 ? 255 : (uint8_t)(i*17+frame*47);
        AsciiOutput actual = {}, expected = {};
        REQUIRE(ascii_context_convert(fft,rgb,&actual,error,sizeof(error)));
        fft_options.bloom_method = 1;
        REQUIRE(ascii_convert(rgb,67,51,fill.rgb,fill.width,fill.height,edge.rgb,edge.width,
            edge.height,&fft_options,1,1,&expected,error,sizeof(error)));
        for (int i = 0; i < 64*48; ++i) REQUIRE(abs(actual.final_pixels[i]-expected.final_pixels[i]) <= 2);
        REQUIRE(!memcmp(actual.edge_pixels,expected.edge_pixels,64*48));
        REQUIRE(!memcmp(actual.fill_pixels,expected.fill_pixels,64*48));
        float stage_sum = 0;
        for (int i = 0; i < ASCII_GPU_STAGES; ++i) stage_sum += actual.gpu_stage_milliseconds[i];
        REQUIRE(stage_sum <= actual.gpu_milliseconds+0.1f);
        ascii_output_free(&actual); ascii_output_free(&expected);
    }
    ascii_context_destroy(fft);
    /* Exhaust all attainable integer gradients against the original angle bins. */
    int bins[81], *device_bins = NULL;
    REQUIRE(cudaMalloc(&device_bins,sizeof(bins)) == cudaSuccess);
    gradient_bins<<<1,128>>>(device_bins);
    REQUIRE(cudaMemcpy(bins,device_bins,sizeof(bins),cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaFree(device_bins) == cudaSuccess);
    for (int gx = -4; gx <= 4; ++gx)
        for (int gy = -4; gy <= 4; ++gy) {
            int expected = -1;
            if (gx || gy) {
                float theta = atan2f((float)gy, (float)gx);
                float angle = fabsf(theta)/3.14159265358979323846f;
                if (angle <= 0.2f) { angle = 0; theta = 0; }
                if (angle < 0.05f || angle > 0.9f) expected = 1;
                else if (angle > 0.45f && angle < 0.55f) expected = 0;
                else if (angle > 0.05f && angle < 0.45f) expected = theta > 0 ? 2 : 3;
                else if (angle > 0.55f && angle < 0.9f) expected = theta > 0 ? 3 : 2;
            }
            int actual = bins[(gx+4)*9+gy+4];
            REQUIRE(actual == expected);
        }
    image_free(&fill); image_free(&edge);
    puts("PASS: reusable graphs, independent contexts, outputs, timing, and integer bins");
    return 0;
}
