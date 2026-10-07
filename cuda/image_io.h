#ifndef ASCII_IMAGE_IO_H
#define ASCII_IMAGE_IO_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int width, height;
    uint8_t *rgb;
} Image;

/* PNG/JPEG are decoded to RGB8. Alpha is ignored, as in the Python converter. */
int image_load(const char *path, Image *image, char *error, size_t error_size);
int image_write_gray(const char *path, int width, int height,
                     const uint8_t *pixels, char *error, size_t error_size);
/* Lossless PNG: default level 1 favors throughput; levels 6..9 enable filters. */
int image_write_gray_compressed(const char *path, int width, int height,
                     const uint8_t *pixels, int compression, char *error, size_t error_size);
void image_free(Image *image);
#endif
