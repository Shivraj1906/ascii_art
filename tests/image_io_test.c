#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <png.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "Check failed at line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

int main(void) {
    char error[512];
    Image image = {0};
    const uint8_t pixels[6] = {0, 1, 64, 128, 254, 255};
    CHECK(image_write_gray("io_test.png", 3, 2, pixels, error, sizeof(error)));
    CHECK(image_load("io_test.png", &image, error, sizeof(error)));
    CHECK(image.width == 3 && image.height == 2);
    for (int i = 0; i < 6; ++i)
        for (int c = 0; c < 3; ++c) CHECK(image.rgb[3*i+c] == pixels[i]);
    image_free(&image);
    CHECK(image.rgb == NULL && image.width == 0);
    CHECK(!image_write_gray("io_test.png", 0, 2, pixels, error, sizeof(error)));
    for (int level = 0; level <= 9; ++level) {
        CHECK(image_write_gray_compressed("io_test.png",3,2,pixels,level,error,sizeof(error)));
        CHECK(image_load("io_test.png",&image,error,sizeof(error)));
        for (int i = 0; i < 6; ++i)
            for (int c = 0; c < 3; ++c) CHECK(image.rgb[3*i+c] == pixels[i]);
        image_free(&image);
    }
    CHECK(!image_write_gray_compressed("io_test.png",3,2,pixels,10,error,sizeof(error)));
    CHECK(!image_write_gray("missing_io_directory/io.png",3,2,pixels,error,sizeof(error)));
    CHECK(!image_load("missing_io_test.png", &image, error, sizeof(error)));

    /* Transparent pixels must retain RGB, rather than being flattened to black. */
    const uint8_t rgba[] = {240, 10, 30, 0, 10, 220, 20, 128};
    png_image png;
    memset(&png, 0, sizeof(png));
    png.version = PNG_IMAGE_VERSION; png.width = 2; png.height = 1;
    png.format = PNG_FORMAT_RGBA;
    CHECK(png_image_write_to_file(&png, "io_rgba.png", 0, rgba, 0, NULL));
    png_image_free(&png);
    CHECK(image_load("io_rgba.png", &image, error, sizeof(error)));
    for (int i = 0; i < 2; ++i)
        for (int c = 0; c < 3; ++c) CHECK(image.rgb[3*i+c] == rgba[4*i+c]);
    image_free(&image);

    FILE *file = fopen("io_test.jpg", "wb");
    CHECK(file);
    struct jpeg_compress_struct encoder;
    struct jpeg_error_mgr jpeg_error;
    encoder.err = jpeg_std_error(&jpeg_error);
    jpeg_create_compress(&encoder);
    jpeg_stdio_dest(&encoder, file);
    encoder.image_width = 8; encoder.image_height = 8;
    encoder.input_components = 1; encoder.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&encoder);
    jpeg_set_quality(&encoder, 100, TRUE);
    jpeg_start_compress(&encoder, TRUE);
    uint8_t row[8]; memset(row, 128, sizeof(row));
    while (encoder.next_scanline < encoder.image_height) {
        JSAMPROW pointer = row;
        jpeg_write_scanlines(&encoder, &pointer, 1);
    }
    jpeg_finish_compress(&encoder);
    jpeg_destroy_compress(&encoder);
    fclose(file);
    CHECK(image_load("io_test.jpg", &image, error, sizeof(error)));
    CHECK(image.width == 8 && image.height == 8);
    for (int i = 0; i < 8*8*3; ++i) CHECK(abs(image.rgb[i] - 128) <= 1);
    image_free(&image);

    file = fopen("io_bad.jpg", "wb"); CHECK(file);
    const uint8_t bad_jpeg[] = {0xff, 0xd8, 0, 0, 0, 0, 0, 0};
    CHECK(fwrite(bad_jpeg, 1, sizeof(bad_jpeg), file) == sizeof(bad_jpeg)); fclose(file);
    CHECK(!image_load("io_bad.jpg", &image, error, sizeof(error)));
    file = fopen("io_bad.png", "wb"); CHECK(file);
    const uint8_t bad_png[] = {137, 80, 78, 71, 13, 10, 26, 10};
    CHECK(fwrite(bad_png, 1, sizeof(bad_png), file) == sizeof(bad_png)); fclose(file);
    CHECK(!image_load("io_bad.png", &image, error, sizeof(error)));
    remove("io_test.png"); remove("io_test.jpg");
    remove("io_bad.png"); remove("io_bad.jpg"); remove("io_rgba.png");
    puts("PNG round trip, ignored alpha, grayscale JPEG, and decoder error recovery passed");
    return 0;
}
