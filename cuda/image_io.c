#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <setjmp.h>
#include <png.h>
#include <jpeglib.h>

static int fail(char *error, size_t size, const char *message) {
    snprintf(error, size, "%s", message);
    return 0;
}

static int dimensions_ok(unsigned long width, unsigned long height) {
    return width && height && width <= INT_MAX && height <= INT_MAX &&
           width <= SIZE_MAX / 4 / height;
}

static int load_png(const char *path, Image *out, char *error, size_t size) {
    png_image png;
    memset(&png, 0, sizeof(png));
    png.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&png, path)) {
        fail(error, size, png.message);
        png_image_free(&png);
        return 0;
    }
    if (!dimensions_ok(png.width, png.height)) {
        png_image_free(&png);
        return fail(error, size, "image dimensions are too large or empty");
    }
    /* Decode RGBA and discard alpha without compositing against a background. */
    png.format = PNG_FORMAT_RGBA;
    uint8_t *rgba = malloc(PNG_IMAGE_SIZE(png));
    if (!rgba) {
        png_image_free(&png);
        return fail(error, size, "out of memory decoding PNG");
    }
    if (!png_image_finish_read(&png, NULL, rgba, 0, NULL)) {
        fail(error, size, png.message);
        free(rgba);
        png_image_free(&png);
        return 0;
    }
    size_t count = (size_t)png.width * png.height;
    out->rgb = malloc(count * 3);
    if (!out->rgb) {
        free(rgba);
        png_image_free(&png);
        return fail(error, size, "out of memory decoding PNG");
    }
    for (size_t i = 0; i < count; ++i)
        memcpy(out->rgb + 3 * i, rgba + 4 * i, 3);
    out->width = (int)png.width;
    out->height = (int)png.height;
    free(rgba);
    png_image_free(&png);
    return 1;
}

typedef struct {
    struct jpeg_error_mgr base;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
} JpegError;

typedef struct {
    struct jpeg_decompress_struct decoder;
    JpegError error;
    FILE *file;
    uint8_t *pixels;
    int created;
} JpegContext;

static void jpeg_failure(j_common_ptr common) {
    JpegError *error = (JpegError *)common->err;
    (*common->err->format_message)(common, error->message);
    longjmp(error->jump, 1);
}

static void jpeg_cleanup(JpegContext *ctx) {
    if (ctx->created) jpeg_destroy_decompress(&ctx->decoder);
    if (ctx->file) fclose(ctx->file);
    free(ctx->pixels);
    free(ctx);
}

static int load_jpeg(const char *path, Image *out, char *error, size_t size) {
    /* Heap state remains defined after libjpeg's longjmp error recovery. */
    JpegContext *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return fail(error, size, "out of memory decoding JPEG");
    ctx->decoder.err = jpeg_std_error(&ctx->error.base);
    ctx->error.base.error_exit = jpeg_failure;
    if (setjmp(ctx->error.jump)) {
        fail(error, size, ctx->error.message);
        jpeg_cleanup(ctx);
        return 0;
    }
    ctx->file = fopen(path, "rb");
    if (!ctx->file) {
        jpeg_cleanup(ctx);
        return fail(error, size, "cannot open JPEG input");
    }
    jpeg_create_decompress(&ctx->decoder);
    ctx->created = 1;
    jpeg_stdio_src(&ctx->decoder, ctx->file);
    jpeg_read_header(&ctx->decoder, TRUE);
    ctx->decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&ctx->decoder);
    if (!dimensions_ok(ctx->decoder.output_width, ctx->decoder.output_height)) {
        jpeg_cleanup(ctx);
        return fail(error, size, "image dimensions are too large or empty");
    }
    size_t row_size = (size_t)ctx->decoder.output_width * 3;
    ctx->pixels = malloc(row_size * ctx->decoder.output_height);
    if (!ctx->pixels) {
        jpeg_cleanup(ctx);
        return fail(error, size, "out of memory decoding JPEG");
    }
    while (ctx->decoder.output_scanline < ctx->decoder.output_height) {
        JSAMPROW row = ctx->pixels + row_size * ctx->decoder.output_scanline;
        jpeg_read_scanlines(&ctx->decoder, &row, 1);
    }
    out->width = (int)ctx->decoder.output_width;
    out->height = (int)ctx->decoder.output_height;
    jpeg_finish_decompress(&ctx->decoder);
    out->rgb = ctx->pixels;
    ctx->pixels = NULL;
    jpeg_cleanup(ctx);
    return 1;
}

int image_load(const char *path, Image *image, char *error, size_t size) {
    memset(image, 0, sizeof(*image));
    FILE *file = fopen(path, "rb");
    if (!file) return fail(error, size, "cannot open input file");
    unsigned char signature[8];
    size_t count = fread(signature, 1, sizeof(signature), file);
    fclose(file);
    if (count == 8 && !png_sig_cmp(signature, 0, 8))
        return load_png(path, image, error, size);
    if (count >= 2 && signature[0] == 0xff && signature[1] == 0xd8)
        return load_jpeg(path, image, error, size);
    return fail(error, size, "unsupported input: expected PNG or JPEG");
}

int image_write_gray(const char *path, int width, int height,
                     const uint8_t *pixels, char *error, size_t size) {
    if (width <= 0 || height <= 0 || !pixels)
        return fail(error, size, "invalid output image");
    png_image png;
    memset(&png, 0, sizeof(png));
    png.version = PNG_IMAGE_VERSION;
    png.width = (png_uint_32)width;
    png.height = (png_uint_32)height;
    png.format = PNG_FORMAT_GRAY;
    if (!png_image_write_to_file(&png, path, 0, pixels, 0, NULL)) {
        fail(error, size, png.message);
        png_image_free(&png);
        return 0;
    }
    png_image_free(&png);
    return 1;
}

void image_free(Image *image) {
    free(image->rgb);
    memset(image, 0, sizeof(*image));
}
