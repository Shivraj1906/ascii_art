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
    /* RGB inputs need no intermediate RGBA buffer or packing pass. Alpha inputs
     * still discard alpha without compositing, matching the Python converter. */
    int alpha = (png.format & PNG_FORMAT_FLAG_ALPHA) != 0;
    png.format = alpha ? PNG_FORMAT_RGBA : PNG_FORMAT_RGB;
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
    if (!alpha) {
        out->rgb = rgba;
        out->width = (int)png.width;
        out->height = (int)png.height;
        png_image_free(&png);
        return 1;
    }
    out->rgb = malloc(count * 3);
    if (!out->rgb) {
        free(rgba);
        png_image_free(&png);
        return fail(error, size, "out of memory decoding PNG");
    }
    for (size_t i = 0; i < count; ++i) {
        out->rgb[3*i] = rgba[4*i];
        out->rgb[3*i+1] = rgba[4*i+1];
        out->rgb[3*i+2] = rgba[4*i+2];
    }
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

typedef struct {
    png_structp png;
    png_infop info;
    FILE *file;
    char message[256];
} PngWriter;

static void png_failure(png_structp png, png_const_charp message) {
    PngWriter *ctx = png_get_error_ptr(png);
    snprintf(ctx->message, sizeof(ctx->message), "%s", message);
    png_longjmp(png, 1);
}

static void png_warning_ignored(png_structp png, png_const_charp message) {
    (void)png; (void)message;
}

int image_write_gray_compressed(const char *path, int width, int height,
                     const uint8_t *pixels, int compression, char *error, size_t size) {
    if (width <= 0 || height <= 0 || !pixels || compression < 0 || compression > 9)
        return fail(error, size, "invalid output image");
    PngWriter *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return fail(error, size, "out of memory writing PNG");
    int success = 0;
    ctx->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, ctx, png_failure, png_warning_ignored);
    if (!ctx->png) { free(ctx); return fail(error, size, "cannot initialize PNG writer"); }
    if (setjmp(png_jmpbuf(ctx->png))) {
        fail(error, size, ctx->message);
        goto cleanup;
    }
    ctx->info = png_create_info_struct(ctx->png);
    if (!ctx->info) png_error(ctx->png, "cannot allocate PNG metadata");
    ctx->file = fopen(path, "wb");
    if (!ctx->file) png_error(ctx->png, "cannot open PNG output");
    png_init_io(ctx->png, ctx->file);
    png_set_IHDR(ctx->png, ctx->info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_sRGB(ctx->png, ctx->info, PNG_sRGB_INTENT_PERCEPTUAL);
    png_set_compression_level(ctx->png, compression);
    /* Avoid five filter trials per row. Level 6+ trades time for smaller files. */
    png_set_filter(ctx->png, PNG_FILTER_TYPE_BASE,
                   compression >= 6 ? PNG_ALL_FILTERS : PNG_FILTER_NONE);
    png_write_info(ctx->png, ctx->info);
    for (int y = 0; y < height; ++y)
        png_write_row(ctx->png, pixels + (size_t)y * width);
    png_write_end(ctx->png, ctx->info);
    success = 1;
cleanup:
    png_destroy_write_struct(&ctx->png, &ctx->info);
    if (ctx->file && fclose(ctx->file) && success) {
        fail(error, size, "cannot finish writing PNG"); success = 0;
    }
    free(ctx);
    return success;
}

int image_write_gray(const char *path, int width, int height,
                     const uint8_t *pixels, char *error, size_t size) {
    return image_write_gray_compressed(path, width, height, pixels, 1, error, size);
}

void image_free(Image *image) {
    free(image->rgb);
    memset(image, 0, sizeof(*image));
}
