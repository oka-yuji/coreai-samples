#include "image_png.h"
#include "lodepng.h"
#include <stdlib.h>

static unsigned valid_chunks(const uint8_t *input, size_t count) {
    size_t offset = 8;
    while (offset < count) {
        if (count - offset < 12) return 0;
        const uint8_t *chunk = input + offset;
        size_t length = ((size_t)chunk[0] << 24) | ((size_t)chunk[1] << 16)
            | ((size_t)chunk[2] << 8) | chunk[3];
        if (length > count - offset - 12 || lodepng_chunk_check_crc(chunk)) return 0;
        if (lodepng_chunk_type_equals(chunk, "IEND")) return length == 0 && offset + 12 == count;
        offset += length + 12;
    }
    return 0;
}

unsigned coreai_png_decode_rgb8(const uint8_t *input, size_t input_count,
                              uint32_t maximum_dimension, size_t maximum_pixels,
                              uint8_t **pixels, uint32_t *width, uint32_t *height,
                              size_t *byte_count) {
    if (!pixels || !width || !height || !byte_count) return 1;
    *pixels = NULL;
    *width = 0;
    *height = 0;
    *byte_count = 0;
    if (!input || input_count < 33 || !maximum_dimension || !maximum_pixels) return 1;

    LodePNGState header;
    lodepng_state_init(&header);
    unsigned w = 0, h = 0;
    unsigned error = lodepng_inspect(&w, &h, &header, input, input_count);
    unsigned valid = !error && w && h && w <= maximum_dimension && h <= maximum_dimension
        && (size_t)w <= maximum_pixels / h && (size_t)w <= SIZE_MAX / h / 4
        && header.info_png.color.bitdepth == 8
        && (header.info_png.color.colortype == LCT_RGB || header.info_png.color.colortype == LCT_RGBA);
    lodepng_state_cleanup(&header);
    if (!valid) return error ? error : 1;

    size_t count = (size_t)w * h * 4;
    if ((size_t)h > (SIZE_MAX - 16) / 2 || count > SIZE_MAX - 2 * (size_t)h - 16
        || !valid_chunks(input, input_count)) return 1;
    LodePNGState decoder;
    lodepng_state_init(&decoder);
    decoder.info_raw.colortype = LCT_RGBA;
    decoder.info_raw.bitdepth = 8;
    decoder.decoder.zlibsettings.max_output_size = count + 2 * (size_t)h + 16;
    unsigned decoded_width = 0, decoded_height = 0;
    error = lodepng_decode(pixels, &decoded_width, &decoded_height, &decoder, input, input_count);
    lodepng_state_cleanup(&decoder);
    if (error || !*pixels || decoded_width != w || decoded_height != h) {
        free(*pixels);
        *pixels = NULL;
        return error ? error : 1;
    }
    *width = w;
    *height = h;
    *byte_count = count;
    return 0;
}

void coreai_png_release(uint8_t *pixels) {
    free(pixels);
}
