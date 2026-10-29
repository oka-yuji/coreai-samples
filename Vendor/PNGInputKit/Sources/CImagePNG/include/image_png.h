#ifndef COREAI_IMAGE_PNG_H
#define COREAI_IMAGE_PNG_H

#include <stddef.h>
#include <stdint.h>

unsigned coreai_png_decode_rgb8(const uint8_t *input, size_t input_count,
                              uint32_t maximum_dimension, size_t maximum_pixels,
                              uint8_t **pixels, uint32_t *width, uint32_t *height,
                              size_t *byte_count);
void coreai_png_release(uint8_t *pixels);

#endif
