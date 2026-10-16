#ifndef IRODORI_TOKENIZERS_H
#define IRODORI_TOKENIZERS_H
#include <stdint.h>
#include <stddef.h>
typedef struct { uint32_t *data; size_t len; size_t capacity; } IroTokens;
void *iro_tokenizer_create(const uint8_t *json, size_t len, char **error);
int32_t iro_tokenizer_encode(const void *handle, const uint8_t *text, size_t len, IroTokens *out, char **error);
void iro_tokenizer_destroy(void *handle);
void iro_tokens_free(IroTokens tokens);
void iro_error_free(char *error);
typedef struct { uint8_t *data; size_t len; size_t capacity; } IroBytes;
int32_t iro_nfkc(const uint8_t *text, size_t len, IroBytes *out, char **error);
void iro_bytes_free(IroBytes value);
#endif
