#include "header.h"

#include <string.h>

#include "util.h"

static int hexval(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

int fbm_hex_decode(const char *hex, uint8_t *out, size_t out_len)
{
    if (strlen(hex) != 2 * out_len)
        return -1;
    for (size_t i = 0; i < out_len; i++) {
        int hi = hexval(hex[2 * i]), lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return -1;
        out[i] = (uint8_t)(hi << 4 | lo);
    }
    return 0;
}

void fbm_hex_encode(const uint8_t *in, size_t len, char *out)
{
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = digits[in[i] >> 4];
        out[2 * i + 1] = digits[in[i] & 15];
    }
    out[2 * len] = '\0';
}

void fbm_hash_display(const uint8_t hash[32], char out[65])
{
    uint8_t rev[32];
    for (int i = 0; i < 32; i++)
        rev[i] = hash[31 - i];
    fbm_hex_encode(rev, 32, out);
}

uint32_t fbm_header_get(const uint8_t *hdr, size_t off)
{
    return fbm_load_le32(hdr + off);
}

void fbm_header_set(uint8_t *hdr, size_t off, uint32_t v)
{
    fbm_store_le32(hdr + off, v);
}

int fbm_bits_to_target(uint32_t bits, uint8_t target_le[32])
{
    uint32_t exp = bits >> 24;
    uint32_t mant = bits & 0x007fffff;

    memset(target_le, 0, 32);
    if (bits & 0x00800000)
        return -1; /* negative */
    if (exp <= 3) {
        mant >>= 8 * (3 - exp);
        for (int i = 0; i < 3; i++)
            target_le[i] = (uint8_t)(mant >> (8 * i));
        return 0;
    }
    for (uint32_t i = 0; i < 3; i++) {
        uint8_t byte = (uint8_t)(mant >> (8 * i));
        uint32_t pos = exp - 3 + i;
        if (pos >= 32) {
            if (byte)
                return -1; /* overflow */
            continue;
        }
        target_le[pos] = byte;
    }
    return 0;
}

int fbm_cmp256_le(const uint8_t a[32], const uint8_t b[32])
{
    for (int i = 31; i >= 0; i--) {
        if (a[i] != b[i])
            return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

uint32_t fbm_top32_le(const uint8_t v[32])
{
    return fbm_load_le32(v + 28);
}

double fbm_target_difficulty(const uint8_t target_le[32])
{
    uint8_t one[32];
    fbm_bits_to_target(0x1d00ffff, one);
    double t = 0, d1 = 0;
    for (int i = 31; i >= 0; i--) {
        t = t * 256.0 + target_le[i];
        d1 = d1 * 256.0 + one[i];
    }
    return t > 0 ? d1 / t : 0;
}
