#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "oec_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * Internal helpers
 * ============================================================ */

static float clamp_float(float value, float min, float max)
{
    if (value < min)
        return min;

    if (value > max)
        return max;

    return value;
}

static float get_pixel(
    const OecImage *image,
    int x,
    int y,
    int channel)
{
    size_t index;

    index =
        ((size_t)y * (size_t)image->width +
         (size_t)x) *
        (size_t)image->channels +
        (size_t)channel;

    return image->data[index];
}

static void set_pixel(
    OecImage *image,
    int x,
    int y,
    int channel,
    float value)
{
    size_t index;

    index =
        ((size_t)y * (size_t)image->width +
         (size_t)x) *
        (size_t)image->channels +
        (size_t)channel;

    image->data[index] = value;
}

/*
 * Reset an image structure.
 */
static void image_clear(
    OecImage *image)
{
    if (image == NULL)
        return;

    image->width = 0;
    image->height = 0;
    image->channels = 0;
    image->data = NULL;
}


/* ============================================================
 * Memory
 * ============================================================ */

int oec_image_create(
    OecImage *image,
    int width,
    int height,
    int channels)
{
    size_t count;

    if (image == NULL)
        return -1;

    if (width <= 0 ||
        height <= 0 ||
        channels <= 0)
        return -1;

    count =
        (size_t)width *
        (size_t)height *
        (size_t)channels;

    image->data = calloc(count, sizeof(float));

    if (image->data == NULL)
        return -1;

    image->width = width;
    image->height = height;
    image->channels = channels;

    return 0;
}

void oec_image_free(
    OecImage *image)
{
    if (image == NULL)
        return;

    free(image->data);

    image_clear(image);
}

/*
 * Load an image from disk.
 *
 * STB loads the source image and converts it
 * to RGB.
 *
 * Internal representation:
 *
 *     float RGB
 *     range = 0.0 ... 1.0
 */
int oec_image_load(
    const char *filename,
    OecImage *image)
{
    int width;
    int height;
    int channels;

    unsigned char *pixels;
    float *data;

    size_t pixel_count;
    size_t i;

    if (filename == NULL || image == NULL)
        return -1;

    image_clear(image);

    pixels = stbi_load(
        filename,
        &width,
        &height,
        &channels,
        OEC_IMAGE_CHANNELS
    );

    if (pixels == NULL) {

        fprintf(
            stderr,
            "Failed to load image: %s\n",
            filename
        );

        return -1;
    }

    pixel_count =
        (size_t)width *
        (size_t)height *
        OEC_IMAGE_CHANNELS;

    data = malloc(
        pixel_count * sizeof(float)
    );

    if (data == NULL) {

        stbi_image_free(pixels);

        fprintf(
            stderr,
            "Failed to allocate image memory\n"
        );

        return -1;
    }

    /*
     * Convert uint8 [0,255]
     * to float [0.0,1.0].
     */
    for (i = 0; i < pixel_count; i++) {

        data[i] =
            (float)pixels[i] / 255.0f;
    }

    stbi_image_free(pixels);

    image->width = width;
    image->height = height;
    image->channels =
        OEC_IMAGE_CHANNELS;

    image->data = data;

    return 0;
}



/* ============================================================
 * Geometry
 * ============================================================ */

int oec_image_crop(
    const OecImage *source,
    OecImage *destination,
    int x,
    int y,
    int width,
    int height)
{
    int px;
    int py;
    int c;

    if (source == NULL ||
        destination == NULL ||
        source->data == NULL)
        return -1;

    if (source->channels <= 0)
        return -1;

    if (width <= 0 ||
        height <= 0)
        return -1;

    if (x < 0 ||
        y < 0 ||
        x + width > source->width ||
        y + height > source->height)
        return -1;

    oec_image_free(destination);

    if (oec_image_create(
            destination,
            width,
            height,
            source->channels) != 0)
        return -1;

    for (py = 0; py < height; py++) {
        for (px = 0; px < width; px++) {
            for (c = 0; c < source->channels; c++) {
                float value;

                value = get_pixel(
                    source,
                    x + px,
                    y + py,
                    c
                );

                set_pixel(
                    destination,
                    px,
                    py,
                    c,
                    value
                );
            }
        }
    }

    return 0;
}


int oec_image_resize(
    const OecImage *source,
    OecImage *destination,
    int width,
    int height)
{
    int x;
    int y;
    int c;

    if (source == NULL ||
        destination == NULL ||
        source->data == NULL)
        return -1;

    if (width <= 0 ||
        height <= 0)
        return -1;

    oec_image_free(destination);

    if (oec_image_create(
            destination,
            width,
            height,
            source->channels) != 0)
        return -1;

    for (y = 0; y < height; y++) {
        int source_y;

        source_y =
            (y * source->height) / height;

        if (source_y >= source->height)
            source_y = source->height - 1;

        for (x = 0; x < width; x++) {
            int source_x;

            source_x =
                (x * source->width) / width;

            if (source_x >= source->width)
                source_x = source->width - 1;

            for (c = 0; c < source->channels; c++) {
                float value;

                value = get_pixel(
                    source,
                    source_x,
                    source_y,
                    c
                );

                set_pixel(
                    destination,
                    x,
                    y,
                    c,
                    value
                );
            }
        }
    }

    return 0;
}


int oec_image_resize_letterbox(
    const OecImage *source,
    OecImage *destination)
{
    float scale;

    int resized_width;
    int resized_height;

    int offset_x;
    int offset_y;

    OecImage resized;

    int x;
    int y;
    int c;

    if (source == NULL ||
        destination == NULL ||
        source->data == NULL)
        return -1;

    if (source->width <= 0 ||
        source->height <= 0)
        return -1;

    image_clear(&resized);

    scale = fminf(
        (float)OEC_IMAGE_WIDTH /
            (float)source->width,
        (float)OEC_IMAGE_HEIGHT /
            (float)source->height
    );

    resized_width =
        (int)lroundf(
            source->width * scale
        );

    resized_height =
        (int)lroundf(
            source->height * scale
        );

    if (resized_width < 1)
        resized_width = 1;

    if (resized_height < 1)
        resized_height = 1;

    if (oec_image_resize(
            source,
            &resized,
            resized_width,
            resized_height) != 0)
        return -1;

    oec_image_free(destination);

    if (oec_image_create(
            destination,
            OEC_IMAGE_WIDTH,
            OEC_IMAGE_HEIGHT,
            source->channels) != 0) {
        oec_image_free(&resized);
        return -1;
    }

    /*
     * calloc() already initialized the padding to black.
     */

    offset_x =
        (OEC_IMAGE_WIDTH - resized_width) / 2;

    offset_y =
        (OEC_IMAGE_HEIGHT - resized_height) / 2;

    for (y = 0; y < resized_height; y++) {
        for (x = 0; x < resized_width; x++) {
            for (c = 0; c < source->channels; c++) {
                float value;

                value = get_pixel(
                    &resized,
                    x,
                    y,
                    c
                );

                set_pixel(
                    destination,
                    offset_x + x,
                    offset_y + y,
                    c,
                    value
                );
            }
        }
    }

    oec_image_free(&resized);

    return 0;
}


int oec_image_flip_horizontal(
    OecImage *image)
{
    int y;
    int x;
    int c;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width / 2; x++) {
            int opposite_x;

            opposite_x =
                image->width - 1 - x;

            for (c = 0; c < image->channels; c++) {
                float temp;

                temp = get_pixel(
                    image,
                    x,
                    y,
                    c
                );

                set_pixel(
                    image,
                    x,
                    y,
                    c,
                    get_pixel(
                        image,
                        opposite_x,
                        y,
                        c
                    )
                );

                set_pixel(
                    image,
                    opposite_x,
                    y,
                    c,
                    temp
                );
            }
        }
    }

    return 0;
}


int oec_image_flip_vertical(
    OecImage *image)
{
    int y;
    int x;
    int c;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    for (y = 0; y < image->height / 2; y++) {
        int opposite_y;

        opposite_y =
            image->height - 1 - y;

        for (x = 0; x < image->width; x++) {
            for (c = 0; c < image->channels; c++) {
                float temp;

                temp = get_pixel(
                    image,
                    x,
                    y,
                    c
                );

                set_pixel(
                    image,
                    x,
                    y,
                    c,
                    get_pixel(
                        image,
                        x,
                        opposite_y,
                        c
                    )
                );

                set_pixel(
                    image,
                    x,
                    opposite_y,
                    c,
                    temp
                );
            }
        }
    }

    return 0;
}


int oec_image_rotate_90(
    const OecImage *source,
    OecImage *destination)
{
    int x;
    int y;
    int c;

    if (source == NULL ||
        destination == NULL ||
        source->data == NULL)
        return -1;

    oec_image_free(destination);

    if (oec_image_create(
            destination,
            source->height,
            source->width,
            source->channels) != 0)
        return -1;

    /*
     * Clockwise 90 degree rotation:
     *
     * source (x,y)
     *       ->
     * destination(height - 1 - y, x)
     */

    for (y = 0; y < source->height; y++) {
        for (x = 0; x < source->width; x++) {
            int destination_x;
            int destination_y;

            destination_x =
                source->height - 1 - y;

            destination_y = x;

            for (c = 0; c < source->channels; c++) {
                set_pixel(
                    destination,
                    destination_x,
                    destination_y,
                    c,
                    get_pixel(
                        source,
                        x,
                        y,
                        c
                    )
                );
            }
        }
    }

    return 0;
}


/* ============================================================
 * Brightness / contrast
 * ============================================================ */

int oec_image_adjust_brightness(
    OecImage *image,
    float factor)
{
    size_t count;
    size_t i;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (factor < 0.0f)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    for (i = 0; i < count; i++) {
        image->data[i] =
            clamp_float(
                image->data[i] * factor,
                0.0f,
                1.0f
            );
    }

    return 0;
}


int oec_image_adjust_contrast(
    OecImage *image,
    float factor)
{
    size_t count;
    size_t i;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (factor < 0.0f)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    for (i = 0; i < count; i++) {
        float value;

        value =
            (image->data[i] - 0.5f) *
            factor +
            0.5f;

        image->data[i] =
            clamp_float(
                value,
                0.0f,
                1.0f
            );
    }

    return 0;
}


/* ============================================================
 * Saturation / hue
 * ============================================================ */

int oec_image_adjust_saturation(
    OecImage *image,
    float factor)
{
    int x;
    int y;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (image->channels < 3 ||
        factor < 0.0f)
        return -1;

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            float r;
            float g;
            float b;
            float gray;

            r = get_pixel(image, x, y, 0);
            g = get_pixel(image, x, y, 1);
            b = get_pixel(image, x, y, 2);

            gray =
                0.299f * r +
                0.587f * g +
                0.114f * b;

            r = gray + (r - gray) * factor;
            g = gray + (g - gray) * factor;
            b = gray + (b - gray) * factor;

            set_pixel(
                image, x, y, 0,
                clamp_float(r, 0.0f, 1.0f)
            );

            set_pixel(
                image, x, y, 1,
                clamp_float(g, 0.0f, 1.0f)
            );

            set_pixel(
                image, x, y, 2,
                clamp_float(b, 0.0f, 1.0f)
            );
        }
    }

    return 0;
}


int oec_image_adjust_hue(
    OecImage *image,
    float degrees)
{
    int x;
    int y;

    float radians;
    float cos_h;
    float sin_h;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (image->channels < 3)
        return -1;

    radians =
        degrees *
        (float)M_PI /
        180.0f;

    cos_h = cosf(radians);
    sin_h = sinf(radians);

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            float r;
            float g;
            float b;

            float nr;
            float ng;
            float nb;

            r = get_pixel(image, x, y, 0);
            g = get_pixel(image, x, y, 1);
            b = get_pixel(image, x, y, 2);

            /*
             * YIQ hue rotation.
             */

            nr =
                0.299f * r +
                0.587f * g +
                0.114f * b;

            ng =
                0.596f * r -
                0.274f * g -
                0.322f * b;

            nb =
                0.211f * r -
                0.523f * g +
                0.312f * b;

            {
                float rotated_g;
                float rotated_b;

                rotated_g =
                    ng * cos_h -
                    nb * sin_h;

                rotated_b =
                    ng * sin_h +
                    nb * cos_h;

                nr = nr;
                ng = rotated_g;
                nb = rotated_b;
            }

            r =
                nr +
                0.956f * ng +
                0.621f * nb;

            g =
                nr -
                0.272f * ng -
                0.647f * nb;

            b =
                nr -
                1.106f * ng +
                1.703f * nb;

            set_pixel(
                image, x, y, 0,
                clamp_float(r, 0.0f, 1.0f)
            );

            set_pixel(
                image, x, y, 1,
                clamp_float(g, 0.0f, 1.0f)
            );

            set_pixel(
                image, x, y, 2,
                clamp_float(b, 0.0f, 1.0f)
            );
        }
    }

    return 0;
}


/* ============================================================
 * Gamma
 * ============================================================ */

int oec_image_adjust_gamma(
    OecImage *image,
    float gamma)
{
    size_t count;
    size_t i;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (gamma <= 0.0f)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    for (i = 0; i < count; i++) {
        float value;

        value =
            powf(
                clamp_float(
                    image->data[i],
                    0.0f,
                    1.0f
                ),
                1.0f / gamma
            );

        image->data[i] =
            clamp_float(
                value,
                0.0f,
                1.0f
            );
    }

    return 0;
}


/* ============================================================
 * Temperature
 * ============================================================ */

int oec_image_adjust_temperature(
    OecImage *image,
    float amount)
{
    int x;
    int y;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (image->channels < 3)
        return -1;

    /*
     * amount:
     *
     *  -1.0 = strongly cool
     *   0.0 = unchanged
     *  +1.0 = strongly warm
     */

    amount =
        clamp_float(
            amount,
            -1.0f,
            1.0f
        );

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            float r;
            float g;
            float b;

            r = get_pixel(image, x, y, 0);
            g = get_pixel(image, x, y, 1);
            b = get_pixel(image, x, y, 2);

            /*
             * Warm:
             *   increase red
             *   slightly increase green
             *   reduce blue
             *
             * Cool:
             *   inverse operation.
             */

            r += amount * 0.10f;
            g += amount * 0.025f;
            b -= amount * 0.10f;

            set_pixel(
                image, x, y, 0,
                clamp_float(r, 0.0f, 1.0f)
            );

            set_pixel(
                image, x, y, 1,
                clamp_float(g, 0.0f, 1.0f)
            );

            set_pixel(
                image, x, y, 2,
                clamp_float(b, 0.0f, 1.0f)
            );
        }
    }

    return 0;
}


/* ============================================================
 * Grayscale / invert
 * ============================================================ */

int oec_image_grayscale(
    OecImage *image)
{
    int x;
    int y;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (image->channels < 3)
        return -1;

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            float r;
            float g;
            float b;
            float gray;

            r = get_pixel(image, x, y, 0);
            g = get_pixel(image, x, y, 1);
            b = get_pixel(image, x, y, 2);

            gray =
                0.299f * r +
                0.587f * g +
                0.114f * b;

            set_pixel(image, x, y, 0, gray);
            set_pixel(image, x, y, 1, gray);
            set_pixel(image, x, y, 2, gray);
        }
    }

    return 0;
}


int oec_image_invert(
    OecImage *image)
{
    size_t count;
    size_t i;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    for (i = 0; i < count; i++) {
        image->data[i] =
            1.0f - image->data[i];
    }

    return 0;
}


/* ============================================================
 * Blur
 * ============================================================ */

int oec_image_blur(
    OecImage *image,
    int radius)
{
    float *temporary;

    int x;
    int y;
    int c;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (radius < 0)
        return -1;

    if (radius == 0)
        return 0;

    temporary =
        malloc(
            (size_t)image->width *
            (size_t)image->height *
            (size_t)image->channels *
            sizeof(float)
        );

    if (temporary == NULL)
        return -1;

    /*
     * Horizontal pass.
     */

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            for (c = 0; c < image->channels; c++) {
                float sum;
                int count;
                int dx;

                sum = 0.0f;
                count = 0;

                for (dx = -radius; dx <= radius; dx++) {
                    int sx;

                    sx = x + dx;

                    if (sx < 0 ||
                        sx >= image->width)
                        continue;

                    sum += get_pixel(
                        image,
                        sx,
                        y,
                        c
                    );

                    count++;
                }

                temporary[
                    ((size_t)y *
                     (size_t)image->width +
                     (size_t)x) *
                    (size_t)image->channels +
                    (size_t)c
                ] = sum / (float)count;
            }
        }
    }

    /*
     * Vertical pass.
     */

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            for (c = 0; c < image->channels; c++) {
                float sum;
                int count;
                int dy;

                sum = 0.0f;
                count = 0;

                for (dy = -radius; dy <= radius; dy++) {
                    int sy;

                    sy = y + dy;

                    if (sy < 0 ||
                        sy >= image->height)
                        continue;

                    sum +=
                        temporary[
                            ((size_t)sy *
                             (size_t)image->width +
                             (size_t)x) *
                            (size_t)image->channels +
                            (size_t)c
                        ];

                    count++;
                }

                set_pixel(
                    image,
                    x,
                    y,
                    c,
                    sum / (float)count
                );
            }
        }
    }

    free(temporary);

    return 0;
}


/* ============================================================
 * Sharpen
 * ============================================================ */

int oec_image_sharpen(
    OecImage *image,
    float amount)
{
    float *original;

    size_t count;
    size_t i;

    int x;
    int y;
    int c;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (amount < 0.0f)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    original =
        malloc(count * sizeof(float));

    if (original == NULL)
        return -1;

    memcpy(
        original,
        image->data,
        count * sizeof(float)
    );

    for (y = 1; y < image->height - 1; y++) {
        for (x = 1; x < image->width - 1; x++) {
            for (c = 0; c < image->channels; c++) {
                float center;
                float neighbours;
                float detail;
                float value;

                center =
                    get_pixel(
                        image,
                        x,
                        y,
                        c
                    );

                neighbours =
                    get_pixel(
                        image,
                        x - 1,
                        y,
                        c
                    ) +
                    get_pixel(
                        image,
                        x + 1,
                        y,
                        c
                    ) +
                    get_pixel(
                        image,
                        x,
                        y - 1,
                        c
                    ) +
                    get_pixel(
                        image,
                        x,
                        y + 1,
                        c
                    );

                neighbours *= 0.25f;

                detail =
                    center - neighbours;

                value =
                    original[
                        ((size_t)y *
                         (size_t)image->width +
                         (size_t)x) *
                        (size_t)image->channels +
                        (size_t)c
                    ] +
                    detail * amount;

                set_pixel(
                    image,
                    x,
                    y,
                    c,
                    clamp_float(
                        value,
                        0.0f,
                        1.0f
                    )
                );
            }
        }
    }

    free(original);

    return 0;
}


/* ============================================================
 * Noise
 * ============================================================ */

int oec_image_add_noise(
    OecImage *image,
    float amount)
{
    size_t count;
    size_t i;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (amount < 0.0f)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    for (i = 0; i < count; i++) {
        float random_value;
        float noise;

        random_value =
            (float)rand() /
            (float)RAND_MAX;

        noise =
            (random_value * 2.0f - 1.0f) *
            amount;

        image->data[i] =
            clamp_float(
                image->data[i] + noise,
                0.0f,
                1.0f
            );
    }

    return 0;
}


/* ============================================================
 * Pixelate
 * ============================================================ */

int oec_image_pixelate(
    OecImage *image,
    int block_size)
{
    int x;
    int y;
    int c;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (block_size <= 0)
        return -1;

    for (y = 0; y < image->height; y += block_size) {
        for (x = 0; x < image->width; x += block_size) {
            int end_x;
            int end_y;

            end_x =
                x + block_size;

            end_y =
                y + block_size;

            if (end_x > image->width)
                end_x = image->width;

            if (end_y > image->height)
                end_y = image->height;

            for (c = 0; c < image->channels; c++) {
                float sum;
                int count;

                int px;
                int py;

                sum = 0.0f;
                count = 0;

                for (py = y; py < end_y; py++) {
                    for (px = x; px < end_x; px++) {
                        sum += get_pixel(
                            image,
                            px,
                            py,
                            c
                        );

                        count++;
                    }
                }

                if (count > 0) {
                    float average;

                    average =
                        sum / (float)count;

                    for (py = y; py < end_y; py++) {
                        for (px = x; px < end_x; px++) {
                            set_pixel(
                                image,
                                px,
                                py,
                                c,
                                average
                            );
                        }
                    }
                }
            }
        }
    }

    return 0;
}


/* ============================================================
 * Auto contrast
 * ============================================================ */

int oec_image_auto_contrast(
    OecImage *image)
{
    size_t count;
    size_t i;

    float minimum;
    float maximum;
    float range;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    count =
        (size_t)image->width *
        (size_t)image->height *
        (size_t)image->channels;

    if (count == 0)
        return -1;

    minimum = 1.0f;
    maximum = 0.0f;

    for (i = 0; i < count; i++) {
        float value;

        value =
            clamp_float(
                image->data[i],
                0.0f,
                1.0f
            );

        if (value < minimum)
            minimum = value;

        if (value > maximum)
            maximum = value;
    }

    range = maximum - minimum;

    if (range <= 0.000001f)
        return 0;

    for (i = 0; i < count; i++) {
        image->data[i] =
            clamp_float(
                (image->data[i] - minimum) /
                range,
                0.0f,
                1.0f
            );
    }

    return 0;
}


/* ============================================================
 * Histogram equalization
 * ============================================================ */

int oec_image_histogram_equalize(
    OecImage *image)
{
    unsigned int histogram[256];
    unsigned int cumulative[256];

    size_t pixel_count;

    int i;
    int x;
    int y;

    if (image == NULL ||
        image->data == NULL)
        return -1;

    if (image->channels < 3)
        return -1;

    memset(
        histogram,
        0,
        sizeof(histogram)
    );

    /*
     * Build luminance histogram.
     */

    pixel_count =
        (size_t)image->width *
        (size_t)image->height;

    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            float r;
            float g;
            float b;
            float gray;

            int bin;

            r = get_pixel(image, x, y, 0);
            g = get_pixel(image, x, y, 1);
            b = get_pixel(image, x, y, 2);

            gray =
                0.299f * r +
                0.587f * g +
                0.114f * b;

            bin =
                (int)lroundf(
                    clamp_float(
                        gray,
                        0.0f,
                        1.0f
                    ) * 255.0f
                );

            if (bin < 0)
                bin = 0;

            if (bin > 255)
                bin = 255;

            histogram[bin]++;
        }
    }

    cumulative[0] = histogram[0];

    for (i = 1; i < 256; i++) {
        cumulative[i] =
            cumulative[i - 1] +
            histogram[i];
    }

    {
        unsigned int minimum;

        minimum = 0;

        for (i = 0; i < 256; i++) {
            if (cumulative[i] != 0) {
                minimum = cumulative[i];
                break;
            }
        }

        if (pixel_count <= minimum)
            return 0;

        for (y = 0; y < image->height; y++) {
            for (x = 0; x < image->width; x++) {
                float r;
                float g;
                float b;

                float gray;
                float new_gray;

                float scale;

                r = get_pixel(image, x, y, 0);
                g = get_pixel(image, x, y, 1);
                b = get_pixel(image, x, y, 2);

                gray =
                    0.299f * r +
                    0.587f * g +
                    0.114f * b;

                {
                    int bin;

                    bin =
                        (int)lroundf(
                            clamp_float(
                                gray,
                                0.0f,
                                1.0f
                            ) * 255.0f
                        );

                    if (bin < 0)
                        bin = 0;

                    if (bin > 255)
                        bin = 255;

                    scale =
                        (float)(
                            cumulative[bin] -
                            minimum
                        ) /
                        (float)(
                            pixel_count -
                            minimum
                        );
                }

                new_gray =
                    clamp_float(
                        scale,
                        0.0f,
                        1.0f
                    );

                /*
                 * Preserve chromatic ratios while
                 * replacing luminance.
                 */

                if (gray > 0.000001f) {
                    float ratio;

                    ratio =
                        new_gray / gray;

                    r *= ratio;
                    g *= ratio;
                    b *= ratio;
                } else {
                    r = new_gray;
                    g = new_gray;
                    b = new_gray;
                }

                set_pixel(
                    image, x, y, 0,
                    clamp_float(r, 0.0f, 1.0f)
                );

                set_pixel(
                    image, x, y, 1,
                    clamp_float(g, 0.0f, 1.0f)
                );

                set_pixel(
                    image, x, y, 2,
                    clamp_float(b, 0.0f, 1.0f)
                );
            }
        }
    }

    return 0;
}


/*
 * Set one RGB pixel to red.
 */
static void set_box_pixel(
    OecImage *image,
    int x,
    int y,
    float red,
    float green,
    float blue)
{
    size_t index;

    if (image == NULL)
        return;

    if (image->data == NULL)
        return;

    if (x < 0 ||
        x >= image->width ||
        y < 0 ||
        y >= image->height)
        return;

    index =
        (
            (size_t)y *
            (size_t)image->width +
            (size_t)x
        ) * 3;

    image->data[index + 0] = red;
    image->data[index + 1] = green;
    image->data[index + 2] = blue;
}


/*
 * Draw a bounding box.
 *
 * Coordinates are normalized:
 *
 *     cx = 0.0 ... 1.0
 *     cy = 0.0 ... 1.0
 *     w  = 0.0 ... 1.0
 *     h  = 0.0 ... 1.0
 *
 * The box is represented using center
 * coordinates, matching Dragonfly labels.
 */
void oec_image_draw_box(
    OecImage *image,
    float cx,
    float cy,
    float width,
    float height,
    int thickness,
    float red,
    float green,
    float blue)
{
    int left;
    int right;
    int top;
    int bottom;

    int t;

    if (image == NULL ||
        image->data == NULL)
        return;

    if (image->channels != 3)
        return;

    if (thickness < 1)
        thickness = 1;

    /*
     * Convert normalized center/size
     * into pixel coordinates.
     */
    left =
        (int)(
            (cx - width * 0.5f) *
            (float)image->width
        );

    right =
        (int)(
            (cx + width * 0.5f) *
            (float)image->width
        );

    top =
        (int)(
            (cy - height * 0.5f) *
            (float)image->height
        );

    bottom =
        (int)(
            (cy + height * 0.5f) *
            (float)image->height
        );

    /*
     * Clamp the box to the image.
     */
    if (left < 0)
        left = 0;

    if (right < 0)
        right = 0;

    if (top < 0)
        top = 0;

    if (bottom < 0)
        bottom = 0;

    if (left >= image->width)
        left = image->width - 1;

    if (right >= image->width)
        right = image->width - 1;

    if (top >= image->height)
        top = image->height - 1;

    if (bottom >= image->height)
        bottom = image->height - 1;

    if (right < left ||
        bottom < top)
        return;

    /*
     * Draw thickness.
     */
    for (t = 0; t < thickness; t++) {

        int x;
        int y;

        /*
         * Top edge.
         */
        for (x = left; x <= right; x++) {

            set_box_pixel(
                image,
                x,
                top + t,
                red,
                green,
                blue
            );
        }

        /*
         * Bottom edge.
         */
        for (x = left; x <= right; x++) {

            set_box_pixel(
                image,
                x,
                bottom - t,
                red,
                green,
                blue
            );
        }

        /*
         * Left edge.
         */
        for (y = top; y <= bottom; y++) {

            set_box_pixel(
                image,
                left + t,
                y,
                red,
                green,
                blue
            );
        }

        /*
         * Right edge.
         */
        for (y = top; y <= bottom; y++) {

            set_box_pixel(
                image,
                right - t,
                y,
                red,
                green,
                blue
            );
        }
    }
}




static inline void oec_image_set_pixel(
    OecImage *image,
    int x,
    int y,
    float r,
    float g,
    float b)
{
    if (!image || !image->data)
        return;

    if (x < 0 || y < 0 ||
        x >= image->width || y >= image->height)
        return;

    float *p = image->data +
        ((size_t)y * image->width + x) * image->channels;

    p[0] = r;
    p[1] = g;
    p[2] = b;
}

void oec_image_draw_text(
    OecImage *image,
    const char *text,
    float x,
    float y,
    int scale,
    float red,
    float green,
    float blue)
{
    if (!image || !image->data || !text)
        return;

    if (scale <= 0 || image->channels < 3)
        return;

    int px = (int)(x * image->width);
    int py = (int)(y * image->height);

    const int glyph_width  = 5;
    const int glyph_height = 7;
    const int glyph_spacing = 1;

    int start_x = px;

    for (size_t i = 0; text[i] != '\0'; i++) {

        unsigned char ch = (unsigned char)text[i];

        /* New line */
        if (ch == '\n') {
            px = start_x;
            py += (glyph_height + 1) * scale;
            continue;
        }

        /* Tab */
        if (ch == '\t') {
            px += 4 * (glyph_width + glyph_spacing) * scale;
            continue;
        }

        /*
         * ASCII only.
         * 0-127 are valid ASCII values.
         */
        if (ch >= 128)
            ch = ' ';
            
        const uint8_t *glyph =
            dragonfly_font_5x7[ch];
            
        for (int row = 0; row < glyph_height; row++) {

            uint8_t bits = glyph[row];

            for (int col = 0; col < glyph_width; col++) {

                if (!(bits & (1u << (4 - col))))
                    continue;

                int gx = px + col * scale;
                int gy = py + row * scale;

                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {

                        oec_image_set_pixel(
                            image,
                            gx + sx,
                            gy + sy,
                            red,
                            green,
                            blue
                        );
                    }
                }
            }
        }

        px += (glyph_width + glyph_spacing) * scale;
    }
}

void oec_image_draw_text2(
    OecImage *image,
    const char *text,
    float x,
    float y,
    int scale,
    float red,
    float green,
    float blue)
{
    if (!image || !image->data || !text)
        return;

    if (scale <= 0 || image->channels < 3)
        return;

    int px = (int)(x * image->width);
    int py = (int)(y * image->height);

    for (size_t i = 0; text[i] != '\0'; i++) {

        unsigned char ch = (unsigned char)text[i];

        /* Convert lowercase ASCII to uppercase. */
        if (ch >= 'a' && ch <= 'z')
            ch -= 'a' - 'A';

        if (ch >= 128)
            ch = ' ';

        const uint8_t *glyph =
            dragonfly_font_5x7[ch];

        for (int row = 0; row < 7; row++) {

            uint8_t bits = glyph[row];

            for (int col = 0; col < 5; col++) {

                if (!(bits & (1 << (4 - col))))
                    continue;

                int gx = px + col * scale;
                int gy = py + row * scale;

                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {

                        oec_image_set_pixel(
                            image,
                            gx + sx,
                            gy + sy,
                            red,
                            green,
                            blue
                        );
                    }
                }
            }
        }

        px += 6 * scale;
    }
}


void oec_image_draw_watermark(
    OecImage *image,
    const char *text,
    int margin,
    int scale,
    float r,
    float g,
    float b)
{
    if (!image || !text)
        return;

    int text_width = 0;

    for (const char *p = text; *p; p++)
        text_width += 6 * scale;

    text_width -= scale;  // remove final spacing

    int text_height = 7 * scale;

    int x = image->width - text_width - margin;
    int y = image->height - text_height - margin;

    oec_image_draw_text(
        image,
        text,
        (float)x / image->width,
        (float)y / image->height,
        scale,
        r, g, b
    );
}

void oec_image_draw_crosshair(
    OecImage *image,
    float x,
    float y,
    int size,
    int gap,
    int thickness,
    float red,
    float green,
    float blue)
{
    int cx = (int)(x * image->width);
    int cy = (int)(y * image->height);

    if (gap <= 0) {

    /* Horizontal */
    for (int dy = -thickness / 2; dy <= thickness / 2; dy++) {
        for (int dx = -size; dx <= size; dx++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red, green, blue
            );
        }
    }

    /* Vertical */
    for (int dx = -thickness / 2; dx <= thickness / 2; dx++) {
        for (int dy = -size; dy <= size; dy++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red, green, blue
            );
        }
    }

    return;
}
 
    
    
    /* Left arm */
    for (int dy = -thickness / 2; dy <= thickness / 2; dy++) {
        for (int dx = -size; dx < -gap; dx++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red, green, blue
            );
        }
    }

    /* Right arm */
    for (int dy = -thickness / 2; dy <= thickness / 2; dy++) {
        for (int dx = gap + 1; dx <= size; dx++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red, green, blue
            );
        }
    }

    /* Top arm */
    for (int dx = -thickness / 2; dx <= thickness / 2; dx++) {
        for (int dy = -size; dy < -gap; dy++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red, green, blue
            );
        }
    }

    /* Bottom arm */
    for (int dx = -thickness / 2; dx <= thickness / 2; dx++) {
        for (int dy = gap + 1; dy <= size; dy++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red, green, blue
            );
        }
    }
}

void oec_image_draw_crosshair2(
    OecImage *image,
    float x,
    float y,
    int size,
    int thickness,
    float red,
    float green,
    float blue)
{
    int cx = (int)(x * image->width);
    int cy = (int)(y * image->height);

    /* Horizontal */
    for (int dy = -thickness / 2; dy <= thickness / 2; dy++) {
        for (int dx = -size; dx <= size; dx++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red,
                green,
                blue
            );
        }
    }

    /* Vertical */
    for (int dx = -thickness / 2; dx <= thickness / 2; dx++) {
        for (int dy = -size; dy <= size; dy++) {

            oec_image_set_pixel(
                image,
                cx + dx,
                cy + dy,
                red,
                green,
                blue
            );
        }
    }
}

/* ============================================================
 * Saving helpers
 * ============================================================ */

static unsigned char *image_to_u8(
    const OecImage *image,
    size_t *pixel_count)
{
    unsigned char *pixels;
    size_t count;
    size_t i;

    if (image == NULL ||
        image->data == NULL ||
        pixel_count == NULL)
        return NULL;

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->channels != 3)
        return NULL;

    count =
        (size_t)image->width *
        (size_t)image->height *
        3;

    pixels =
        malloc(count);

    if (pixels == NULL)
        return NULL;

    for (i = 0; i < count; i++) {
        float value;

        value =
            clamp_float(
                image->data[i],
                0.0f,
                1.0f
            );

        pixels[i] =
            (unsigned char)lroundf(
                value * 255.0f
            );
    }

    *pixel_count = count;

    return pixels;
}


/*
 * Save a float RGB image as PNG.
 *
 * Internal:
 *
 *     float [0.0,1.0]
 *
 * PNG:
 *
 *     uint8 [0,255]
 */
 
/* ============================================================
 * PNG
 * ============================================================ */
 
int oec_image_save_png(
    const char *filename,
    const OecImage *image)
{
    unsigned char *pixels;

    size_t pixel_count;
    size_t i;

    if (filename == NULL ||
        image == NULL ||
        image->data == NULL)
        return -1;

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->channels != 3)
        return -1;

    pixel_count =
        (size_t)image->width *
        (size_t)image->height *
        3;

    pixels = malloc(pixel_count);

    if (pixels == NULL) {

        fprintf(
            stderr,
            "Failed to allocate PNG buffer\n"
        );

        return -1;
    }

    /*
     * Convert float RGB to uint8 RGB.
     */
    for (i = 0; i < pixel_count; i++) {

        float value;

        value = image->data[i];

        if (value < 0.0f)
            value = 0.0f;

        if (value > 1.0f)
            value = 1.0f;

        pixels[i] =
            (unsigned char)lroundf(
                value * 255.0f
            );
    }

    if (!stbi_write_png(
            filename,
            image->width,
            image->height,
            3,
            pixels,
            image->width * 3)) {

        fprintf(
            stderr,
            "Failed to save PNG: %s\n",
            filename
        );

        free(pixels);

        return -1;
    }

    free(pixels);

    return 0;
}

/* ============================================================
 * BMP
 * ============================================================ */
 
int oec_image_save_bmp(
	const char *filename,
	const OecImage *image)
{
	if (filename == NULL || image == NULL || image->data == NULL) {
		return -1;
	}
	
	

    size_t pixel_count =
        (size_t)image->width *
        (size_t)image->height *
        3;

    unsigned char *pixels =
        (unsigned char *)malloc(pixel_count);

    if (pixels == NULL) {
        return -1;
    }

    for (size_t i = 0; i < pixel_count; i++) {
        float value = image->data[i];

        if (value < 0.0f) {
            value = 0.0f;
        }

        if (value > 1.0f) {
            value = 1.0f;
        }

        pixels[i] = (unsigned char)lroundf(value * 255.0f);
    }

    if (!stbi_write_bmp(
        filename,
        image->width,
        image->height,
        3,
        pixels
    )) {
    	free(pixels);

        return -1;
    }

    free(pixels);
    
    return 0;
}

/* ============================================================
 * TGA
 * ============================================================ */
 
int oec_image_save_tga(
    const char *filename,
    const OecImage *image)
{
    unsigned char *pixels;

    size_t pixel_count;
    size_t i;

    if (filename == NULL ||
        image == NULL ||
        image->data == NULL)
        return -1;

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->channels != 3)
        return -1;

    pixel_count =
        (size_t)image->width *
        (size_t)image->height *
        3;

    pixels = malloc(pixel_count);

    if (pixels == NULL) {

        fprintf(
            stderr,
            "Failed to allocate TGA buffer\n"
        );

        return -1;
    }

    /*
     * Convert float RGB to uint8 RGB.
     */
    for (i = 0; i < pixel_count; i++) {

        float value;

        value = image->data[i];

        if (value < 0.0f)
            value = 0.0f;

        if (value > 1.0f)
            value = 1.0f;

        pixels[i] =
            (unsigned char)lroundf(
                value * 255.0f
            );
    }

    if (!stbi_write_tga(
            filename,
            image->width,
            image->height,
            3,
            pixels)) {

        fprintf(
            stderr,
            "Failed to save TGA: %s\n",
            filename
        );

        free(pixels);

        return -1;
    }

    free(pixels);

    return 0;
}


/* ============================================================
 * JPEG
 * ============================================================ */
 
int oec_image_save_jpg(
    const char *filename,
    const OecImage *image,
    int quality)
{
    unsigned char *pixels;

    size_t pixel_count;
    size_t i;

    if (filename == NULL ||
        image == NULL ||
        image->data == NULL)
        return -1;

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->channels != 3)
        return -1;

    if (quality < 1)
        quality = 1;

    if (quality > 100)
        quality = 100;

    pixel_count =
        (size_t)image->width *
        (size_t)image->height *
        3;

    pixels = malloc(pixel_count);

    if (pixels == NULL) {

        fprintf(
            stderr,
            "Failed to allocate JPEG buffer\n"
        );

        return -1;
    }

    /*
     * Convert float RGB to uint8 RGB.
     */
    for (i = 0; i < pixel_count; i++) {

        float value;

        value = image->data[i];

        if (value < 0.0f)
            value = 0.0f;

        if (value > 1.0f)
            value = 1.0f;

        pixels[i] =
            (unsigned char)lroundf(
                value * 255.0f
            );
    }

    if (!stbi_write_jpg(
            filename,
            image->width,
            image->height,
            3,
            pixels,
            quality)) {

        fprintf(
            stderr,
            "Failed to save JPEG: %s\n",
            filename
        );

        free(pixels);

        return -1;
    }

    free(pixels);

    return 0;
}


#include "font_8x16.h"


#define FONT_WIDTH  8
#define FONT_HEIGHT 16

void draw_char(
    uint8_t *fb,
    int fb_width,
    int fb_height,
    int x,
    int y,
    int scale,
    char c)
{
    if (!fb || scale <= 0)
        return;

    unsigned char ch = (unsigned char)c;

    const uint8_t *glyph =
        &fontdata_8x16[ch * 16];

    for (int row = 0; row < 16; row++) {

        uint8_t bits = glyph[row];

        for (int col = 0; col < 8; col++) {

            if (!(bits & (0x80u >> col)))
                continue;

            for (int sy = 0; sy < scale; sy++) {
                for (int sx = 0; sx < scale; sx++) {

                    int px = x + col * scale + sx;
                    int py = y + row * scale + sy;

                    if (px < 0 || px >= fb_width ||
                        py < 0 || py >= fb_height)
                        continue;

                    uint8_t *p =
                        &fb[(py * fb_width + px) * 3];

                    p[0] = 255;
                    p[1] = 255;
                    p[2] = 255;
                }
            }
        }
    }
}