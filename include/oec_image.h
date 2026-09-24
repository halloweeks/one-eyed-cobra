#ifndef OEC_IMAGE_H
#define OEC_IMAGE_H

#include <stddef.h>

#define OEC_IMAGE_WIDTH    640
#define OEC_IMAGE_HEIGHT   640
#define OEC_IMAGE_CHANNELS 3

typedef struct {
    int width;
    int height;
    int channels;

    float *data;
} OecImage;


/*
 * Free image memory.
 */
int oec_image_create(
    OecImage *image,
    int width,
    int height,
    int channels
);

void oec_image_free(
    OecImage *image
);


/*
 * Load an image from disk.
 *
 * The image is converted to RGB and normalized
 * from [0, 255] to [0.0, 1.0].
 */
int oec_image_load(
    const char *filename,
    OecImage *image
);


/*
 * Geometry
 */

int oec_image_crop(
    const OecImage *source,
    OecImage *destination,
    int x,
    int y,
    int width,
    int height
);

int oec_image_resize(
    const OecImage *source,
    OecImage *destination,
    int width,
    int height
);

int oec_image_resize_letterbox(
    const OecImage *source,
    OecImage *destination
);

int oec_image_flip_horizontal(
    OecImage *image
);

int oec_image_flip_vertical(
    OecImage *image
);

int oec_image_rotate_90(
    const OecImage *source,
    OecImage *destination
);


/*
 * Color / brightness
 */

int oec_image_adjust_brightness(
    OecImage *image,
    float factor
);

int oec_image_adjust_contrast(
    OecImage *image,
    float factor
);

int oec_image_adjust_saturation(
    OecImage *image,
    float factor
);

int oec_image_adjust_hue(
    OecImage *image,
    float degrees
);

int oec_image_adjust_gamma(
    OecImage *image,
    float gamma
);

int oec_image_adjust_temperature(
    OecImage *image,
    float amount
);

int oec_image_grayscale(
    OecImage *image
);

int oec_image_invert(
    OecImage *image
);


/*
 * Image effects
 */

int oec_image_blur(
    OecImage *image,
    int radius
);

int oec_image_sharpen(
    OecImage *image,
    float amount
);

int oec_image_add_noise(
    OecImage *image,
    float amount
);

int oec_image_pixelate(
    OecImage *image,
    int block_size
);


/*
 * Histogram / tonal adjustment
 */

int oec_image_auto_contrast(
    OecImage *image
);

int oec_image_histogram_equalize(
    OecImage *image
);



/*
 * Draw a normalized bounding box on an image.
 *
 * cx     = normalized center X
 * cy     = normalized center Y
 * width  = normalized box width
 * height = normalized box height
 *
 * thickness = box line thickness in pixels.
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
    float blue
);

void oec_image_draw_text(
    OecImage *image,
    const char *text,
    float x,
    float y,
    int scale,
    float red,
    float green,
    float blue
);

void oec_image_draw_crosshair(
    OecImage *image,
    float x,
    float y,
    int size,
    int gap,
    int thickness,
    float red,
    float green,
    float blue
);

#include <stdint.h>
#include <stddef.h>

static const uint8_t dragonfly_font_5x7[128][7] = {
    [' '] = {0, 0, 0, 0, 0, 0, 0},

    ['0'] = {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
    ['1'] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
    ['2'] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
    ['3'] = {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E},
    ['4'] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
    ['5'] = {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E},
    ['6'] = {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
    ['7'] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
    ['8'] = {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
    ['9'] = {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},

    ['A'] = {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
    ['B'] = {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},
    ['C'] = {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},
    ['D'] = {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E},
    ['E'] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},
    ['F'] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
    ['G'] = {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},
    ['H'] = {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
    ['I'] = {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
    ['J'] = {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E},
    ['K'] = {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
    ['L'] = {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
    ['M'] = {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},
    ['N'] = {0x11, 0x19, 0x19, 0x15, 0x13, 0x13, 0x11},
    ['O'] = {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    ['P'] = {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
    ['Q'] = {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},
    ['R'] = {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
    ['S'] = {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},
    ['T'] = {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
    ['U'] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    ['V'] = {0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04},
    ['W'] = {0x11, 0x11, 0x15, 0x15, 0x15, 0x1B, 0x11},
    ['X'] = {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
    ['Y'] = {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},
    ['Z'] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},
    [':'] = {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00}
};


/*
 * Saving
 */
int oec_image_save_png(
    const char *filename,
    const OecImage *image
);

int oec_image_save_bmp(
    const char *filename,
    const OecImage *image
);

int oec_image_save_tga(
    const char *filename,
    const OecImage *image
);

int oec_image_save_jpg(
    const char *filename,
    const OecImage *image,
    int quality
);


void oec_image_draw_watermark(
    OecImage *image,
    const char *text,
    int margin,
    int scale,
    float r,
    float g,
    float b
);


#endif