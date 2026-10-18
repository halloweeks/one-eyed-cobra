#ifndef OEC_DATASET_H
#define OEC_DATASET_H

#include <stddef.h>
#include <sys/types.h>
#include "oec_tensor.h"

#define OEC_PATH_MAX 256

typedef struct {
    char image_path[OEC_PATH_MAX];
    char label_path[OEC_PATH_MAX];
} OEC_SAMPLE;

typedef struct {
    OEC_SAMPLE *samples;
    size_t count;
} OEC_DATASET_SPLIT;

typedef struct {
    OEC_DATASET_SPLIT train;
    OEC_DATASET_SPLIT valid;
    OEC_DATASET_SPLIT test;
} OEC_DATASET;

typedef struct {
	float *data;
	int width;
	int height;
} OEC_DATA;

#define OEC_MAX_BOXES 32

typedef struct {
    int class_id;
    float x_center;
    float y_center;
    float width;
    float height;
} OEC_BBox;

typedef struct {
    OEC_TENSOR *image;
    OEC_BBox boxes[OEC_MAX_BOXES];
    size_t box_count;
} OEC_SAMPLE_DATA;

OEC_DATASET *oec_dataset_open(const char *root);
void oec_dataset_close(OEC_DATASET *dataset);

/*
 * Load an image into a preallocated 1 (gray) or 3 (RGB) channel tensor.
 * The image is converted to the tensor's channel count and
 * nearest-neighbor resized to the tensor's width/height.
 */
int oec_dataset_load_image(
    const char *path,
    OEC_TENSOR *image
);

/*
 * Inspect the image headers of every split and report the channel
 * count a model for this dataset should use:
 *
 *   1  - every image is grayscale
 *   3  - at least one image is color (gray images are replicated)
 *  -1  - no readable images
 *
 * Only headers are read, images are not decoded.
 */
int oec_dataset_detect_channels(
    const OEC_DATASET *dataset
);

OEC_SAMPLE_DATA oec_dataset_load_label(
    const char *path
);

OEC_SAMPLE_DATA oec_dataset_load_sample(
    const OEC_SAMPLE *sample,
    OEC_TENSOR *image
);

#endif