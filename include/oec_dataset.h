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

int oec_dataset_load_image(
    const char *path,
    OEC_TENSOR *image
);

OEC_SAMPLE_DATA oec_dataset_load_label(
    const char *path
);

OEC_SAMPLE_DATA oec_dataset_load_sample(
    const OEC_SAMPLE *sample,
    OEC_TENSOR *image
);

#endif