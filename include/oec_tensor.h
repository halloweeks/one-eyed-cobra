/*
 * One-Eyed Cobra (OEC)
 *
 * Lightweight object detection framework designed for
 * resource-constrained SoC systems.
 *
 * Author: Hallo Weeks
 * Copyright (C) 2026 Hallo Weeks
 *
 * Contact:
 * Email:    halloweeks@gmail.com
 * Telegram: @halloweeks
 * GitHub:   @halloweeks
 *
 * This software is provided as-is, without warranty of any kind.
 */
#ifndef OEC_TENSOR_H
#define OEC_TENSOR_H

#include <stddef.h>
#include "oec_image.h"

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} OEC_TENSOR;

/* Tensor */
OEC_TENSOR *oec_tensor_create(
	int w,
	int h,
	int c
);

void oec_tensor_free(
	OEC_TENSOR *tensor
);

/*
 * Build a CHW tensor from an image, keeping its color layout:
 * 1 or 2 channels (gray / gray+alpha) -> 1 plane,
 * 3 or 4 channels (RGB / RGBA)        -> 3 planes.
 */
OEC_TENSOR *oec_tensor_from_image(
	const OecImage *image
);

/*
 * Build a CHW tensor with an explicit channel count (1 or 3).
 *
 * RGB -> 1 channel : luminance (see oec_rgb_to_gray)
 * gray -> 3 channel: the gray plane is replicated
 *
 * Use this with model->input->c so the image always matches
 * the channel count the model was trained with.
 */
OEC_TENSOR *oec_tensor_from_image_channels(
	const OecImage *image,
	int channels
);

/*
 * RGB -> gray, used by both training (dataset loader) and
 * inference so both see identical pixel values.
 */
float oec_rgb_to_gray(
	float r,
	float g,
	float b
);

#endif
