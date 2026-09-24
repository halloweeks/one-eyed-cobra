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

OEC_TENSOR *oec_tensor_from_image(
	const OecImage *image
);

#endif
