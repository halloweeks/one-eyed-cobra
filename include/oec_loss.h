#ifndef OEC_LOSS_H
#define OEC_LOSS_H

#include "oec_tensor.h"
#include "oec_target.h"

/*
typedef struct {
    float confidence_weight;
    float box_weight;
} OEC_LOSS;
*/

typedef struct {
	float object_weight;
	float no_object_weight;
	float box_weight;
} OEC_LOSS;

/* Loss */
OEC_LOSS *oec_loss_create(
	float object_weight,
	float no_object_weight,
	float box_weight
);

/*
OEC_LOSS *oec_loss_create(
    float confidence_weight,
    float box_weight
);
*/

void oec_loss_free(
    OEC_LOSS *loss
);

float oec_loss_forward(
	const OEC_LOSS *loss,
	const OEC_TENSOR *prediction,
	const OEC_TARGET *target
);

int oec_loss_backward(
	const OEC_LOSS *loss,
	const OEC_TENSOR *prediction,
	const OEC_TARGET *target,
	OEC_TENSOR *grad
);

#endif