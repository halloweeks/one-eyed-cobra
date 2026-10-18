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

/*
 * Multi-object versions. Targets are assigned to cells with
 * oec_target_assign(); every assigned cell is positive and all other cells
 * are negative. Positive and negative terms are averaged separately,
 * as in the single-target loss, so with exactly one target the result is
 * identical to oec_loss_forward / oec_loss_backward. With no targets
 * (empty image) every cell is a negative.
 */
float oec_loss_forward_multi(
	const OEC_LOSS *loss,
	const OEC_TENSOR *prediction,
	const OEC_TARGETS *targets
);

int oec_loss_backward_multi(
	const OEC_LOSS *loss,
	const OEC_TENSOR *prediction,
	const OEC_TARGETS *targets,
	OEC_TENSOR *grad
);

#endif