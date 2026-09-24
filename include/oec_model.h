#ifndef OEC_MODEL_H
#define OEC_MODEL_H

#include <stddef.h>
#include "oec_layer.h"

typedef struct {
    float confidence;
    float x;
    float y;
    float width;
    float height;
} OEC_PREDICTION;

typedef struct {
    OEC_LAYER **layers;

    /* Activation produced by each layer */
    OEC_TENSOR **outputs;

    /* Gradient entering each layer */
    OEC_TENSOR **grads;

    /* Input supplied to the model */
    OEC_TENSOR *input;

    int count;
    int capacity;
} OEC_MODEL;

/* Model */

OEC_MODEL *oec_model_create(
    int capacity
);

int oec_model_build(
    OEC_MODEL *model,
    int input_w,
    int input_h,
    int input_c
);

int oec_model_save(
	const OEC_MODEL *model,
	const char *path
);

OEC_MODEL *oec_model_load(
	const char *path
);

int oec_model_predict(
	OEC_MODEL *model,
	const OEC_TENSOR *input,
	OEC_PREDICTION *prediction
);


int oec_model_add_layer(
    OEC_MODEL *model,
    OEC_LAYER *layer
);

int oec_model_forward(
    OEC_MODEL *model,
    const OEC_TENSOR *input,
    OEC_TENSOR *output
);

int oec_model_backward(
    OEC_MODEL *model,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input
);

void oec_model_zero_grad(OEC_MODEL *model);

void oec_model_sgd_step(
    OEC_MODEL *model,
    float learning_rate
);

void oec_model_free(
    OEC_MODEL *model
);

#endif
