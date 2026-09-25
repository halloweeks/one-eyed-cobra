#ifndef OEC_ACTIVATION_H
#define OEC_ACTIVATION_H

#include "oec_tensor.h"

typedef enum {
	OEC_ACTIVATION_RELU = 0,
	OEC_ACTIVATION_SIGMOID,
	OEC_ACTIVATION_TANH,
	OEC_ACTIVATION_LEAKY_RELU
} OEC_ACTIVATION_TYPE;

typedef struct {
	OEC_ACTIVATION_TYPE type;
	union {
		float alpha; /* Leaky ReLU */
	} param;
} OEC_ACTIVATION;

/* Activation */
int oec_activation_forward(
	const OEC_ACTIVATION *activation,
	const OEC_TENSOR *input,
	OEC_TENSOR *output
);

int oec_activation_backward(
	const OEC_ACTIVATION *activation,
	const OEC_TENSOR *input,
	const OEC_TENSOR *output,
	const OEC_TENSOR *grad_output,
	OEC_TENSOR *grad_input
);

#endif