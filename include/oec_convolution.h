#ifndef OEC_CONV_H
#define OEC_CONV_H

#include "oec_tensor.h"

typedef struct {
	int in_channels;
	int out_channels;
	int kernel;
	int stride;
	
	float *weights;
	float *bias;
	
	float *grad_weights;
	float *grad_bias;
} OEC_CONV;

/* Convolution */
OEC_CONV *oec_conv_create(
    int in_channels,
    int out_channels,
    int kernel,
    int stride
);

void oec_conv_free(
	OEC_CONV *conv
);

int oec_conv_forward(
	const OEC_CONV *conv,
    const OEC_TENSOR *input,
    OEC_TENSOR *output
);

int oec_conv_backward(
	OEC_CONV *conv,
    const OEC_TENSOR *input,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input
);

#endif