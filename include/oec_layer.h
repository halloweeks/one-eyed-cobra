#ifndef OEC_LAYER_H
#define OEC_LAYER_H

#include "oec_tensor.h"
#include "oec_conv.h"
#include "oec_activation.h"

typedef enum {
	OEC_LAYER_CONV = 0,
	OEC_LAYER_ACTIVATION,
	OEC_LAYER_GLOBAL_AVG_POOL,
	OEC_LAYER_GAP
} OEC_LAYER_TYPE;

typedef struct {
    OEC_LAYER_TYPE type;

    union {
        OEC_CONV *conv;
        OEC_ACTIVATION *activation;
    } param;

} OEC_LAYER;


OEC_LAYER *oec_layer_activation_create(
	OEC_ACTIVATION_TYPE type
);

OEC_LAYER *oec_layer_conv_create(
	int in_channels,
	int out_channels,
	int kernel,
	int stride
);

OEC_LAYER *oec_layer_global_avg_pool_create(
	void
);
 

/* Layer */
int oec_layer_forward(
    OEC_LAYER *layer,
    const OEC_TENSOR *input,
    OEC_TENSOR *output
);

int oec_layer_backward(
    OEC_LAYER *layer,
    const OEC_TENSOR *input,
    const OEC_TENSOR *output,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input);
    
void oec_layer_free(
    OEC_LAYER *layer
);

OEC_LAYER *oec_layer_create_conv(
    int in_channels,
    int out_channels,
    int kernel,
    int stride
);

OEC_LAYER *oec_layer_create_relu(void);

#endif