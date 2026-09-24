#include <stdlib.h>
#include "oec_layer.h"

OEC_LAYER *oec_layer_activation_create(OEC_ACTIVATION_TYPE type)
{
    OEC_LAYER *layer =
        calloc(1, sizeof(OEC_LAYER));

    if (layer == NULL)
        return NULL;

    layer->type = OEC_LAYER_ACTIVATION;

    layer->param.activation =
        calloc(1, sizeof(OEC_ACTIVATION));

    if (layer->param.activation == NULL)
    {
        free(layer);
        return NULL;
    }

    layer->param.activation->type = type;

    return layer;
}

OEC_LAYER *oec_layer_conv_create(
    int in_channels,
    int out_channels,
    int kernel,
    int stride)
{
    OEC_LAYER *layer = calloc(1, sizeof(OEC_LAYER));

    if (layer == NULL)
        return NULL;

    layer->type = OEC_LAYER_CONV;

    layer->param.conv = oec_conv_create(
        in_channels,
        out_channels,
        kernel,
        stride
    );

    if (layer->param.conv == NULL) {
        free(layer);
        return NULL;
    }

    return layer;
}


OEC_LAYER *oec_layer_global_avg_pool_create(void)
{
	OEC_LAYER *layer = calloc(1, sizeof(OEC_LAYER));
	
	if (layer == NULL) {
		return NULL;
	}
	
	layer->type = OEC_LAYER_GLOBAL_AVG_POOL;
	
	return layer;
}

static int oec_global_avg_pool_forward(
    const OEC_TENSOR *input,
    OEC_TENSOR *output)
{
    if (input == NULL || output == NULL)
        return -1;

    if (output->w != 1 ||
        output->h != 1 ||
        output->c != input->c)
        return -1;

    const int spatial = input->w * input->h;

    if (spatial <= 0)
        return -1;

    for (int c = 0; c < input->c; c++) {

        float sum = 0.0f;

        for (int y = 0; y < input->h; y++) {
            for (int x = 0; x < input->w; x++) {

                const int index =
                    (c * input->h + y) * input->w + x;

                sum += input->data[index];
            }
        }

        output->data[c] = sum / (float)spatial;
    }

    return 0;
}

static int oec_global_avg_pool_backward(
    const OEC_TENSOR *input,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input)
{
    if (input == NULL ||
        grad_output == NULL ||
        grad_input == NULL)
        return -1;

    if (grad_output->w != 1 ||
        grad_output->h != 1 ||
        grad_output->c != input->c)
        return -1;

    if (grad_input->w != input->w ||
        grad_input->h != input->h ||
        grad_input->c != input->c)
        return -1;

    const int spatial = input->w * input->h;

    if (spatial <= 0)
        return -1;

    for (int c = 0; c < input->c; c++) {

        const float gradient =
            grad_output->data[c] / (float)spatial;

        for (int y = 0; y < input->h; y++) {
            for (int x = 0; x < input->w; x++) {

                const int index =
                    (c * input->h + y) * input->w + x;

                grad_input->data[index] = gradient;
            }
        }
    }

    return 0;
}

/*
OEC_LAYER *oec_layer_relu_create(void)
{
    OEC_LAYER *layer = calloc(1, sizeof(OEC_LAYER));

    if (layer == NULL)
        return NULL;

    layer->type = OEC_LAYER_RELU;

    return layer;
}*/

void oec_layer_free(OEC_LAYER *layer)
{
    if (layer == NULL)
        return;

    if (layer->type == OEC_LAYER_CONV)
        oec_conv_free(layer->param.conv);

    free(layer);
}

int oec_layer_forward(OEC_LAYER *layer, const OEC_TENSOR *input, OEC_TENSOR *output)
{
	if (layer == NULL || input == NULL || output == NULL)
		return -1;
	
	switch (layer->type) {
		case OEC_LAYER_CONV:
			if (layer->param.conv == NULL)
				return -1;
			
			return oec_conv_forward(
				layer->param.conv,
				input,
				output
			);
		case OEC_LAYER_ACTIVATION:
			if (layer->param.activation == NULL)
				return -1;
			
			return oec_activation_forward(layer->param.activation, input, output);
		case OEC_LAYER_GLOBAL_AVG_POOL: 
			return oec_global_avg_pool_forward(input, output);
		default:
			return -1;
	}
}

int oec_layer_backward(
    OEC_LAYER *layer,
    const OEC_TENSOR *input,
    const OEC_TENSOR *output,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input)
{
    if (layer == NULL ||
        input == NULL ||
        grad_output == NULL ||
        grad_input == NULL)
        return -1;

    switch (layer->type) {

    case OEC_LAYER_CONV:
        if (layer->param.conv == NULL)
            return -1;

        return oec_conv_backward(
            layer->param.conv,
            input,
            grad_output,
            grad_input
        );

    case OEC_LAYER_ACTIVATION:
        if (layer->param.activation == NULL)
            return -1;

        return oec_activation_backward(
            layer->param.activation,
            input,
            output,
            grad_output,
            grad_input
        );
    case OEC_LAYER_GLOBAL_AVG_POOL: 
    	return oec_global_avg_pool_backward(
        input,
        grad_output,
        grad_input
    );
    default:
        return -1;
    }
}