#include <stdlib.h>
#include <math.h>
#include "oec_conv.h"

OEC_CONV *oec_conv_create(int in_channels, int out_channels, int kernel, int stride)
{
	if (in_channels <= 0 || out_channels <= 0 || kernel <= 0 || stride <= 0) {
		return NULL;
	}
	
	OEC_CONV *conv = malloc(sizeof(*conv));
	
	if (conv == NULL) {
		return NULL;
	}
	
	conv->in_channels = in_channels;
	conv->out_channels = out_channels;
	conv->kernel = kernel;
	conv->stride = stride;
	
	size_t weight_count = (size_t)out_channels * (size_t)in_channels * (size_t)kernel * (size_t)kernel;
	
	conv->weights = calloc(weight_count, sizeof(float));
	conv->bias = calloc((size_t)out_channels, sizeof(float));
	
	conv->grad_weights = calloc(weight_count, sizeof(float));
	conv->grad_bias = calloc((size_t)out_channels, sizeof(float));
	
	if (conv->weights == NULL || conv->bias == NULL || conv->grad_weights == NULL || conv->grad_bias == NULL) {
		free(conv->weights);
		free(conv->bias);
		free(conv->grad_weights);
		free(conv->grad_bias);
		free(conv);
		return NULL;
	}
	
	
	/*
     * He initialization.
     *
     * fan_in = number of inputs to one output neuron.
     */
    int fan_in =
        in_channels *
        kernel *
        kernel;

    float scale =
        sqrtf(2.0f / (float)fan_in);

    for (size_t i = 0; i < weight_count; i++)
    {
        float r =
            (float)rand() /
            (float)RAND_MAX;

        /* [0, 1] -> [-1, 1] */
        r = r * 2.0f - 1.0f;

        conv->weights[i] = r * scale;
    }

    /*
     * Bias starts at zero.
     */
    for (int i = 0; i < out_channels; i++) {
        conv->bias[i] = 0.0f;
	}
	
	return conv;
}

void oec_conv_free(OEC_CONV *conv)
{
    if (conv == NULL)
        return;

    free(conv->weights);
    free(conv->bias);
    
    free(conv->grad_weights);
    free(conv->grad_bias);
    free(conv);
}

int oec_conv_forward(
	const OEC_CONV *conv,
    const OEC_TENSOR *input,
    OEC_TENSOR *output)
{
    if (input == NULL ||
        conv == NULL ||
        output == NULL)
        return -1;

    int output_w =
        (input->w - conv->kernel) /
        conv->stride + 1;

    int output_h =
        (input->h - conv->kernel) /
        conv->stride + 1;

    if (output_w <= 0 || output_h <= 0)
        return -1;

    if (output->w != output_w ||
        output->h != output_h ||
        output->c != conv->out_channels)
        return -1;

    for (int oc = 0; oc < conv->out_channels; oc++) {

        for (int oy = 0; oy < output_h; oy++) {

            for (int ox = 0; ox < output_w; ox++) {

                float sum = conv->bias[oc];

                for (int ic = 0;
                     ic < conv->in_channels;
                     ic++) {

                    for (int ky = 0;
                         ky < conv->kernel;
                         ky++) {

                        for (int kx = 0;
                             kx < conv->kernel;
                             kx++) {

                            int ix =
                                ox * conv->stride + kx;

                            int iy =
                                oy * conv->stride + ky;

                            size_t input_index =
                                ((size_t)ic *
                                 input->h *
                                 input->w) +
                                ((size_t)iy *
                                 input->w) +
                                (size_t)ix;

                            size_t weight_index =
                                (((size_t)oc *
                                  conv->in_channels +
                                  (size_t)ic) *
                                 conv->kernel *
                                 conv->kernel) +
                                ((size_t)ky *
                                 conv->kernel) +
                                (size_t)kx;

                            sum +=
                                input->data[input_index] *
                                conv->weights[weight_index];
                        }
                    }
                }

                size_t output_index =
                    ((size_t)oc *
                     output->h *
                     output->w) +
                    ((size_t)oy *
                     output->w) +
                    (size_t)ox;

                output->data[output_index] = sum;
            }
        }
    }

    return 0;
}

int oec_conv_backward(
	OEC_CONV *conv,
    const OEC_TENSOR *input,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input)
{
    if (input == NULL ||
        conv == NULL ||
        grad_output == NULL ||
        grad_input == NULL)
        return -1;

    int output_w =
        (input->w - conv->kernel) /
        conv->stride + 1;

    int output_h =
        (input->h - conv->kernel) /
        conv->stride + 1;

    if (grad_output->w != output_w ||
        grad_output->h != output_h ||
        grad_output->c != conv->out_channels)
        return -1;

    if (grad_input->w != input->w ||
        grad_input->h != input->h ||
        grad_input->c != input->c)
        return -1;

    size_t weight_count =
        (size_t)conv->out_channels *
        (size_t)conv->in_channels *
        (size_t)conv->kernel *
        (size_t)conv->kernel;

    /*
     * Gradients accumulate during backward.
     * Clear them before calculating this sample.
     */
    for (size_t i = 0; i < weight_count; i++)
        conv->grad_weights[i] = 0.0f;

    for (int oc = 0; oc < conv->out_channels; oc++)
        conv->grad_bias[oc] = 0.0f;

    size_t input_count =
        (size_t)input->w *
        (size_t)input->h *
        (size_t)input->c;

    for (size_t i = 0; i < input_count; i++)
        grad_input->data[i] = 0.0f;

    for (int oc = 0; oc < conv->out_channels; oc++) {

        for (int oy = 0; oy < output_h; oy++) {

            for (int ox = 0; ox < output_w; ox++) {

                size_t output_index =
                    ((size_t)oc *
                     output_h *
                     output_w) +
                    ((size_t)oy *
                     output_w) +
                    (size_t)ox;

                float go =
                    grad_output->data[output_index];

                /*
                 * Bias gradient
                 */
                conv->grad_bias[oc] += go;

                for (int ic = 0;
                     ic < conv->in_channels;
                     ic++) {

                    for (int ky = 0;
                         ky < conv->kernel;
                         ky++) {

                        for (int kx = 0;
                             kx < conv->kernel;
                             kx++) {

                            int ix =
                                ox * conv->stride + kx;

                            int iy =
                                oy * conv->stride + ky;

                            size_t input_index =
                                ((size_t)ic *
                                 input->h *
                                 input->w) +
                                ((size_t)iy *
                                 input->w) +
                                (size_t)ix;

                            size_t weight_index =
                                (((size_t)oc *
                                  conv->in_channels +
                                  (size_t)ic) *
                                 conv->kernel *
                                 conv->kernel) +
                                ((size_t)ky *
                                 conv->kernel) +
                                (size_t)kx;

                            /*
                             * dL/dW
                             */
                            conv->grad_weights[weight_index] +=
                                input->data[input_index] * go;

                            /*
                             * dL/dInput
                             */
                            grad_input->data[input_index] +=
                                conv->weights[weight_index] * go;
                        }
                    }
                }
            }
        }
    }

    return 0;
}