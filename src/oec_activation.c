#include "oec_activation.h"
#include <math.h>

/* ReLU */
static void oec_relu_forward(const OEC_TENSOR *input, OEC_TENSOR *output) {
	size_t count = (size_t)input->w * (size_t)input->h * (size_t)input->c;
	
	for (size_t i = 0; i < count; i++) {
		output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
	}
}

static void oec_relu_backward(const OEC_TENSOR *input, const OEC_TENSOR *grad_output, OEC_TENSOR *grad_input) {
	size_t count = (size_t)input->w * (size_t)input->h * (size_t)input->c;
	
	for (size_t i = 0; i < count; i++) {
		grad_input->data[i] = input->data[i] > 0.0f ? grad_output->data[i] : 0.0f;
	}
}

/* Sigmoid */
static void oec_sigmoid_forward(const OEC_TENSOR *input, OEC_TENSOR *output) {
	size_t count = (size_t)input->w * (size_t)input->h * (size_t)input->c;
	
	for (size_t i = 0; i < count; i++) {
		output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
	}
}

static void oec_sigmoid_backward(const OEC_TENSOR *output, const OEC_TENSOR *grad_output, OEC_TENSOR *grad_input) {
	size_t count = (size_t)output->w * (size_t)output->h * (size_t)output->c;
	
	for (size_t i = 0; i < count; i++) {
		float y = output->data[i];
		
		grad_input->data[i] = grad_output->data[i] * y * (1.0f - y);
	}
}

/* Tanh */
static void oec_tanh_forward(const OEC_TENSOR *input, OEC_TENSOR *output) {
	size_t count = (size_t)input->w * (size_t)input->h * (size_t)input->c;
	
	for (size_t i = 0; i < count; i++) {
		output->data[i] = tanhf(input->data[i]);
	}
}

static void oec_tanh_backward(const OEC_TENSOR *output, const OEC_TENSOR *grad_output, OEC_TENSOR *grad_input) {
	size_t count = (size_t)output->w * (size_t)output->h * (size_t)output->c;
	
	for (size_t i = 0; i < count; i++) {
		float y = output->data[i];
		
		grad_input->data[i] = grad_output->data[i] * (1.0f - y * y);
	}
}

/* Leaky ReLU */
static void oec_leaky_relu_forward(const OEC_ACTIVATION *activation, const OEC_TENSOR *input, OEC_TENSOR *output) {
	size_t count = (size_t)input->w * (size_t)input->h * (size_t)input->c;
	
	float alpha = activation->param.alpha;
	
	for (size_t i = 0; i < count; i++) {
		output->data[i] = input->data[i] > 0.0f ? input->data[i] : alpha * input->data[i];
	}
}

static void oec_leaky_relu_backward(const OEC_ACTIVATION *activation, const OEC_TENSOR *input, const OEC_TENSOR *grad_output, OEC_TENSOR *grad_input) {
	size_t count = (size_t)input->w * (size_t)input->h * (size_t)input->c;
	
	float alpha = activation->param.alpha;
	
	for (size_t i = 0; i < count; i++) {
		grad_input->data[i] = grad_output->data[i] * (input->data[i] > 0.0f ? 1.0f : alpha);
	}
}

/* Activation dispatcher */
int oec_activation_forward(const OEC_ACTIVATION *activation, const OEC_TENSOR *input, OEC_TENSOR *output)
{
	switch (activation->type) {
		case OEC_ACTIVATION_RELU:
			oec_relu_forward(input, output);
			break;
		case OEC_ACTIVATION_SIGMOID:
			oec_sigmoid_forward(input, output);
			break;
		case OEC_ACTIVATION_TANH:
			oec_tanh_forward(input, output);
			break;
		case OEC_ACTIVATION_LEAKY_RELU:
			oec_leaky_relu_forward(activation, input, output);
			break;
		default:
			return -1;
	}
	
	return 0;
}

int oec_activation_backward(const OEC_ACTIVATION *activation, const OEC_TENSOR *input, const OEC_TENSOR *output, const OEC_TENSOR *grad_output, OEC_TENSOR *grad_input)
{
	switch (activation->type) {
		case OEC_ACTIVATION_RELU: 
			oec_relu_backward(input, grad_output, grad_input);
			break;
		case OEC_ACTIVATION_SIGMOID:
			oec_sigmoid_backward(output, grad_output, grad_input);
			break;
		case OEC_ACTIVATION_TANH:
			oec_tanh_backward(output, grad_output, grad_input);
			break;
		case OEC_ACTIVATION_LEAKY_RELU:
			oec_leaky_relu_backward(activation, input, grad_output, grad_input);
			break;
		default:
			return -1;
	}
	
	return 0;
}