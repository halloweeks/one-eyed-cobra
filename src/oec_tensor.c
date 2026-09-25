#include <stdlib.h>
#include "oec_tensor.h"

OEC_TENSOR *oec_tensor_create(int w, int h, int c)
{
	if (w <= 0 || h <= 0 || c <= 0) {
		return NULL;
	}
	
	OEC_TENSOR *tensor = malloc(sizeof(*tensor));
	
	if (tensor == NULL) {
		return NULL;
	}
	
	size_t count = (size_t)w * (size_t)h * (size_t)c;
	
	tensor->data = calloc(count, sizeof(float));
	
	if (tensor->data == NULL) {
		free(tensor);
		return NULL;
	}
	
	tensor->w = w;
	tensor->h = h;
	tensor->c = c;
	
	return tensor;
}

OEC_TENSOR *oec_tensor_from_image(const OecImage *image) {
	if (image == NULL || image->data == NULL) {
		return NULL;
	}
	
	if (image->width <= 0 || image->height <= 0 || image->channels <= 0) {
		return NULL;
	}
	
	OEC_TENSOR *tensor = malloc(sizeof(*tensor));
	
	if (tensor == NULL) {
		return NULL;
	}
	
	tensor->w = image->width;
	tensor->h = image->height;
	tensor->c = image->channels;
	
	size_t hw = (size_t)tensor->w * (size_t)tensor->h;
	
	tensor->data = malloc(hw * tensor->c * sizeof(float));
	
	if (tensor->data == NULL) {
		free(tensor);
		return NULL;
	}
	
	for (size_t p = 0; p < hw; p++) {
		tensor->data[0 * hw + p] = image->data[p * image->channels + 0]; // R plane
		tensor->data[1 * hw + p] = image->data[p * image->channels + 1]; // G plane
		tensor->data[2 * hw + p] = image->data[p * image->channels + 2]; // B plane
	}
	
	return tensor;
}

void oec_tensor_free(OEC_TENSOR *tensor)
{
	if (tensor == NULL) {
		return;
	}
	
	free(tensor->data);
	free(tensor);
}