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

float oec_rgb_to_gray(float r, float g, float b)
{
	return 0.299f * r + 0.587f * g + 0.114f * b;
}

OEC_TENSOR *oec_tensor_from_image_channels(const OecImage *image, int channels)
{
	if (image == NULL || image->data == NULL) {
		return NULL;
	}
	
	if (image->width <= 0 || image->height <= 0 || image->channels <= 0) {
		return NULL;
	}
	
	if (channels != 1 && channels != 3) {
		return NULL;
	}
	
	OEC_TENSOR *tensor = oec_tensor_create(image->width, image->height, channels);
	
	if (tensor == NULL) {
		return NULL;
	}
	
	size_t hw = (size_t)tensor->w * (size_t)tensor->h;
	size_t stride = (size_t)image->channels;
	
	// 1 (gray) or 2 (gray + alpha) channel sources are gray, 3+ are RGB(A).
	int src_is_rgb = image->channels >= 3;
	
	for (size_t p = 0; p < hw; p++) {
		const float *src = &image->data[p * stride];
		
		if (channels == 3) {
			if (src_is_rgb) {
				tensor->data[0 * hw + p] = src[0]; // R plane
				tensor->data[1 * hw + p] = src[1]; // G plane
				tensor->data[2 * hw + p] = src[2]; // B plane
			} else {
				tensor->data[0 * hw + p] = src[0];
				tensor->data[1 * hw + p] = src[0];
				tensor->data[2 * hw + p] = src[0];
			}
		} else {
			if (src_is_rgb) {
				tensor->data[p] = oec_rgb_to_gray(src[0], src[1], src[2]);
			} else {
				tensor->data[p] = src[0];
			}
		}
	}
	
	return tensor;
}

OEC_TENSOR *oec_tensor_from_image(const OecImage *image) {
	if (image == NULL || image->data == NULL) {
		return NULL;
	}
	
	return oec_tensor_from_image_channels(image, image->channels >= 3 ? 3 : 1);
}

void oec_tensor_free(OEC_TENSOR *tensor)
{
	if (tensor == NULL) {
		return;
	}
	
	free(tensor->data);
	free(tensor);
}