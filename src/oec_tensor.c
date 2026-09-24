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

/*
int oec_image_from_tensor(
    OecImage *image,
    const OEC_TENSOR *tensor)
{
    if (image == NULL || tensor == NULL)
        return -1;

    if (tensor->data == NULL)
        return -1;

    if (tensor->c != 3)
        return -1;

    image->width = tensor->w;
    image->height = tensor->h;
    image->channels = 3;

    size_t count =
        (size_t)tensor->w *
        (size_t)tensor->h *
        3;

    image->data = malloc(
        count * sizeof(float)
    );

    if (image->data == NULL)
        return -1;

    for (int y = 0; y < tensor->h; y++) {
        for (int x = 0; x < tensor->w; x++) {

            size_t p =
                (size_t)y * tensor->w + x;

            image->data[p * 3 + 0] =
                tensor->data[0 * tensor->w * tensor->h + p];

            image->data[p * 3 + 1] =
                tensor->data[1 * tensor->w * tensor->h + p];

            image->data[p * 3 + 2] =
                tensor->data[2 * tensor->w * tensor->h + p];
        }
    }

    return 0;
}

int oec_image_to_tensor(
    const OecImage *image,
    OEC_TENSOR *tensor)
{
    if (image == NULL || tensor == NULL)
        return -1;

    if (image->data == NULL)
        return -1;

    if (image->channels != 3)
        return -1;

    if (tensor->w != 128 ||
        tensor->h != 128 ||
        tensor->c != 3)
        return -1;

    for (int y = 0; y < 128; y++) {

        int sy = y * image->height / 128;

        for (int x = 0; x < 128; x++) {

            int sx = x * image->width / 128;

            size_t src =
                ((size_t)sy * image->width + sx) * 3;

            size_t dst =
                (size_t)y * 128 + x;

            tensor->data[
                0 * 128 * 128 + dst
            ] = image->data[src + 0];

            tensor->data[
                1 * 128 * 128 + dst
            ] = image->data[src + 1];

            tensor->data[
                2 * 128 * 128 + dst
            ] = image->data[src + 2];
        }
    }

    return 0;
}
*/


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