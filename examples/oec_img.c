#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "oec.h"

int main() {
	OecImage image;
	
	if (oec_image_load("prediction/predict_00004.png", &image) != 0) {
		fprintf(stderr, "Failed to load input image\n");
		return EXIT_FAILURE;
	}
	
	printf("img: %ux%ux%u\n", image.width, image.height, image.channels);
	
	oec_image_free(&image);
}