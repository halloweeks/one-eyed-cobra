/*
 * One-Eyed Cobra (OEC)
 *
 * A dedicated, single-object detection engine written in pure C for
 * resource-constrained embedded CPUs / SoCs. Not a general-purpose,
 * multi-class detector like YOLO's 80-class COCO models — OEC is built
 * to find one specific kind of object per frame, kept small and fast
 * enough to run directly on the target device with no external ML
 * runtime and no GPU dependency.
 *
 * Author:    Hallo Weeks
 * Copyright: (C) 2026 Hallo Weeks
 *
 * Contact:
 *   Email:    halloweeks@gmail.com
 *   Telegram: @halloweeks
 *   GitHub:   @halloweeks
 *
 * This software is provided "as is", without warranty of any kind,
 * express or implied.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// core
#include "oec.h"

int main(void) {
	OecImage image;
	OEC_TENSOR *input;
	
	// Load model
	OEC_MODEL *model = oec_model_load("finetune_last.oec");//"pretrained_best.oec"); //"finetune_best.oec");
	
	if (!model) {
		printf("load failed\n");
		return EXIT_FAILURE;
	}
	
	if (oec_image_load("dataset/train/images/img_00033.png", &image) != 0) {
		fprintf(stderr, "Failed to load input image\n");
		oec_model_free(model);
		return EXIT_FAILURE;
	}
	
	input = oec_tensor_from_image(&image);
	
	if (input == NULL) {
		fprintf(stderr, "Tensor error!\n");
		oec_image_free(&image);
		oec_model_free(model);
		oec_tensor_free(input);
		return EXIT_FAILURE;
	}
	
	// Prediction result
	OEC_PREDICTION output = {0};
		
	// Run prediction 
	if (oec_model_predict(model, input, &output) != 0) {
		fprintf(stderr, "Prediction failed!\n");
		oec_image_free(&image);
		oec_model_free(model);
		oec_tensor_free(input);
		return EXIT_FAILURE;
	}
	
	// Detection threshold
	if (output.confidence < 0.00) {
		printf("Target not found!\n");
		printf("presence = %.6f\n", output.confidence);
		oec_image_free(&image);
		oec_model_free(model);
		oec_tensor_free(input);
		return EXIT_FAILURE;
	}
	
	// Show prediction information 
	printf("Target found!\n");
	printf("presence = %f\n", output.confidence);
	printf("x = %f\n", output.x);
	printf("y = %f\n", output.y);
	printf("w = %f\n", output.width);
	printf("h = %f\n", output.height);
	
	// Draw box on that image with predicted information 
	oec_image_draw_box(
		&image,
		output.x,
		output.y,
		output.width,
		output.height,
		1,
		0.0f,
		1.0f,
		0.0f
	);
	
	// Save output visual image 
	if (oec_image_save_png("prediction/predict_00033.png", &image) != 0) {
		fprintf(stderr, "Failed to save predict image\n");
		oec_image_free(&image);
		oec_model_free(model);
		return EXIT_FAILURE;
	}
	
	printf("prediction image saved!\n");
	
	oec_tensor_free(input);
	oec_image_free(&image);
	oec_model_free(model);
	return EXIT_SUCCESS;
}