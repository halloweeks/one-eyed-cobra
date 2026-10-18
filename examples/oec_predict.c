/*
 * One-Eyed Cobra (OEC)
 *
 * Author:    Hallo Weeks
 * Copyright: (C) 2026 Hallo Weeks
 *
 * This software is provided "as is", without warranty of any kind,
 * express or implied.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// core implement 
#include "oec.h"

int main(void) {
	OecImage image;
	OEC_TENSOR *input;
	
	// Load model
	OEC_MODEL *model = oec_model_load("pretrained_best.oec");
	
	if (!model) {
		printf("load failed\n");
		return EXIT_FAILURE;
	}
	
	if (oec_image_load("dataset/train/images/img_00004.png", &image) != 0) {
		fprintf(stderr, "Failed to load input image\n");
		oec_model_free(model);
		return EXIT_FAILURE;
	}
	
	printf("Model input: %dx%dx%d (%s), image: %dx%dx%d\n",
		model->input->w, model->input->h, model->input->c,
		model->input->c == 1 ? "grayscale" : "RGB",
		image.width, image.height, image.channels);
	
	// Convert the image to the channel count the model was trained with.
	input = oec_tensor_from_image_channels(&image, model->input->c);
	
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
	if (output.confidence < 0.20) {
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
	
	// Drawing and saving work on RGB, so make an RGB view (gray input is replicated).
	OecImage view;
	
	if (oec_image_to_rgb(&image, &view) != 0) {
		fprintf(stderr, "Failed to create RGB view\n");
		oec_image_free(&image);
		oec_model_free(model);
		oec_tensor_free(input);
		return EXIT_FAILURE;
	}
	
	// Draw box on that image with predicted information 
	oec_image_draw_box(
		&view,
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
	if (oec_image_save_png("prediction/predict_00003.png", &view) != 0) {
		fprintf(stderr, "Failed to save predict image\n");
		oec_image_free(&view);
		oec_image_free(&image);
		oec_model_free(model);
		return EXIT_FAILURE;
	}
	
	printf("prediction image saved!\n");
	
	oec_tensor_free(input);
	oec_image_free(&view);
	oec_image_free(&image);
	oec_model_free(model);
	return EXIT_SUCCESS;
}