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
#include <math.h>

// core
#include "oec.h"

#define EPOCHS 100
#define LEARNING_RATE 0.0001f

#define INPUT_MODEL_PATH "pretrained_best.oec"
#define OUTPUT_BEST_MODEL_PATH "finetuned_best.oec"
#define OUTPUT_LAST_MODEL_PATH "finetuned_last.oec"

int main(void)
{
	OEC_DATASET *dataset = oec_dataset_open("dataset");
	
	if (!dataset) {
		fprintf(stderr, "Failed to open dataset\n");
		return EXIT_FAILURE;
	}
	
	if (dataset->train.count == 0) {
		fprintf(stderr, "Couldn't found any images in dataset/train\n");
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	printf("Found %zd training samples: %zd\n", dataset->train.count);
	
	OEC_MODEL *model = oec_model_load(INPUT_MODEL_PATH);
	
	if (!model) {
		fprintf(stderr, "Failed to load model\n");
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	printf("Model Total Layers: %d\n", model->count);
	
	OEC_TENSOR *last = model->outputs[model->count - 1];
	
	if (last->c != 5) {
		fprintf(stderr, "Model's final layer must output 5 channels (confidence,x,y,w,h), got %d\n", last->c);
		oec_model_free(model);
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	OEC_TENSOR *image = oec_tensor_create(model->input->w, model->input->h, model->input->c);
	
	if (image == NULL) {
		fprintf(stderr, "Failed to allocate tensor!\n");
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	printf("model layers: %d\n", model->count);
	printf("model output shape: %dx%dx%d (grid %dx%d)\n", last->w, last->h, last->c, last->w, last->h);
	// printf(" size: %dx%dx%d\n", model->input->w, model->input->h, model->input->c);
	
	OEC_LOSS *loss = oec_loss_create(5.0f, 0.5f, 5.0f);
	
	if (loss == NULL) {
		fprintf(stderr, "Failed to create loss\n");
		return EXIT_FAILURE;
	}
	
	OEC_OPTIMIZER *optimizer = oec_optimizer_init(OEC_OPTIMIZER_ADAM, LEARNING_RATE);
	
	if (optimizer == NULL) {
		fprintf(stderr, "Failed to create optimizer\n");
		return 1;
	}
	
	OEC_TENSOR *output = oec_tensor_create(last->w, last->h, last->c);
	
	OEC_TENSOR *grad = oec_tensor_create(last->w, last->h, last->c);
	
	OEC_TENSOR *grad_input = oec_tensor_create(model->input->w, model->input->h, model->input->c);
	
	if (grad == NULL) {
		fprintf(stderr, "Failed to create loss gradient tensor\n");
		oec_model_free(model);
		oec_tensor_free(image);
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	float best_loss = INFINITY;
	
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		float total_loss = 0.0f;
		
		for (size_t i = 0; i < dataset->train.count; i++) {
			OEC_SAMPLE *sample = &dataset->train.samples[i];
			
			OEC_SAMPLE_DATA data = oec_dataset_load_sample(sample, image);
			
			OEC_TARGET target = select_target(&data);
			
			oec_optimizer_zero_grad(model);
			
			if (oec_model_forward(model, image, output) != 0) {
				fprintf(stderr, "Model forward failed\n");
				return EXIT_FAILURE;
			}
			
			float sample_loss = oec_loss_forward(loss, output, &target);
			total_loss += sample_loss;
			
			if (oec_loss_backward(loss, output, &target, grad) != 0) {
				fprintf(stderr, "Loss backward failed\n");
				return EXIT_FAILURE;
			}
			
			oec_model_backward(model, grad, grad_input);
			
			oec_optimizer_step(optimizer, model);
		}
		
		printf(
			"Epoch %d/%d  loss = %.6f\n",
			epoch + 1,
			EPOCHS,
			total_loss / (float)dataset->train.count
		);
		
		float epoch_loss = total_loss / dataset->train.count;
		
		if (epoch_loss < best_loss) {
			best_loss = epoch_loss;
			
			if (oec_model_save(model, "finetune_best.oec") != 0) {
				fprintf(stderr, "Failed to save best model\n");
				return EXIT_FAILURE;
			}
			
			printf("  -> new best: %.6f\n", best_loss);
		}
	}
	
	if (oec_model_save(model, "finetune_last.oec") != 0) {
		fprintf(stderr, "Failed to save model\n");
	}
	
	printf("Model saved: finetune_last.oec\n");
	
	oec_tensor_free(grad_input);
	oec_tensor_free(grad);
	oec_tensor_free(output);
	oec_tensor_free(image);
	oec_model_free(model);
	oec_loss_free(loss);
	oec_dataset_close(dataset);
	oec_optimizer_free(optimizer);
    return 0;
}
