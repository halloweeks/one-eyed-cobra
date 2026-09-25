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
	OEC_TRAINER trainer;
	
	trainer.dataset = oec_dataset_open("dataset");
	
	if (!trainer.dataset) {
		fprintf(stderr, "Failed to open dataset\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	if (trainer.dataset->train.count == 0) {
		fprintf(stderr, "Couldn't found any images in dataset/train\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	printf("Found %zd samples\n", trainer.dataset->train.count);
	
	trainer.model = oec_model_load(INPUT_MODEL_PATH);
	
	if (!trainer.model) {
		fprintf(stderr, "Failed to load model\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	printf("Model Total Layers: %d\n", trainer.model->count);
	
	OEC_TENSOR *last = trainer.model->outputs[trainer.model->count - 1];
	
	if (last->c != 5) {
		fprintf(stderr, "Model's final layer must output 5 channels (confidence,x,y,w,h), got %d\n", last->c);
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	printf("model layers: %d\n", trainer.model->count);
	printf("model output shape: %dx%dx%d (grid %dx%d)\n", last->w, last->h, last->c, last->w, last->h);
	
	trainer.loss = oec_loss_create(5.0f, 0.5f, 5.0f);
	
	if (trainer.loss == NULL) {
		fprintf(stderr, "Failed to create loss\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.optimizer = oec_optimizer_init(OEC_OPTIMIZER_ADAM, LEARNING_RATE);
	
	if (trainer.optimizer == NULL) {
		fprintf(stderr, "Failed to create optimizer\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.input = oec_tensor_create(trainer.model->input->w, trainer.model->input->h, trainer.model->input->c);
	
	if (trainer.input == NULL) {
		fprintf(stderr, "Failed to allocate input tensor!\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.output = oec_tensor_create(last->w, last->h, last->c);
	
	if (trainer.output == NULL) {
		fprintf(stderr, "Failed to allocate output tensor\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.grad = oec_tensor_create(last->w, last->h, last->c);
	
	if (trainer.grad == NULL) {
		fprintf(stderr, "Failed to create loss gradient tensor\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.grad_input = oec_tensor_create(trainer.model->input->w, trainer.model->input->h, trainer.model->input->c);
	
	if (trainer.grad_input == NULL) {
		fprintf(stderr, "Failed to allocate input gradient tensor\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	float best_loss = INFINITY;
	
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		float total_loss = 0.0f;
		
		for (size_t i = 0; i < trainer.dataset->train.count; i++) {
			OEC_SAMPLE *sample = &trainer.dataset->train.samples[i];
			
			OEC_SAMPLE_DATA data = oec_dataset_load_sample(sample, trainer.input);
			
			OEC_TARGET target = select_target(&data);
			
			oec_optimizer_zero_grad(trainer.model);
			
			if (oec_model_forward(trainer.model, trainer.input, trainer.output) != 0) {
				fprintf(stderr, "Model forward failed\n");
				oec_trainer_free(&trainer);
				return EXIT_FAILURE;
			}
			
			float sample_loss = oec_loss_forward(trainer.loss, trainer.output, &target);
			total_loss += sample_loss;
			
			if (oec_loss_backward(trainer.loss, trainer.output, &target, trainer.grad) != 0) {
				fprintf(stderr, "Loss backward failed\n");
				oec_trainer_free(&trainer);
				return EXIT_FAILURE;
			}
			
			if (oec_model_backward(trainer.model, trainer.grad, trainer.grad_input) != 0) {
				fprintf(stderr, "Model backward failed\n");
				oec_trainer_free(&trainer);
				return EXIT_FAILURE;
			}
			
			oec_optimizer_step(trainer.optimizer, trainer.model);
		}
		
		printf(
			"Epoch %d/%d  loss = %.6f\n",
			epoch + 1,
			EPOCHS,
			total_loss / (float)trainer.dataset->train.count
		);
		
		float epoch_loss = total_loss / trainer.dataset->train.count;
		
		if (epoch_loss < best_loss) {
			best_loss = epoch_loss;
			
			if (oec_model_save(trainer.model, "finetune_best.oec") != 0) {
				fprintf(stderr, "Failed to save best model\n");
				oec_trainer_free(&trainer);
				return EXIT_FAILURE;
			}
			
			printf("  -> new best: %.6f\n", best_loss);
		}
	}
	
	if (oec_model_save(trainer.model, "finetune_last.oec") != 0) {
		fprintf(stderr, "Failed to save finetune_last.oec\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	printf("Model saved: finetune_last.oec\n");
	
	oec_trainer_free(&trainer);
    return 0;
}
