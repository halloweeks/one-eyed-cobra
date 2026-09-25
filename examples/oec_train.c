#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "oec.h"

#define EPOCHS 100
#define LEARNING_RATE 0.001f

#define OEC_INPUT_WIDTH  64
#define OEC_INPUT_HEIGHT 64
#define OEC_INPUT_CHANNEL 3

#define OUTPUT_BEST_MODEL_PATH "pretrained_best.oec"
#define OUTPUT_LAST_MODEL_PATH "pretrained_last.oec"

int main(void)
{
	// For initialize random weight!
	srand(1234);
	
	OEC_TRAINER trainer;
	
	oec_trainer_init(&trainer);
	
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
	
	printf("Found %zd training samples\n", trainer.dataset->train.count);
	
	trainer.model = oec_model_create(9);
	
	if (trainer.model == NULL) {
		fprintf(stderr, "Failed to create model\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	OEC_LAYER *conv1 = oec_layer_conv_create(3, 16, 3, 2);
	OEC_LAYER *conv2 = oec_layer_conv_create(16, 32, 3, 2);
	OEC_LAYER *conv3 = oec_layer_conv_create(32, 64, 3, 2);
	OEC_LAYER *conv4 = oec_layer_conv_create(64, 96, 3, 2);
	OEC_LAYER *conv5 = oec_layer_conv_create(96, 5, 1, 1);
	
	OEC_LAYER *act1  = oec_layer_activation_create(OEC_ACTIVATION_RELU);
	OEC_LAYER *act2  = oec_layer_activation_create(OEC_ACTIVATION_RELU);
	OEC_LAYER *act3  = oec_layer_activation_create(OEC_ACTIVATION_RELU);
	OEC_LAYER *act4  = oec_layer_activation_create(OEC_ACTIVATION_RELU);
	
	oec_model_add_layer(trainer.model, conv1);
	oec_model_add_layer(trainer.model, act1);
	oec_model_add_layer(trainer.model, conv2);
	oec_model_add_layer(trainer.model, act2);
	oec_model_add_layer(trainer.model, conv3);
	oec_model_add_layer(trainer.model, act3);
	oec_model_add_layer(trainer.model, conv4);
	oec_model_add_layer(trainer.model, act4);
	oec_model_add_layer(trainer.model, conv5);
	
	if (oec_model_build(trainer.model, OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, OEC_INPUT_CHANNEL) != 0) {
		fprintf(stderr, "Failed to build model\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	OEC_TENSOR *last = trainer.model->outputs[trainer.model->count - 1];
	
	if (last->c != 5) {
		fprintf(stderr, "Model's final layer must output 5 channels (confidence,x,y,w,h), got %d\n", last->c);
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	printf("Model Total Layers: %d\n", trainer.model->count);
	printf("Model output shape: %dx%dx%d (grid %dx%d)\n", last->w, last->h, last->c, last->w, last->h);
	
	trainer.loss = oec_loss_create(5.0f, 0.5f, 5.0f);
	
	if (trainer.loss == NULL) {
		fprintf(
			stderr,
			"Failed to create loss function "
			"(confidence=5.0, coordinate=0.5, size=5.0)\n"
		);
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.optimizer = oec_optimizer_init(OEC_OPTIMIZER_ADAM, LEARNING_RATE);
	
	if (trainer.optimizer == NULL) {
		fprintf(stderr, "Failed to create optimizer\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	trainer.input = oec_tensor_create(OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, OEC_INPUT_CHANNEL);
	
	if (trainer.input == NULL) {
		fprintf(stderr, "Failed to allocate input tensor\n");
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
	
	trainer.best_loss = INFINITY;
	
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
			
			oec_model_backward(trainer.model, trainer.grad, trainer.grad_input);
			
			oec_optimizer_step(trainer.optimizer, trainer.model);
		}
		
		printf(
			"Epoch %d/%d  loss = %.6f\n",
			trainer.epoch + 1,
			EPOCHS,
			total_loss / (float)trainer.dataset->train.count
		);
		
		float epoch_loss = total_loss / trainer.dataset->train.count;
		
		if (epoch_loss < trainer.best_loss) {
			trainer.best_loss = epoch_loss;
			
			if (oec_model_save(trainer.model, OUTPUT_BEST_MODEL_PATH) != 0) {
				fprintf(stderr, "Failed to save best model\n");
				oec_trainer_free(&trainer);
				return EXIT_FAILURE;
			}
			
			printf("  -> new best: %.6f\n", trainer.best_loss);
		}
	}
	
	if (oec_model_save(trainer.model, OUTPUT_LAST_MODEL_PATH) != 0) {
		fprintf(stderr, "Failed to save model\n");
		oec_trainer_free(&trainer);
		return EXIT_FAILURE;
	}
	
	printf("Model saved: pretrained_last.oec\n");
    return 0;
}
