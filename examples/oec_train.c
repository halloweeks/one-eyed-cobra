#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "oec.h"

#define EPOCHS 100
#define LEARNING_RATE 0.001f

// x = (64 * 64 * 3) x = 12,288
#define OEC_INPUT_WIDTH  64
#define OEC_INPUT_HEIGHT 64
#define OEC_INPUT_CHANNEL 3

#define OUTPUT_BEST_MODEL_PATH "pretrained_best.oec"
#define OUTPUT_LAST_MODEL_PATH "pretrained_last.oec"

int main(void)
{
	// For initialize random weight!
	srand(1234);
	
	OEC_DATASET *dataset = NULL;
	OEC_MODEL *model = NULL;
	OEC_LOSS *loss = NULL;
	OEC_OPTIMIZER *optimizer = NULL;
	
	OEC_TENSOR *input = NULL;
	OEC_TENSOR *output = NULL;
	OEC_TENSOR *grad = NULL;
	OEC_TENSOR *grad_input = NULL;
	
	
	dataset = oec_dataset_open("dataset");
	
	if (!dataset) {
		fprintf(stderr, "Failed to open dataset\n");
		goto oec_exit;
	}
	
	if (dataset->train.count == 0) {
		fprintf(stderr, "Couldn't found any images in dataset/train\n");
		goto oec_exit;
	}
	
	printf("Found %zd training samples\n", dataset->train.count);
	
	model = oec_model_create(9);
	
	if (model == NULL) {
		fprintf(stderr, "Failed to create model\n");
		goto oec_exit;
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
	
	oec_model_add_layer(model, conv1);
	oec_model_add_layer(model, act1);
	oec_model_add_layer(model, conv2);
	oec_model_add_layer(model, act2);
	oec_model_add_layer(model, conv3);
	oec_model_add_layer(model, act3);
	oec_model_add_layer(model, conv4);
	oec_model_add_layer(model, act4);
	oec_model_add_layer(model, conv5);
	
	if (oec_model_build(model, OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, OEC_INPUT_CHANNEL) != 0) {
		fprintf(stderr, "Failed to build model\n");
		goto oec_exit;
	}
	
	OEC_TENSOR *last = model->outputs[model->count - 1];
	
	if (last->c != 5) {
		fprintf(stderr, "Model's final layer must output 5 channels (confidence,x,y,w,h), got %d\n", last->c);
		goto oec_exit;
	}
	
	printf("Model Total Layers: %d\n", model->count);
	printf("Model output shape: %dx%dx%d (grid %dx%d)\n", last->w, last->h, last->c, last->w, last->h);
	
	loss = oec_loss_create(5.0f, 0.5f, 5.0f);
	
	if (loss == NULL) {
		fprintf(
			stderr,
			"Failed to create loss function "
			"(confidence=5.0, coordinate=0.5, size=5.0)\n"
		);
		goto oec_exit;
	}
	
	optimizer = oec_optimizer_init(OEC_OPTIMIZER_ADAM, LEARNING_RATE);
	
	if (optimizer == NULL) {
		fprintf(stderr, "Failed to create optimizer\n");
		goto oec_exit;
	}
	
	input = oec_tensor_create(OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, OEC_INPUT_CHANNEL);
	
	if (input == NULL) {
		fprintf(stderr, "Failed to allocate input tensor\n");
		goto oec_exit;
	}
	
	output = oec_tensor_create(last->w, last->h, last->c);
	
	if (output == NULL) {
		fprintf(stderr, "Failed to allocate output tensor\n");
		goto oec_exit;
	}
	
	grad = oec_tensor_create(last->w, last->h, last->c);
	
	if (grad == NULL) {
		fprintf(stderr, "Failed to create loss gradient tensor\n");
		goto oec_exit;
	}
	
	grad_input = oec_tensor_create(model->input->w, model->input->h, model->input->c);
	
	if (grad_input == NULL) {
		fprintf(stderr, "Failed to allocate input gradient tensor\n");
		goto oec_exit;
	}
	
	float best_loss = INFINITY;
	
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		float total_loss = 0.0f;
		
		for (size_t i = 0; i < dataset->train.count; i++) {
			OEC_SAMPLE *sample = &dataset->train.samples[i];
			
			OEC_SAMPLE_DATA data = oec_dataset_load_sample(sample, input);
			
			OEC_TARGET target = select_target(&data);
			
			oec_optimizer_zero_grad(model);
			
			if (oec_model_forward(model, input, output) != 0) {
				fprintf(stderr, "Model forward failed\n");
				goto oec_exit;
			}
			
			float sample_loss = oec_loss_forward(loss, output, &target);
			total_loss += sample_loss;
			
			if (oec_loss_backward(loss, output, &target, grad) != 0) {
				fprintf(stderr, "Loss backward failed\n");
				goto oec_exit;
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
			
			if (oec_model_save(model, OUTPUT_BEST_MODEL_PATH) != 0) {
				fprintf(stderr, "Failed to save best model\n");
				goto oec_exit;
			}
			
			printf("  -> new best: %.6f\n", best_loss);
		}
	}
	
	if (oec_model_save(model, OUTPUT_LAST_MODEL_PATH) != 0) {
		fprintf(stderr, "Failed to save model\n");
		goto oec_exit;
	}
	
	printf("Model saved: pretrained_last.oec\n");
	
	oec_exit: 
		oec_model_free(model);
		oec_dataset_close(dataset);
		oec_loss_free(loss);
		oec_optimizer_free(optimizer);
		oec_tensor_free(input);
		oec_tensor_free(output);
		oec_tensor_free(grad);
		oec_tensor_free(grad_input);
    return 0;
}
