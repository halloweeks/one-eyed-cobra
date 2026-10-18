#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "oec.h"

#define EPOCHS 200
#define LEARNING_RATE 0.001f

#define OEC_INPUT_WIDTH  64
#define OEC_INPUT_HEIGHT 64

/*
 * Input channels are chosen at runtime:
 *
 *   oec_train          auto: grayscale if every dataset image is
 *                      grayscale, otherwise RGB
 *   oec_train gray     force a 1-channel model
 *   oec_train rgb      force a 3-channel model
 */

#define OUTPUT_BEST_MODEL_PATH "pretrained_best.oec"
#define OUTPUT_LAST_MODEL_PATH "pretrained_last.oec"

int main(int argc, char **argv)
{
	// For initialize random weight!
	srand(1234);
	
	int in_channels = 0;
	
	OEC_DATASET *dataset = NULL;
	OEC_MODEL *model = NULL;
	OEC_LOSS *loss = NULL;
	OEC_OPTIMIZER *optimizer = NULL;
	
	OEC_TENSOR *input = NULL;
	OEC_TENSOR *output = NULL;
	OEC_TENSOR *grad = NULL;
	OEC_TENSOR *grad_input = NULL;
	
	// Cell -> target map, only used for the first-epoch collision report.
	int *cell_target = NULL;
	
	
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
	
	if (argc > 1 && strcmp(argv[1], "gray") == 0) {
		in_channels = 1;
	} else if (argc > 1 && strcmp(argv[1], "rgb") == 0) {
		in_channels = 3;
	} else if (argc > 1 && strcmp(argv[1], "auto") != 0) {
		fprintf(stderr, "Usage: %s [auto|gray|rgb]\n", argv[0]);
		goto oec_exit;
	} else {
		in_channels = oec_dataset_detect_channels(dataset);
		
		if (in_channels < 0) {
			fprintf(stderr, "Couldn't read any dataset image\n");
			goto oec_exit;
		}
	}
	
	printf("Input: %dx%dx%d (%s)\n", OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, in_channels, in_channels == 1 ? "grayscale" : "RGB");
	
	model = oec_model_create(9);
	
	if (model == NULL) {
		fprintf(stderr, "Failed to create model\n");
		goto oec_exit;
	}
	
	// First conv takes 1 (grayscale) or 3 (RGB) input channels.
	OEC_LAYER *conv1 = oec_layer_conv_create(in_channels, 16, 3, 2);
	OEC_LAYER *conv2 = oec_layer_conv_create(16, 32, 3, 2);
	OEC_LAYER *conv3 = oec_layer_conv_create(32, 64, 3, 2);
	OEC_LAYER *conv4 = oec_layer_conv_create(64, 96, 3, 2);
	OEC_LAYER *conv5 = oec_layer_conv_create(96, 5,  1, 1);

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
	
	if (oec_model_build(model, OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, in_channels) != 0) {
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
	
	input = oec_tensor_create(OEC_INPUT_WIDTH, OEC_INPUT_HEIGHT, in_channels);
	
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
	
	cell_target = malloc((size_t)last->w * (size_t)last->h * sizeof(int));
	
	if (cell_target == NULL) {
		fprintf(stderr, "Failed to allocate cell map\n");
		goto oec_exit;
	}
	
	float best_loss = INFINITY;
	
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		float total_loss = 0.0f;
		
		// Box statistics (first epoch only).
		long box_total = 0;
		long box_moved = 0;
		long box_dropped = 0;
		
		for (size_t i = 0; i < dataset->train.count; i++) {
			OEC_SAMPLE *sample = &dataset->train.samples[i];
			
			OEC_SAMPLE_DATA data = oec_dataset_load_sample(sample, input);
			
			// Every labeled box becomes a training target.
			OEC_TARGETS targets = select_targets(&data);
			
			if (epoch == 0) {
				int moved = 0;
				int dropped = oec_target_assign(&targets, last->w, last->h, cell_target, &moved);
				
				box_total += targets.count;
				box_moved += moved;
				
				if (dropped > 0) {
					box_dropped += dropped;
				}
			}
			
			oec_optimizer_zero_grad(model);
			
			if (oec_model_forward(model, input, output) != 0) {
				fprintf(stderr, "Model forward failed\n");
				goto oec_exit;
			}
			
			float sample_loss = oec_loss_forward_multi(loss, output, &targets);
			total_loss += sample_loss;
			
			// Calculate gradients.
			if (oec_loss_backward_multi(loss, output, &targets, grad) != 0) {
				fprintf(stderr, "Loss backward failed\n");
				goto oec_exit;
			}
			
			// Backpropagation.
			oec_model_backward(model, grad, grad_input);
			
			// Update parameters.
			oec_optimizer_step(optimizer, model);
		}
		
		if (epoch == 0) {
			printf(
				"Targets: %ld boxes in %zu images (%ld moved to a neighbor cell, %ld dropped)\n",
				box_total,
				dataset->train.count,
				box_moved,
				box_dropped
			);
			
			if (box_dropped > 0) {
				fprintf(stderr, "Warning: %ld boxes don't fit in the %dx%d output grid and are not trained on\n", box_dropped, last->w, last->h);
			}
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
		free(cell_target);
    return 0;
}
