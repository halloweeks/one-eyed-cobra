#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "oec_dataset.h"
#include "oec_tensor.h"
#include "oec_model.h"
#include "oec_image.h"

#include "oec_target.h"
#include "oec_loss.h"

#include "oec_optimizer.h"

#include "oec_activation.h"

#define EPOCHS 200
#define LEARNING_RATE 0.001f

static OEC_TARGET select_target(const OEC_SAMPLE_DATA *data)
{
	OEC_TARGET target = {0};
	
	if (data == NULL || data->box_count == 0) {
		return target;
	}
	
	const OEC_BBox *box = &data->boxes[0];
	
	target.confidence = 1.0f;
	target.x = box->x_center;
	target.y = box->y_center;
	target.w = box->width;
	target.h = box->height;
	
	// printf("%f %f %f %f %f\n", target.confidence, target.x, target.y, target.w, target.h);
	
	return target;
}

int main(void)
{
	srand(1234);
	setbuf(stdout, NULL);
	
	OEC_DATASET *dataset = oec_dataset_open("dataset");
	
	if (!dataset) {
		fprintf(stderr, "Failed to open dataset\n");
		return EXIT_FAILURE;
	}
	
	if (dataset->test.count == 0) {
		fprintf(stderr, "Couldn't found any images in dataset/test\n");
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	OEC_TENSOR *image = oec_tensor_create(64, 64, 3);
	
	if (image == NULL) {
		fprintf(stderr, "Failed to allocate tensor!\n");
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	printf("test count: %zd\n", dataset->test.count);
	
	/*
	 * Create OEC model here.
	 */
	
	OEC_MODEL *model = oec_model_create(9);
	
	if (model == NULL) {
		fprintf(stderr, "Failed to create model\n");
		oec_tensor_free(image);
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	OEC_LAYER *conv1 = oec_layer_conv_create(3, 16, 3, 2);
	OEC_LAYER *conv2 = oec_layer_conv_create(16, 32, 3, 2);
	OEC_LAYER *conv3 = oec_layer_conv_create(32, 64, 3, 2);
	OEC_LAYER *conv4 = oec_layer_conv_create(64, 96, 3, 2);
	
	/* Global Average Pooling */
	// OEC_LAYER *gap = oec_layer_global_avg_pool_create();
	
	/* Detection head: 96 -> 5 */
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
	
	// oec_model_add_layer(model, gap);
	oec_model_add_layer(model, conv5);
	
	// 64x64 
	if (oec_model_build(model, 64, 64, 3) != 0) {
		fprintf(stderr, "Failed to build model\n");
		oec_model_free(model);
		oec_tensor_free(image);
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	printf("model layers: %d\n", model->count);
	
	OEC_TENSOR *last = model->outputs[model->count - 1];
	
	if (last->c != 5) {
		fprintf(stderr, "Model's final layer must output 5 channels (confidence,x,y,w,h), got %d\n", last->c);
		oec_model_free(model);
		oec_tensor_free(image);
		oec_dataset_close(dataset);
		return EXIT_FAILURE;
	}
	
	printf("model layers: %d\n", model->count);
	printf("model output shape: %dx%dx%d (grid %dx%d)\n", last->w, last->h, last->c, last->w, last->h);
	
	// OEC_LOSS *loss = oec_loss_create(1.0f, 5.0f);
	
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
		
		for (size_t i = 0; i < dataset->test.count; i++) {
			OEC_SAMPLE *sample = &dataset->test.samples[i];
			/*
			 * Load image into tensor
			 * and all labels.
			 */
			
			OEC_SAMPLE_DATA data = oec_dataset_load_sample(sample, image);
			
			
			/*
			 * Select one object.
			 */
			OEC_TARGET target = select_target(&data);
			
			/*
			 * Clear gradients.
			 */
			oec_optimizer_zero_grad(model);
			
			/*
			 * Forward.
			 */
			if (oec_model_forward(model, image, output) != 0) {
				fprintf(stderr, "Model forward failed\n");
				return EXIT_FAILURE;
			}
			/*
			 * Loss forward 
			 */
			
			float sample_loss = oec_loss_forward(loss, output, &target);
			
			total_loss += sample_loss;
			
			
			/*
			 * Loss backward.
			 */
			if (oec_loss_backward(loss, output, &target, grad) != 0) {
				fprintf(stderr, "Loss backward failed\n");
				return EXIT_FAILURE;
			}
			
			/*
			 * Model backward.
			 */
			oec_model_backward(model, grad, grad_input);
			
			/*
			 * Check weights BEFORE optimizer update.
			 */
			/*
			if (epoch == 0 && i == 0) {
				printf("conv1 before: %f\n", conv1->param.conv->weights[0]);
				printf("conv5 before: %f\n", conv5->param.conv->weights[0]);
			}
			*/
			
			
			/*
			 * Update weights.
			 */
			
			// printf("weight before: %f\n", conv1->param.conv->weights[0]);
			
			oec_optimizer_step(optimizer, model);
			
			// printf("weight after:  %f\n", conv1->param.conv->weights[0]);
			
		}
		
		printf(
			"Epoch %d/%d  loss = %.6f\n",
			epoch + 1,
			EPOCHS,
			total_loss / (float)dataset->train.count
		);
		
		
		float epoch_loss = total_loss / dataset->train.count;
		/*
		 * Save the model whenever this is
		 * the lowest loss seen so far.
		 */
		if (epoch_loss < best_loss) {
			best_loss = epoch_loss;
			
			if (oec_model_save(model, "best.oec") != 0) {
				fprintf(stderr, "Failed to save best model\n");
				return EXIT_FAILURE;
			}
			
			printf("  -> new best: %.6f\n", best_loss);
		}
	}
	
	if (oec_model_save(model, "test.oec") != 0) {
		fprintf(stderr, "Failed to save model\n");
	}
	
	printf("Model saved: person.oec\n");
	
	/*
	 * Cleanup.
	 */
	
	oec_tensor_free(grad_input);
	oec_tensor_free(grad);
	oec_tensor_free(output);
	oec_tensor_free(image);
	oec_model_free(model);
	oec_loss_free(loss);
	oec_dataset_close(dataset);
	oec_optimizer_free(optimizer);
	
	oec_layer_free(conv1);
	oec_layer_free(conv2);
	oec_layer_free(conv3);
	oec_layer_free(conv4);
	oec_layer_free(conv5);
	
	oec_layer_free(act1);
	oec_layer_free(act2);
	oec_layer_free(act3);
	oec_layer_free(act4);
	
    return 0;
}