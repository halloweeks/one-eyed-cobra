#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "oec_model.h"

OEC_MODEL *oec_model_create(int capacity)
{
	if (capacity <= 0) {
		return NULL;
	}
	
	OEC_MODEL *model = calloc(1, sizeof(OEC_MODEL));
	
	if (model == NULL) {
		return NULL;
	}
	
	model->layers  = calloc((size_t)capacity, sizeof(OEC_LAYER *));
	model->outputs = calloc((size_t)capacity, sizeof(OEC_TENSOR *));
	model->grads   = calloc((size_t)capacity, sizeof(OEC_TENSOR *));
	
	if (model->layers == NULL || model->outputs == NULL || model->grads == NULL) {
		free(model->layers);
		free(model->outputs);
		free(model->grads);
		free(model);
		return NULL;
	}
	
	model->capacity = capacity;
	model->count = 0;
	
	return model;
}

int oec_model_save(const OEC_MODEL *model, const char *path)
{
	if (model == NULL || path == NULL) {
		return -1;
	}
	
	if (model->count <= 0 || model->layers == NULL) {
		return -1;
	}
	
	FILE *fp = fopen(path, "wb");
	
	if (fp == NULL) {
		return -1;
	}
	
	const char magic[4] = { 'O', 'E', 'C', 'M' };
	uint32_t version = 1;
	uint32_t layer_count = (uint32_t)model->count;
	
	if (fwrite(magic, sizeof(magic), 1, fp) != 1 ||
        fwrite(&version, sizeof(version), 1, fp) != 1 ||
        fwrite(&layer_count, sizeof(layer_count), 1, fp) != 1)
    {
        fclose(fp);
        return -1;
    }

    /*
     * Save every layer.
     */
    for (int i = 0; i < model->count; i++)
    {
        const OEC_LAYER *layer = model->layers[i];

        if (layer == NULL)
        {
            fclose(fp);
            return -1;
        }

        uint32_t type = (uint32_t)layer->type;

        if (fwrite(&type, sizeof(type), 1, fp) != 1)
        {
            fclose(fp);
            return -1;
        }

        switch (layer->type)
        {
        case OEC_LAYER_CONV:
        {
            const OEC_CONV *conv =
                layer->param.conv;

            if (conv == NULL)
            {
                fclose(fp);
                return -1;
            }

            int32_t in_channels =
                (int32_t)conv->in_channels;

            int32_t out_channels =
                (int32_t)conv->out_channels;

            int32_t kernel =
                (int32_t)conv->kernel;

            int32_t stride =
                (int32_t)conv->stride;

            if (fwrite(&in_channels,
                       sizeof(in_channels), 1, fp) != 1 ||
                fwrite(&out_channels,
                       sizeof(out_channels), 1, fp) != 1 ||
                fwrite(&kernel,
                       sizeof(kernel), 1, fp) != 1 ||
                fwrite(&stride,
                       sizeof(stride), 1, fp) != 1)
            {
                fclose(fp);
                return -1;
            }

            size_t weight_count =
                (size_t)conv->out_channels *
                (size_t)conv->in_channels *
                (size_t)conv->kernel *
                (size_t)conv->kernel;

            /*
             * Only learned parameters are saved.
             *
             * Gradients are runtime state and are not
             * part of the model.
             */
            if (fwrite(conv->weights,
                       sizeof(float),
                       weight_count,
                       fp) != weight_count)
            {
                fclose(fp);
                return -1;
            }

            if (fwrite(conv->bias,
                       sizeof(float),
                       (size_t)conv->out_channels,
                       fp) != (size_t)conv->out_channels)
            {
                fclose(fp);
                return -1;
            }

            break;
        }

        case OEC_LAYER_ACTIVATION:
        {
            const OEC_ACTIVATION *activation =
                layer->param.activation;

            if (activation == NULL)
            {
                fclose(fp);
                return -1;
            }

            uint32_t activation_type =
                (uint32_t)activation->type;

            if (fwrite(&activation_type,
                       sizeof(activation_type),
                       1,
                       fp) != 1)
            {
                fclose(fp);
                return -1;
            }

            /*
             * Save activation parameters.
             */
            if (fwrite(&activation->param.alpha,
                       sizeof(float),
                       1,
                       fp) != 1)
            {
                fclose(fp);
                return -1;
            }

            break;
        }
        
        case OEC_LAYER_GLOBAL_AVG_POOL: {
        	
            break;
        }

        default:
            fclose(fp);
            return -1;
        }
    }

    if (fclose(fp) != 0)
        return -1;

    return 0;
}

OEC_MODEL *oec_model_load(const char *path)
{
	if (path == NULL) {
		return NULL;
	}
	
	

    FILE *fp = fopen(path, "rb");

    if (fp == NULL)
        return NULL;

    char magic[4];
    uint32_t version;
    uint32_t layer_count;

    /*
     * Read header.
     */
    if (fread(magic, sizeof(magic), 1, fp) != 1 ||
        fread(&version, sizeof(version), 1, fp) != 1 ||
        fread(&layer_count, sizeof(layer_count), 1, fp) != 1)
    {
    	printf("invalid\n");
        fclose(fp);
        return NULL;
    }

    /*
     * Validate header.
     */
    if (magic[0] != 'O' ||
        magic[1] != 'E' ||
        magic[2] != 'C' ||
        magic[3] != 'M')
    {
    	printf("signature mismatch \n");
        fclose(fp);
        return NULL;
    }

    if (version != 1)
    {
    	printf("Unsupported version \n");
        fclose(fp);
        return NULL;
    }

    if (layer_count == 0 ||
        layer_count > 1000)
    {
    	printf("Layer count problem!\n");
        fclose(fp);
        return NULL;
    }

    OEC_MODEL *model =
        oec_model_create((int)layer_count);

    if (model == NULL)
    {
    	printf("fail to create model\n");
        fclose(fp);
        return NULL;
    }

    /*
     * Read layers.
     */
    for (uint32_t i = 0; i < layer_count; i++)
    {
        uint32_t type;

        if (fread(&type, sizeof(type), 1, fp) != 1) {
        	//printf("Failed read type \n");
            goto fail;
        }

        OEC_LAYER *layer = NULL;

        switch ((OEC_LAYER_TYPE)type)
        {
        case OEC_LAYER_CONV:
        {
            int32_t in_channels;
            int32_t out_channels;
            int32_t kernel;
            int32_t stride;

            if (fread(&in_channels,
                      sizeof(in_channels), 1, fp) != 1 ||
                fread(&out_channels,
                      sizeof(out_channels), 1, fp) != 1 ||
                fread(&kernel,
                      sizeof(kernel), 1, fp) != 1 ||
                fread(&stride,
                      sizeof(stride), 1, fp) != 1)
            {
            	//printf("Failed channel, kernel , stride\n");
                goto fail;
            }

            if (in_channels <= 0 ||
                out_channels <= 0 ||
                kernel <= 0 ||
                stride <= 0)
            {
            	//printf("Failed hola\n");
                goto fail;
            }

            layer =
                oec_layer_conv_create(
                    (int)in_channels,
                    (int)out_channels,
                    (int)kernel,
                    (int)stride);

            if (layer == NULL) {
            	//printf("Layer null \n");
                goto fail;
               }

            OEC_CONV *conv =
                layer->param.conv;

            size_t weight_count =
                (size_t)out_channels *
                (size_t)in_channels *
                (size_t)kernel *
                (size_t)kernel;

            if (fread(conv->weights,
                      sizeof(float),
                      weight_count,
                      fp) != weight_count)
            {
            	//printf("Failed read weight\n");
                oec_layer_free(layer);
                goto fail;
            }

            if (fread(conv->bias,
                      sizeof(float),
                      (size_t)out_channels,
                      fp) != (size_t)out_channels)
            {
            //	printf("Failed read bias \n");
                oec_layer_free(layer);
                goto fail;
            }

            break;
        }

        case OEC_LAYER_ACTIVATION:
        {
            uint32_t activation_type;
            float alpha;

            if (fread(&activation_type,
                      sizeof(activation_type),
                      1,
                      fp) != 1)
            {
            //	printf("Failed read activation type\n");
                goto fail;
            }

            if (fread(&alpha,
                      sizeof(alpha),
                      1,
                      fp) != 1)
            {
            	//printf("Failed read alpha \n");
                goto fail;
            }

            layer =
                oec_layer_activation_create(
                    (OEC_ACTIVATION_TYPE)activation_type);

            if (layer == NULL) {
            //	printf("Failed layer\n");
                goto fail;
               }

            layer->param.activation->param.alpha =
                alpha;

            break;
        }
        
        case OEC_LAYER_GLOBAL_AVG_POOL:
        {
        	
      
 layer =
        oec_layer_global_avg_pool_create();
        
        // printf("GAP create: layer=%p\n", (void *)layer);
        
    if (layer == NULL)
    {
    	// printf("error in 421\n");
        goto fail;
    }


    break;
}

        default:
            goto fail;
        }

        if (oec_model_add_layer(model, layer) != 0)
        {
        	
            oec_layer_free(layer);
            /*printf("error in layer %u: count=%d capacity=%d layer=%p\n",
           i,
           model->count,
           model->capacity,
           (void *)layer);*/
            goto fail;
        }
    }

    fclose(fp);

    /*
     * Recreate runtime tensors.
     *
     * The model file contains architecture + parameters,
     * but runtime input/output/gradient tensors are rebuilt.
     */
    if (oec_model_build(model, 64, 64, 3) != 0)
    {
    	// printf("failed to build loaded model\n");
        oec_model_free(model);
        return NULL;
    }

    return model;

fail:
    fclose(fp);
    oec_model_free(model);
    return NULL;
}

int oec_model_add_layer(
    OEC_MODEL *model,
    OEC_LAYER *layer)
{
    /*printf("ADD: model=%p layer=%p count=%d capacity=%d\n",
           (void *)model,
           (void *)layer,
           model ? model->count : -1,
           model ? model->capacity : -1);*/

    if (model == NULL || layer == NULL) {
       // printf("ADD FAILED: NULL\n");
        return -1;
    }

    if (model->count >= model->capacity) {
        // printf("ADD FAILED: capacity\n");
        return -1;
    }

    model->layers[model->count] = layer;
    model->count++;

   // printf("ADD OK: new count=%d\n", model->count);

    return 0;
}

int oec_model_add_layer2(
    OEC_MODEL *model,
    OEC_LAYER *layer)
{
    if (model == NULL || layer == NULL)
        return -1;

    if (model->count >= model->capacity)
        return -1;

    model->layers[model->count] = layer;
    model->count++;

    return 0;
}


/*
 * Determine output dimensions for a layer.
 */
static int layer_output_size(
    const OEC_LAYER *layer,
    const OEC_TENSOR *input,
    int *w,
    int *h,
    int *c)
{
    if (layer == NULL ||
        input == NULL ||
        w == NULL ||
        h == NULL ||
        c == NULL)
        return -1;

    switch (layer->type)
    {
    case OEC_LAYER_CONV:
    {
        const OEC_CONV *conv = layer->param.conv;

        if (conv == NULL)
            return -1;

        if (conv->stride <= 0 ||
            conv->kernel <= 0)
            return -1;

        if (input->w < conv->kernel ||
            input->h < conv->kernel)
            return -1;

        *w =
            (input->w - conv->kernel)
            / conv->stride + 1;

        *h =
            (input->h - conv->kernel)
            / conv->stride + 1;

        *c = conv->out_channels;

        return 0;
    }

    case OEC_LAYER_ACTIVATION:
        *w = input->w;
        *h = input->h;
        *c = input->c;

        return 0;
    case OEC_LAYER_GLOBAL_AVG_POOL:
    	*w = 1;
        *h = 1;
        *c = input->c;
        return 0; 
    default:
        return -1;
    }
}


int oec_model_build(OEC_MODEL *model, int input_w, int input_h, int input_c)
{
	if (model == NULL) {
		printf("model == NULL\n");
		return -1;
	}
	
	if (input_w <= 0 || input_h <= 0 || input_c <= 0) {
		printf("input_w <= 0 || input_h <= 0 || input_c <= 0\n");
		return -1;
	}
	
	/*
	 * Free an existing build.
	 */
	for (int i = 0; i < model->count; i++) {
		oec_tensor_free(model->outputs[i]);
		oec_tensor_free(model->grads[i]);
		
		model->outputs[i] = NULL;
		model->grads[i] = NULL;
	}
	
	oec_tensor_free(model->input);
	model->input = NULL;
	
	/*
	 * Store model input.
	 */
	model->input = oec_tensor_create(input_w, input_h, input_c);
	
	if (model->input == NULL) {
		printf("model->input == NULL 2\n");
		return -1;
	}
	
	OEC_TENSOR *current = model->input;
	
	for (int i = 0; i < model->count; i++) {
		int w;
		int h;
		int c;
		
		if (layer_output_size(model->layers[i], current, &w, &h, &c) != 0) {
			printf("error in layer_output_size(model->layers[i], current, &w, &h, &c)\n");
			return -1;
		}
		
		model->outputs[i] = oec_tensor_create(w, h, c);
		model->grads[i]   = oec_tensor_create(w, h, c);
		
		if (model->outputs[i] == NULL || model->grads[i] == NULL) {
			printf("error in model->outputs[i] == NULL || model->grads[i] == NULL)\n");
			return -1;
		}
		
		current = model->outputs[i];
	}
	
	return 0;
}

int oec_model_forward(OEC_MODEL *model, const OEC_TENSOR *input, OEC_TENSOR *output)
{
	if (model == NULL || input == NULL || output == NULL) {
		fprintf(stderr, "Moy moy\n");
		return -1;
	}
	
	if (model->count <= 0) {
		printf("Lol model count not enough: %d\n", model->count);
		return -1;
	}

    /*
     * Copy input into model-owned
     * activation storage.
     */
    size_t count =
        (size_t)input->w *
        (size_t)input->h *
        (size_t)input->c;

    size_t expected =
        (size_t)model->input->w *
        (size_t)model->input->h *
        (size_t)model->input->c;

    if (count != expected) {
    	fprintf(stderr,
            "count mismatch: count=%zu expected=%zu\n",
            count,
            expected);
        return -1;
       }

    for (size_t i = 0; i < count; i++)
        model->input->data[i] = input->data[i];

    const OEC_TENSOR *current =
        model->input;

    for (int i = 0; i < model->count; i++) {
    	/*
    	printf("forward layer %zu: input = %d x %d x %d\n",
       i,
       current->w,
       current->h,
       current->c);
       */
        if (oec_layer_forward(
                model->layers[i],
                current,
                model->outputs[i]) != 0) {
            
            printf("Failed oec_layer_forward()\n");
            return -1;
        }
        
        current = model->outputs[i];
    }

    /*
     * Copy final output to caller.
     */
    size_t out_count =
        (size_t)output->w *
        (size_t)output->h *
        (size_t)output->c;

    size_t model_out_count =
        (size_t)current->w *
        (size_t)current->h *
        (size_t)current->c;

    if (out_count != model_out_count) {
    	fprintf(stderr, "mismatch count\n");
  
        return -1;
       }

    for (size_t i = 0; i < out_count; i++)
        output->data[i] = current->data[i];

    return 0;
}


int oec_model_backward(
    OEC_MODEL *model,
    const OEC_TENSOR *grad_output,
    OEC_TENSOR *grad_input)
{
    if (model == NULL ||
        grad_output == NULL ||
        grad_input == NULL)
        return -1;

    if (model->count <= 0)
        return -1;

    int last = model->count - 1;

    /*
     * Copy final gradient into the
     * last layer gradient buffer.
     */
    size_t count =
        (size_t)grad_output->w *
        (size_t)grad_output->h *
        (size_t)grad_output->c;

    size_t expected =
        (size_t)model->grads[last]->w *
        (size_t)model->grads[last]->h *
        (size_t)model->grads[last]->c;

    if (count != expected)
        return -1;

    for (size_t i = 0; i < count; i++)
        model->grads[last]->data[i] =
            grad_output->data[i];

    /*
     * Walk backward through the network.
     */
    for (int i = last; i >= 0; i--)
    {
        const OEC_TENSOR *layer_input;

        if (i == 0)
            layer_input = model->input;
        else
            layer_input = model->outputs[i - 1];

        const OEC_TENSOR *layer_output =
            model->outputs[i];

        OEC_TENSOR *previous_grad;

        if (i == 0)
            previous_grad = grad_input;
        else
            previous_grad = model->grads[i - 1];

        if (oec_layer_backward(
                model->layers[i],
                layer_input,
                layer_output,
                model->grads[i],
                previous_grad) != 0)
            return -1;
    }

    return 0;
}


void oec_model_free(OEC_MODEL *model)
{
	if (model == NULL) {
		return;
	}
	
	for (int i = 0; i < model->count; i++) {
		oec_layer_free(model->layers[i]);
		oec_tensor_free(model->outputs[i]);
		oec_tensor_free(model->grads[i]);
	}
	
	oec_tensor_free(model->input);
	
	free(model->layers);
	free(model->outputs);
	free(model->grads);
	free(model);
}

void oec_model_zero_grad(OEC_MODEL *model) {
	
}

void oec_model_sgd_step(
    OEC_MODEL *model,
    float learning_rate
) {
	
}

int oec_model_predict(OEC_MODEL *model, const OEC_TENSOR *input, OEC_PREDICTION *prediction)
{
	if (model == NULL) {
		fprintf(stderr, "predict: model == NULL\n");
		return -1;
	}

	if (input == NULL) {
		fprintf(stderr, "predict: input == NULL\n");
		return -1;
	}

	if (prediction == NULL) {
		fprintf(stderr, "predict: prediction == NULL\n");
		return -1;
	}

	if (model->count <= 0 || model->outputs == NULL ||
	    model->outputs[model->count - 1] == NULL) {
		fprintf(stderr, "predict: model has not been built\n");
		return -1;
	}

	const OEC_TENSOR *last = model->outputs[model->count - 1];
	
	if (last->c != 5) {
		fprintf(stderr, "predict: model's final layer must output 5 channels, got %d\n", last->c);
		return -1;
	}
	
	OEC_TENSOR *output = oec_tensor_create(last->w, last->h, last->c);
	
	if (output == NULL) {
		fprintf(stderr, "predict: failed to create output tensor\n");
		return -1;
	}
	
	if (oec_model_forward(model, input, output) != 0) {
		oec_tensor_free(output);
		fprintf(stderr, "predict: model forward failed\n");
		return -1;
	}
	
	int grid_w = output->w;
	int grid_h = output->h;
	size_t plane = (size_t)grid_w * (size_t)grid_h;

	/*
	 * Find the grid cell with the highest confidence logit.
	 * sigmoid is monotonic, so comparing raw logits is equivalent
	 * to comparing probabilities and avoids extra expf() calls.
	 */
	size_t best_idx = 0;
	float best_logit = output->data[0];

	for (size_t idx = 1; idx < plane; idx++) {
		float logit = output->data[idx];

		if (logit > best_logit) {
			best_logit = logit;
			best_idx = idx;
		}
	}

	int best_row = (int)(best_idx / (size_t)grid_w);
	int best_col = (int)(best_idx % (size_t)grid_w);

	float confidence_logit = output->data[0 * plane + best_idx];
	float x_offset = output->data[1 * plane + best_idx];
	float y_offset = output->data[2 * plane + best_idx];
	float w = output->data[3 * plane + best_idx];
	float h = output->data[4 * plane + best_idx];

	prediction->confidence = 1.0f / (1.0f + expf(-confidence_logit));

	/*
	 * Invert the cell-relative parameterization used during
	 * training (see oec_loss_forward/backward): x/y were stored
	 * as an offset within the responsible cell, not full-image
	 * coordinates.
	 */
	float cell_w = 1.0f / (float)grid_w;
	float cell_h = 1.0f / (float)grid_h;

	prediction->x = ((float)best_col + x_offset) * cell_w;
	prediction->y = ((float)best_row + y_offset) * cell_h;
	prediction->width  = w;
	prediction->height = h;

	oec_tensor_free(output);

	return 0;
}

int oec_model_predict2(OEC_MODEL *model, const OEC_TENSOR *input, OEC_PREDICTION *prediction) 
{
	
	if (model == NULL) {
        fprintf(stderr, "predict: model == NULL\n");
        return -1;
    }

    if (input == NULL) {
        fprintf(stderr, "predict: input == NULL\n");
        return -1;
    }

    if (prediction == NULL) {
        fprintf(stderr, "predict: prediction == NULL\n");
        return -1;
    }
    
    printf("input: %d x %d x %d\n",
           input->w,
           input->h,
           input->c);
           
    
	if (model == NULL || input == NULL || prediction == NULL) {
		return -1;
	}
	
	printf("input data: %p\n", (void *)input->data);
	
	
	OEC_TENSOR *output = oec_tensor_create(1, 1, 5);
	
	if (output == NULL) {
		fprintf(stderr, "predict: failed to create output tensor\n");
		return -1;
	}
	
	if (oec_model_forward(model, input, output) != 0) {
		oec_tensor_free(output);
		fprintf(stderr, "predict: model forward failed\n");
		return -1;
	}
	
	printf("forward succeeded\n");
	
	
	float confidence_logit = output->data[0];
	
	prediction->confidence = 1.0f / (1.0f + expf(-confidence_logit));
	prediction->x      = output->data[1];
	prediction->y      = output->data[2];
	prediction->width   = output->data[3];
	prediction->height  = output->data[4];
	
	oec_tensor_free(output);
	
	return 0;
}

