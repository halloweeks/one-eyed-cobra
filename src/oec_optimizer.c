#include <stdlib.h>
#include <math.h>

#include "oec_optimizer.h"
#include "oec_layer.h"
#include "oec_conv.h"

OEC_OPTIMIZER *oec_optimizer_init(OEC_OPTIMIZER_TYPE type, float learning_rate)
{
	// OEC_OPTIMIZER *opt = calloc(1, sizeof(OEC_OPTIMIZER));
	OEC_OPTIMIZER *optimizer = malloc(sizeof(OEC_OPTIMIZER));
	
	if (optimizer == NULL) {
		return NULL;
	}
	
	optimizer->type = type;
	optimizer->learning_rate = learning_rate;
	
	/* Momentum defaults */
	optimizer->momentum = 0.9f;
	
	/* RMSProp defaults */
	optimizer->decay = 0.99f;
	
	/* Adam defaults */
	optimizer->beta1 = 0.9f;
	optimizer->beta2 = 0.999f;
	optimizer->epsilon = 1e-8f;
	
	optimizer->step = 0;
	optimizer->initialized = 0;
	optimizer->state = NULL;
	return optimizer;
}

static int oec_optimizer_state_init(OEC_OPTIMIZER *optimizer, OEC_MODEL *model)
{
	if (optimizer == NULL || model == NULL)
		return 0;
	
	switch (optimizer->type) {
		case OEC_OPTIMIZER_SGD:
			return 1;
		case OEC_OPTIMIZER_MOMENTUM:
		{
			OEC_MOMENTUM_STATE *state = calloc(1, sizeof(OEC_MOMENTUM_STATE));
			
			if (state == NULL)
				return 0;
			
			state->count = model->count;
			
			state->velocity_weights = calloc(model->count, sizeof(float *));
			state->velocity_bias    = calloc(model->count, sizeof(float *));
			
			if (state->velocity_weights == NULL || state->velocity_bias == NULL) {
				free(state->velocity_weights);
				free(state->velocity_bias);
				free(state);
				return 0;
			}
			
			for (int i = 0; i < model->count; ++i) {
				OEC_LAYER *layer = model->layers[i];
				
				if (layer == NULL) 
					continue;
				
				if (layer->type != OEC_LAYER_CONV)
					continue;
				
				OEC_CONV *conv = layer->param.conv;
				
				if (conv == NULL)
					continue;
				
				int weight_count = conv->out_channels * conv->in_channels * conv->kernel * conv->kernel;
				
				state->velocity_weights[i] = calloc(weight_count, sizeof(float));
				state->velocity_bias[i]    = calloc(conv->out_channels, sizeof(float));
				
				if (state->velocity_weights[i] == NULL || state->velocity_bias[i] == NULL) {
					for (int j = 0; j <= i; ++j) {
						free(state->velocity_weights[j]);
						free(state->velocity_bias[j]);
					}
					
					free(state->velocity_weights);
					free(state->velocity_bias);
					free(state);
					
					return 0;
				}
			}
			
			optimizer->state = state;
			
			return 1;
		}
		
		case OEC_OPTIMIZER_RMSPROP:
        {
            OEC_RMSPROP_STATE *state =
                calloc(1, sizeof(OEC_RMSPROP_STATE));

            if (state == NULL)
                return 0;

            state->count = model->count;

            state->square_weights =
                calloc(model->count, sizeof(float *));

            state->square_bias =
                calloc(model->count, sizeof(float *));

            if (state->square_weights == NULL ||
                state->square_bias == NULL)
            {
                free(state->square_weights);
                free(state->square_bias);
                free(state);
                return 0;
            }

            for (int i = 0;
                 i < model->count;
                 ++i)
            {
                OEC_LAYER *layer =
                    model->layers[i];

                if (layer == NULL)
                    continue;

                if (layer->type != OEC_LAYER_CONV)
                    continue;

                OEC_CONV *conv =
                    layer->param.conv;

                if (conv == NULL)
                    continue;

                int weight_count =
                    conv->out_channels *
                    conv->in_channels *
                    conv->kernel *
                    conv->kernel;

                state->square_weights[i] =
                    calloc(weight_count, sizeof(float));

                state->square_bias[i] =
                    calloc(conv->out_channels,
                           sizeof(float));

                if (state->square_weights[i] == NULL ||
                    state->square_bias[i] == NULL)
                {
                    for (int j = 0; j <= i; ++j)
                    {
                        free(state->square_weights[j]);
                        free(state->square_bias[j]);
                    }

                    free(state->square_weights);
                    free(state->square_bias);
                    free(state);

                    return 0;
                }
            }

            optimizer->state = state;

            return 1;
        }

        case OEC_OPTIMIZER_ADAM:
        {
            OEC_ADAM_STATE *state =
                calloc(1, sizeof(OEC_ADAM_STATE));

            if (state == NULL)
                return 0;

            state->count = model->count;

            state->moment_weights =
                calloc(model->count, sizeof(float *));

            state->moment_bias =
                calloc(model->count, sizeof(float *));

            state->velocity_weights =
                calloc(model->count, sizeof(float *));

            state->velocity_bias =
                calloc(model->count, sizeof(float *));

            if (state->moment_weights == NULL ||
                state->moment_bias == NULL ||
                state->velocity_weights == NULL ||
                state->velocity_bias == NULL)
            {
                free(state->moment_weights);
                free(state->moment_bias);
                free(state->velocity_weights);
                free(state->velocity_bias);
                free(state);
                return 0;
            }

            for (int i = 0;
                 i < model->count;
                 ++i)
            {
                OEC_LAYER *layer =
                    model->layers[i];

                if (layer == NULL)
                    continue;

                if (layer->type != OEC_LAYER_CONV)
                    continue;

                OEC_CONV *conv =
                    layer->param.conv;

                if (conv == NULL)
                    continue;

                int weight_count =
                    conv->out_channels *
                    conv->in_channels *
                    conv->kernel *
                    conv->kernel;

                state->moment_weights[i] =
                    calloc(weight_count, sizeof(float));

                state->moment_bias[i] =
                    calloc(conv->out_channels,
                           sizeof(float));

                state->velocity_weights[i] =
                    calloc(weight_count, sizeof(float));

                state->velocity_bias[i] =
                    calloc(conv->out_channels,
                           sizeof(float));

                if (state->moment_weights[i] == NULL ||
                    state->moment_bias[i] == NULL ||
                    state->velocity_weights[i] == NULL ||
                    state->velocity_bias[i] == NULL)
                {
                    for (int j = 0; j <= i; ++j)
                    {
                        free(state->moment_weights[j]);
                        free(state->moment_bias[j]);
                        free(state->velocity_weights[j]);
                        free(state->velocity_bias[j]);
                    }

                    free(state->moment_weights);
                    free(state->moment_bias);
                    free(state->velocity_weights);
                    free(state->velocity_bias);
                    free(state);

                    return 0;
                }
            }

            optimizer->state = state;

            return 1;
        }
		
		default:
			return 0;
	}
}


void oec_optimizer_zero_grad(OEC_MODEL *model)
{
	if (model == NULL) {
		return;
	}
	
	for (int i = 0; i < model->count; ++i) {
		OEC_LAYER *layer = model->layers[i];
		
		if (layer == NULL) {
			continue;
		}
		
		if (layer->type == OEC_LAYER_CONV) {
			OEC_CONV *conv = layer->param.conv;
			
			if (conv == NULL)
				continue;
			
			int weight_count = conv->out_channels * conv->in_channels * conv->kernel * conv->kernel;
			
			for (int j = 0; j < weight_count; ++j) {
				conv->grad_weights[j] = 0.0f;
			}
			
			for (int j = 0; j < conv->out_channels; ++j) {
				conv->grad_bias[j] = 0.0f;
			}
		}
	}
}


static void oec_optimizer_sgd_step(OEC_OPTIMIZER *optimizer, OEC_MODEL *model) {
	/*
	 * NULL checks are handled by oec_optimizer_step().
	 * This internal function assumes valid arguments.
	 */
	
	for (int i = 0; i < model->count; ++i) {
		OEC_LAYER *layer = model->layers[i];
		
		if (layer == NULL) {
			continue;
		}
		
		if (layer->type == OEC_LAYER_CONV) {
			OEC_CONV *conv = layer->param.conv;
			
			int weight_count = conv->out_channels * conv->in_channels * conv->kernel * conv->kernel;
			
			/* Update weight */
			for (int j = 0; j < weight_count; ++j) {
				conv->weights[j] -= optimizer->learning_rate * conv->grad_weights[j];
			}
			
			for (int j = 0; j < conv->out_channels; ++j) {
				conv->bias[j] -= optimizer->learning_rate * conv->grad_bias[j];
			}
		}
	}
}

static void oec_optimizer_momentum_step(OEC_OPTIMIZER *optimizer, OEC_MODEL *model) {
	/*
	 * NULL checks are handled by oec_optimizer_step().
	 * This internal function assumes valid arguments.
	 */
	
	float learning_rate = optimizer->learning_rate;
	float momentum      = optimizer->momentum;
	
	OEC_MOMENTUM_STATE *state = optimizer->state;
	
	for (int i = 0; i < model->count; ++i) {
		OEC_LAYER *layer = model->layers[i];
		
		if (layer == NULL) {
			continue;
		}
		
		if (layer->type == OEC_LAYER_CONV) {
			OEC_CONV *conv = layer->param.conv;
			
			if (conv == NULL)
				continue;
			
			float *velocity_weights = state->velocity_weights[i];
			float *velocity_bias    = state->velocity_bias[i];
			
			int weight_count = conv->out_channels * conv->in_channels * conv->kernel * conv->kernel;
			
			for (int j = 0; j < weight_count; ++j) {
				velocity_weights[j] = momentum * velocity_weights[j] + conv->grad_weights[j];
				
				conv->weights[j] -= learning_rate * velocity_weights[j];
			}
			
			for (int j = 0; j < conv->out_channels; ++j) {
				velocity_bias[j] = momentum * velocity_bias[j] + conv->grad_bias[j];
				
				conv->bias[j] -= learning_rate * velocity_bias[j];
			}
		}
	}
}

static void oec_optimizer_rmsprop_step(OEC_OPTIMIZER *optimizer, OEC_MODEL *model)
{
	/*
	 * NULL checks are handled by oec_optimizer_step().
	 * This internal function assumes valid arguments.
	 */
	
	float learning_rate = optimizer->learning_rate;
	float decay         = optimizer->decay;
	float epsilon       = optimizer->epsilon;
	
	OEC_RMSPROP_STATE *state = optimizer->state;
	
	for (int i = 0; i < model->count; ++i) {
		OEC_LAYER *layer = model->layers[i];
		
		if (layer == NULL)
			continue;
		
		if (layer->type == OEC_LAYER_CONV) {
			OEC_CONV *conv = layer->param.conv;
			
			if (conv == NULL)
				continue;
			
			float *square_weights = state->square_weights[i];
			float *square_bias    = state->square_bias[i];
			int weight_count      = conv->out_channels * conv->in_channels * conv->kernel * conv->kernel;
			
			for (int j = 0; j < weight_count; ++j) {
				float gradient = conv->grad_weights[j];
				
				square_weights[j] = decay * square_weights[j] + (1.0f - decay) * gradient * gradient;
				
				conv->weights[j] -= learning_rate * gradient / (sqrtf(square_weights[j]) + epsilon);
			}
			
			for (int j = 0; j < conv->out_channels; ++j) {
				float gradient = conv->grad_bias[j];
				
				square_bias[j] = decay * square_bias[j] + (1.0f - decay) * gradient * gradient;
				
				conv->bias[j] -= learning_rate * gradient / (sqrtf(square_bias[j]) + epsilon);
			}
		}
	}
}

static void oec_optimizer_adam_step(OEC_OPTIMIZER *optimizer, OEC_MODEL *model)
{
	/*
	 * NULL checks are handled by oec_optimizer_step().
	 * This internal function assumes valid arguments.
	 */
	
	float learning_rate = optimizer->learning_rate;
	float beta1         = optimizer->beta1;
	float beta2         = optimizer->beta2;
	float epsilon       = optimizer->epsilon;
	
	OEC_ADAM_STATE *state = optimizer->state;
	
	optimizer->step++;
	
	float beta1_correction = 1.0f - powf(beta1, optimizer->step);
	float beta2_correction = 1.0f - powf(beta2, optimizer->step);
	
	for (int i = 0; i < model->count; ++i) {
		OEC_LAYER *layer = model->layers[i];
		
		if (layer == NULL)
			continue;
		
		if (layer->type == OEC_LAYER_CONV) {
			OEC_CONV *conv = layer->param.conv;
			
			if (conv == NULL)
				continue;
			
			float *moment_weights   = state->moment_weights[i];
			float *moment_bias      = state->moment_bias[i];
			float *velocity_weights = state->velocity_weights[i];
			float *velocity_bias    = state->velocity_bias[i];
			
			
			int weight_count = conv->out_channels * conv->in_channels * conv->kernel * conv->kernel;
			
			for (int j = 0; j < weight_count; ++j) {
				float gradient = conv->grad_weights[j];
				
				moment_weights[j] = beta1 * moment_weights[j] + (1.0f - beta1) * gradient;
				
				velocity_weights[j] = beta2 * velocity_weights[j] + (1.0f - beta2) * gradient * gradient;
				
				float moment_hat   = moment_weights[j] / beta1_correction;
				float velocity_hat = velocity_weights[j] / beta2_correction;
				
				conv->weights[j] -= learning_rate * moment_hat / (sqrtf(velocity_hat) +epsilon);
			}
			
			for (int j = 0; j < conv->out_channels; ++j) {
				float gradient = conv->grad_bias[j];
				
				moment_bias[j] = beta1 * moment_bias[j] + (1.0f - beta1) * gradient;
				
				velocity_bias[j] = beta2 * velocity_bias[j] + (1.0f - beta2) * gradient * gradient;
				
				float moment_hat = moment_bias[j] / beta1_correction;
				
				float velocity_hat = velocity_bias[j] / beta2_correction;
				
				conv->bias[j] -= learning_rate * moment_hat / (sqrtf(velocity_hat) + epsilon);
			}
		}
	}
}

void oec_optimizer_step(OEC_OPTIMIZER *optimizer, OEC_MODEL *model)
{
	if (optimizer == NULL || model == NULL) {
		return;
	}
	
	if (!optimizer->initialized) {
		if (!oec_optimizer_state_init(optimizer, model)) {
			fprintf(stderr,
				"oec_optimizer_step: "
				"failed to allocate optimizer state "
				"(type=%d)\n",
				optimizer->type
			);
			
			return;
		}
		
		optimizer->initialized = 1;
	}
	
	switch (optimizer->type) {
		case OEC_OPTIMIZER_SGD:
			oec_optimizer_sgd_step(optimizer, model);
			break;
		case OEC_OPTIMIZER_MOMENTUM:
			oec_optimizer_momentum_step(optimizer, model);
			break;
		case OEC_OPTIMIZER_RMSPROP:
			oec_optimizer_rmsprop_step(optimizer, model);
			break;
		case OEC_OPTIMIZER_ADAM:
			oec_optimizer_adam_step(optimizer, model);
			break;
	}
}

static void oec_optimizer_state_free(OEC_OPTIMIZER *optimizer)
{
	if (optimizer == NULL || optimizer->state == NULL) {
		return;
	}
	
	switch (optimizer->type) {
		case OEC_OPTIMIZER_SGD: 
		{
			/* SGD has no persistent state. */
			break;
		}
		
		case OEC_OPTIMIZER_MOMENTUM:
		{
			OEC_MOMENTUM_STATE *state = optimizer->state;
			
			for (int i = 0; i < state->count; ++i) {
				free(state->velocity_weights[i]);
				free(state->velocity_bias[i]);
			}
			
			free(state->velocity_weights);
			free(state->velocity_bias);
			free(state);
			
			break;
		}
		
		case OEC_OPTIMIZER_RMSPROP:
		{
			OEC_RMSPROP_STATE *state = optimizer->state;
			
			for (int i = 0; i < state->count; ++i) {
				free(state->square_weights[i]);
				free(state->square_bias[i]);
			}
			
			free(state->square_weights);
			free(state->square_bias);
			free(state);
			
			break;
		}
		
		case OEC_OPTIMIZER_ADAM:
		{
			OEC_ADAM_STATE *state = optimizer->state;
			
			for (int i = 0; i < state->count; ++i) {
				free(state->moment_weights[i]);
				free(state->moment_bias[i]);
				
				free(state->velocity_weights[i]);
				free(state->velocity_bias[i]);
			}
			
			free(state->moment_weights);
			free(state->moment_bias);
			
			free(state->velocity_weights);
			free(state->velocity_bias);
			
			free(state);
			break;
		}
		
		default:
			free(optimizer->state);
			break;
	}
	
	optimizer->state = NULL;
}

void oec_optimizer_free(OEC_OPTIMIZER *optimizer)
{
	if (optimizer == NULL) {
		return;
	}
	
	oec_optimizer_state_free(optimizer);
	
	free(optimizer);
}