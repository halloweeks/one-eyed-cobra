#ifndef OEC_OPTIMIZER_H
#define OEC_OPTIMIZER_H

#include "oec_model.h"

typedef enum {
	OEC_OPTIMIZER_SGD,
	OEC_OPTIMIZER_MOMENTUM,
	OEC_OPTIMIZER_RMSPROP,
	OEC_OPTIMIZER_ADAM
} OEC_OPTIMIZER_TYPE;

typedef struct {
	OEC_OPTIMIZER_TYPE type;
	
	float learning_rate;
	
	/* momentum */
	float momentum;
	
	/* rmsprop */
	float decay;
	
	 /* adam */
	float beta1;
	float beta2;
	float epsilon;
	
	int step;
	int initialized;
	void *state;
} OEC_OPTIMIZER;

typedef struct {
	int count;
    float **velocity_weights;
    float **velocity_bias;
} OEC_MOMENTUM_STATE;

typedef struct {
	int count;
    float **square_weights;
    float **square_bias;
} OEC_RMSPROP_STATE;

typedef struct {
	int count;
    float **moment_weights;
    float **moment_bias;
    float **velocity_weights;
    float **velocity_bias;
} OEC_ADAM_STATE;

OEC_OPTIMIZER *oec_optimizer_init(OEC_OPTIMIZER_TYPE type, float learning_rate);
void oec_optimizer_free(OEC_OPTIMIZER *optimizer);
void oec_optimizer_zero_grad(OEC_MODEL *model);
void oec_optimizer_step(OEC_OPTIMIZER *optimizer, OEC_MODEL *model);

#endif