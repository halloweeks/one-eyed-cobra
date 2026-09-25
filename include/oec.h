#ifndef OEC_H
#define OEC_H

#include "oec_tensor.h"
#include "oec_target.h"
#include "oec_activation.h"
#include "oec_conv.h"
#include "oec_layer.h"
#include "oec_model.h"
#include "oec_loss.h"
#include "oec_optimizer.h"
#include "oec_dataset.h"
#include "oec_image.h"

typedef struct
{
	OEC_DATASET   *dataset;
	OEC_MODEL     *model;
	OEC_LOSS      *loss;
	OEC_OPTIMIZER *optimizer;
	
	OEC_TENSOR    *input;
	OEC_TENSOR    *output;
	OEC_TENSOR    *grad;
	OEC_TENSOR    *grad_input;
	
	float best_loss;
	int epoch;
} OEC_TRAINER;

void oec_trainer_init(OEC_TRAINER *trainer);
void oec_trainer_free(OEC_TRAINER *trainer);

#endif