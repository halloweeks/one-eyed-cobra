#include "oec.h"
#include <string.h>

void oec_trainer_init(OEC_TRAINER *trainer)
{
	if (trainer == NULL) {
		return;
	}
	
	memset(trainer, 0, sizeof(*trainer));
}

void oec_trainer_free(OEC_TRAINER *trainer)
{
	if (trainer == NULL) {
		return;
	}
	
	oec_dataset_close(trainer->dataset);
	oec_model_free(trainer->model);
	oec_loss_free(trainer->loss);
	oec_optimizer_free(trainer->optimizer);
	
	oec_tensor_free(trainer->input);
	oec_tensor_free(trainer->output);
	oec_tensor_free(trainer->grad);
	oec_tensor_free(trainer->grad_input);
	
	memset(trainer, 0, sizeof(*trainer));
}