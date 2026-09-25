#include <math.h>
#include <stdlib.h>

#include "oec_loss.h"

OEC_LOSS *oec_loss_create(float object_weight, float no_object_weight, float box_weight)
{
	if (object_weight < 0.0f || no_object_weight < 0.0f || box_weight < 0.0f) {
		return NULL;
	}
	
	OEC_LOSS *loss = calloc(1, sizeof(*loss));
	
	if (loss == NULL) {
		return NULL;
	}
	
	loss->object_weight    = object_weight;
	loss->no_object_weight = no_object_weight;
	loss->box_weight       = box_weight;
	
	return loss;
}

void oec_loss_free(OEC_LOSS *loss)
{
	if (loss == NULL) {
		return;
	}
	
	free(loss);
}

static int oec_loss_assign_cell(
	const OEC_TARGET *target,
	int grid_w,
	int grid_h,
	int *col,
	int *row)
{
	if (target->confidence <= 0.0f)
		return -1;
	
	int c = (int)(target->x * (float)grid_w);
	int r = (int)(target->y * (float)grid_h);
	
	if (c < 0) c = 0;
	if (c >= grid_w) c = grid_w - 1;
	
	if (r < 0) r = 0;
	if (r >= grid_h) r = grid_h - 1;
	
	*col = c;
	*row = r;
	
	return 0;
}

float oec_loss_forward(const OEC_LOSS *loss, const OEC_TENSOR *prediction, const OEC_TARGET *target)
{
	if (loss == NULL || prediction == NULL || target == NULL) {
		return -1.0f;
	}
	
	if (prediction->c != 5 || prediction->w <= 0 || prediction->h <= 0) {
		return -1.0f;
	}
	
	int grid_w = prediction->w;
	int grid_h = prediction->h;
	
	int assigned_col = -1;
	int assigned_row = -1;
	
	int has_object =
		(oec_loss_assign_cell(target, grid_w, grid_h, &assigned_col, &assigned_row) == 0);
	
	size_t plane = (size_t)grid_w * (size_t)grid_h;
	
	float positive_loss_sum = 0.0f;
	float negative_loss_sum = 0.0f;
	float box_loss = 0.0f;
	
	for (int row = 0; row < grid_h; row++) {
		for (int col = 0; col < grid_w; col++) {
			
			size_t idx = (size_t)row * grid_w + col;
			
			float confidence_logit = prediction->data[0 * plane + idx];
			
			int is_positive =
				has_object && row == assigned_row && col == assigned_col;
			
			float confidence_target = is_positive ? 1.0f : 0.0f;
			
			float cell_bce =
				fmaxf(confidence_logit, 0.0f) -
				confidence_logit * confidence_target +
				log1pf(expf(-fabsf(confidence_logit)));
			
			if (is_positive) {
				positive_loss_sum += cell_bce;
				
				float cell_w = 1.0f / (float)grid_w;
				float cell_h = 1.0f / (float)grid_h;
				
				float target_x = target->x / cell_w - (float)col;
				float target_y = target->y / cell_h - (float)row;
				
				float x = prediction->data[1 * plane + idx];
				float y = prediction->data[2 * plane + idx];
				float w = prediction->data[3 * plane + idx];
				float h = prediction->data[4 * plane + idx];
				
				float dx = x - target_x;
				float dy = y - target_y;
				float dw = w - target->w;
				float dh = h - target->h;
				
				box_loss = 0.25f * (dx * dx + dy * dy + dw * dw + dh * dh);
			} else {
				negative_loss_sum += cell_bce;
			}
		}
	}
	
	/* Positive cell(s) and negative cells contribute separately,
	 * each weighted and averaged over their own count, so the lone
	 * positive cell's signal isn't diluted by the 1000+ negatives. */
	float positive_count = has_object ? 1.0f : 0.0f;
	float negative_count = (float)plane - positive_count;
	
	float positive_loss = positive_count > 0.0f ? positive_loss_sum / positive_count : 0.0f;
	float negative_loss = negative_count > 0.0f ? negative_loss_sum / negative_count : 0.0f;
	
	float confidence_loss =
		loss->object_weight * positive_loss +
		loss->no_object_weight * negative_loss;
	
	return confidence_loss + loss->box_weight * box_loss;
}

int oec_loss_backward(const OEC_LOSS *loss, const OEC_TENSOR *prediction, const OEC_TARGET *target, OEC_TENSOR *grad)
{
	if (loss == NULL || prediction == NULL || target == NULL || grad == NULL) {
		return -1;
	}
	
	if (prediction->c != 5 || prediction->w <= 0 || prediction->h <= 0) {
		return -1;
	}
	
	if (grad->w != prediction->w || grad->h != prediction->h || grad->c != 5) {
		return -1;
	}
	
	int grid_w = prediction->w;
	int grid_h = prediction->h;
	
	int assigned_col = -1;
	int assigned_row = -1;
	
	int has_object =
		(oec_loss_assign_cell(target, grid_w, grid_h, &assigned_col, &assigned_row) == 0);
	
	size_t plane = (size_t)grid_w * (size_t)grid_h;
	size_t total = plane * 5;
	
	for (size_t i = 0; i < total; i++)
		grad->data[i] = 0.0f;
	
	float positive_count = has_object ? 1.0f : 0.0f;
	float negative_count = (float)plane - positive_count;
	
	for (int row = 0; row < grid_h; row++) {
		for (int col = 0; col < grid_w; col++) {
			
			size_t idx = (size_t)row * grid_w + col;
			
			float confidence_logit = prediction->data[0 * plane + idx];
			
			int is_positive =
				has_object && row == assigned_row && col == assigned_col;
			
			float confidence_target = is_positive ? 1.0f : 0.0f;
			
			float confidence = 1.0f / (1.0f + expf(-confidence_logit));
			
			float weight = is_positive
				? loss->object_weight / (positive_count > 0.0f ? positive_count : 1.0f)
				: loss->no_object_weight / (negative_count > 0.0f ? negative_count : 1.0f);
			
			grad->data[0 * plane + idx] = weight * (confidence - confidence_target);
			
			if (is_positive) {
				
				float cell_w = 1.0f / (float)grid_w;
				float cell_h = 1.0f / (float)grid_h;
				
				float target_x = target->x / cell_w - (float)col;
				float target_y = target->y / cell_h - (float)row;
				
				float dx = prediction->data[1 * plane + idx] - target_x;
				float dy = prediction->data[2 * plane + idx] - target_y;
				float dw = prediction->data[3 * plane + idx] - target->w;
				float dh = prediction->data[4 * plane + idx] - target->h;
				
				grad->data[1 * plane + idx] = loss->box_weight * 0.5f * dx;
				grad->data[2 * plane + idx] = loss->box_weight * 0.5f * dy;
				grad->data[3 * plane + idx] = loss->box_weight * 0.5f * dw;
				grad->data[4 * plane + idx] = loss->box_weight * 0.5f * dh;
			}
		}
	}
	
	return 0;
}

/*
OEC_LOSS *oec_loss_create(float confidence_weight, float box_weight)
{
	if (confidence_weight < 0.0f || box_weight < 0.0f) {
		return NULL;
	}
	
	OEC_LOSS *loss = calloc(1, sizeof(*loss));
	
	if (loss == NULL) {
		return NULL;
	}
	
	loss->confidence_weight = confidence_weight;
	loss->box_weight = box_weight;
	
	return loss;
}


void oec_loss_free(OEC_LOSS *loss)
{
	if (loss == NULL) {
		return;
	}
	
	free(loss);
}
*/


/*
 * Locate the single grid cell responsible for `target`, based on its
 * normalized image-space center (target->x, target->y in [0,1)).
 * Returns 0 and fills *col row if an object is present
 * (target->confidence > 0), or -1 if there is no object to assign.
 */
 
 
 /*
static int oec_loss_assign_cell(const OEC_TARGET *target, int grid_w, int grid_h, int *col, int *row) {
	if (target->confidence <= 0.0f) {
		return -1;
	}
	
	int c = (int)(target->x * (float)grid_w);
	int r = (int)(target->y * (float)grid_h);
	
	if (c < 0) c = 0;
	if (c >= grid_w) c = grid_w - 1;
	
	if (r < 0) r = 0;
	if (r >= grid_h) r = grid_h - 1;
	
	*col = c;
	*row = r;
	
	return 0;
}
*/

/*
float oec_loss_forward2(const OEC_LOSS *loss, const OEC_TENSOR *prediction, const OEC_TARGET *target)
{
	
	if (loss == NULL || prediction == NULL || target == NULL) {
		return -1.0f;
	}
	
	if (prediction->w != 1 || prediction->h != 1 || prediction->c != 5) {
		return -1.0f;
	}
	
	const float confidence_logit = prediction->data[0];
	
	const float x = prediction->data[1];
	const float y = prediction->data[2];
	const float w = prediction->data[3];
	const float h = prediction->data[4];
	
	
	 // Binary cross entropy with logits.
	 
	
	float confidence_loss = fmaxf(confidence_logit, 0.0f) - confidence_logit * target->confidence + log1pf(expf(-fabsf(confidence_logit)));
	
	float dx = x - target->x;
	float dy = y - target->y;
	float dw = w - target->w;
	float dh = h - target->h;
	
	float box_loss = 0.25f * (dx * dx + dy * dy + dw * dw + dh * dh);
	
	return loss->confidence_weight * confidence_loss + loss->box_weight * box_loss;
}
*/

/*
int oec_loss_backward2(const OEC_LOSS *loss, const OEC_TENSOR *prediction, const OEC_TARGET *target, OEC_TENSOR *grad)
{
	if (loss == NULL || prediction == NULL || target == NULL || grad == NULL) {
		return -1;
	}
	
	if (prediction->w != 1 || prediction->h != 1 || prediction->c != 5) {
		return -1;
	}
	
	if (grad->w != 1 || grad->h != 1 || grad->c != 5) {
		return -1;
	}
	
	const float confidence_logit = prediction->data[0];
	const float dx = prediction->data[1] - target->x;
	const float dy = prediction->data[2] - target->y;
	const float dw = prediction->data[3] - target->w;
	const float dh = prediction->data[4] - target->h;
	
	 // dL/dz = sigmoid(z) - target
	 
	float confidence = 1.0f / (1.0f + expf(-confidence_logit));
	
	float confidence_grad = confidence - target->confidence;
	
	grad->data[0] = loss->confidence_weight * confidence_grad;
	grad->data[1] = loss->box_weight * 0.5f * dx;
	grad->data[2] = loss->box_weight * 0.5f * dy;
	grad->data[3] = loss->box_weight * 0.5f * dw;
	grad->data[4] = loss->box_weight * 0.5f * dh;
	
	return 0;
}*/

/*
float oec_loss_forward(const OEC_LOSS *loss, const OEC_TENSOR *prediction, const OEC_TARGET *target)
{
	if (loss == NULL || prediction == NULL || target == NULL) {
		return -1.0f;
	}
	
	if (prediction->c != 5 || prediction->w <= 0 || prediction->h <= 0) {
		return -1.0f;
	}
	
	int grid_w = prediction->w;
	int grid_h = prediction->h;
	
	int assigned_col = -1;
	int assigned_row = -1;
	
	int has_object =
		(oec_loss_assign_cell(target, grid_w, grid_h, &assigned_col, &assigned_row) == 0);
	
	size_t plane = (size_t)grid_w * (size_t)grid_h;
	
	float confidence_loss_sum = 0.0f;
	float box_loss = 0.0f;
	
	for (int row = 0; row < grid_h; row++) {
		for (int col = 0; col < grid_w; col++) {
			
			size_t idx = (size_t)row * grid_w + col;
			
			float confidence_logit = prediction->data[0 * plane + idx];
			
			int is_positive =
				has_object && row == assigned_row && col == assigned_col;
			
			float confidence_target = is_positive ? 1.0f : 0.0f;
			
			// Binary cross entropy with logits, per cell. 
			confidence_loss_sum +=
				fmaxf(confidence_logit, 0.0f) -
				confidence_logit * confidence_target +
				log1pf(expf(-fabsf(confidence_logit)));
			
			if (is_positive) {
				
				float cell_w = 1.0f / (float)grid_w;
				float cell_h = 1.0f / (float)grid_h;
				
				// Target x/y re-expressed as an offset within this cell. 
				float target_x = target->x / cell_w - (float)col;
				float target_y = target->y / cell_h - (float)row;
				
				float x = prediction->data[1 * plane + idx];
				float y = prediction->data[2 * plane + idx];
				float w = prediction->data[3 * plane + idx];
				float h = prediction->data[4 * plane + idx];
				
				float dx = x - target_x;
				float dy = y - target_y;
				float dw = w - target->w;
				float dh = h - target->h;
				
				box_loss = 0.25f * (dx * dx + dy * dy + dw * dw + dh * dh);
			}
		}
	}
	
	// Average confidence loss over all cells so scale stays comparable
	 // * across different grid sizes (and matches the old single-cell loss
	// * exactly when grid_w == grid_h == 1). 
	float confidence_loss = confidence_loss_sum / (float)plane;
	
	return loss->confidence_weight * confidence_loss + loss->box_weight * box_loss;
}

int oec_loss_backward(const OEC_LOSS *loss, const OEC_TENSOR *prediction, const OEC_TARGET *target, OEC_TENSOR *grad)
{
	if (loss == NULL || prediction == NULL || target == NULL || grad == NULL) {
		return -1;
	}
	
	if (prediction->c != 5 || prediction->w <= 0 || prediction->h <= 0) {
		return -1;
	}
	
	if (grad->w != prediction->w || grad->h != prediction->h || grad->c != 5) {
		return -1;
	}
	
	int grid_w = prediction->w;
	int grid_h = prediction->h;
	
	int assigned_col = -1;
	int assigned_row = -1;
	
	int has_object =
		(oec_loss_assign_cell(target, grid_w, grid_h, &assigned_col, &assigned_row) == 0);
	
	size_t plane = (size_t)grid_w * (size_t)grid_h;
	size_t total = plane * 5;
	
	for (size_t i = 0; i < total; i++)
		grad->data[i] = 0.0f;
	
	for (int row = 0; row < grid_h; row++) {
		for (int col = 0; col < grid_w; col++) {
			
			size_t idx = (size_t)row * grid_w + col;
			
			float confidence_logit = prediction->data[0 * plane + idx];
			
			int is_positive =
				has_object && row == assigned_row && col == assigned_col;
			
			float confidence_target = is_positive ? 1.0f : 0.0f;
			
			float confidence = 1.0f / (1.0f + expf(-confidence_logit));
			
			grad->data[0 * plane + idx] =
				loss->confidence_weight * (confidence - confidence_target) / (float)plane;
			
			if (is_positive) {
				
				float cell_w = 1.0f / (float)grid_w;
				float cell_h = 1.0f / (float)grid_h;
				
				float target_x = target->x / cell_w - (float)col;
				float target_y = target->y / cell_h - (float)row;
				
				float dx = prediction->data[1 * plane + idx] - target_x;
				float dy = prediction->data[2 * plane + idx] - target_y;
				float dw = prediction->data[3 * plane + idx] - target->w;
				float dh = prediction->data[4 * plane + idx] - target->h;
				
				grad->data[1 * plane + idx] = loss->box_weight * 0.5f * dx;
				grad->data[2 * plane + idx] = loss->box_weight * 0.5f * dy;
				grad->data[3 * plane + idx] = loss->box_weight * 0.5f * dw;
				grad->data[4 * plane + idx] = loss->box_weight * 0.5f * dh;
			}
		}
	}
	
	return 0;
}*/