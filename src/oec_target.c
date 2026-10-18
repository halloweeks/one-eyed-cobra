#include "oec_target.h"

OEC_TARGET select_target(const OEC_SAMPLE_DATA *data)
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
	
	return target;
}

OEC_TARGETS select_targets(const OEC_SAMPLE_DATA *data)
{
	OEC_TARGETS result = {0};
	
	if (data == NULL) {
		return result;
	}
	
	for (size_t i = 0; i < data->box_count && result.count < OEC_MAX_TARGETS; i++) {
		const OEC_BBox *box = &data->boxes[i];
		OEC_TARGET *target = &result.targets[result.count];
		
		target->confidence = 1.0f;
		target->x = box->x_center;
		target->y = box->y_center;
		target->w = box->width;
		target->h = box->height;
		
		result.count++;
	}
	
	return result;
}

int oec_target_assign(
	const OEC_TARGETS *targets,
	int grid_w,
	int grid_h,
	int *cell_target,
	int *moved)
{
	if (targets == NULL || cell_target == NULL || grid_w <= 0 || grid_h <= 0) {
		return -1;
	}
	
	int cells = grid_w * grid_h;
	
	for (int i = 0; i < cells; i++) {
		cell_target[i] = -1;
	}
	
	// Targets that are really objects, biggest first (stable for equal areas).
	int order[OEC_MAX_TARGETS];
	int n = 0;
	
	for (int i = 0; i < targets->count && i < OEC_MAX_TARGETS; i++) {
		if (targets->targets[i].confidence <= 0.0f) {
			continue;
		}
		
		float area = targets->targets[i].w * targets->targets[i].h;
		int j = n++;
		
		while (j > 0) {
			const OEC_TARGET *prev = &targets->targets[order[j - 1]];
			
			if (prev->w * prev->h >= area) {
				break;
			}
			
			order[j] = order[j - 1];
			j--;
		}
		
		order[j] = i;
	}
	
	int moved_count = 0;
	int dropped = 0;
	
	for (int k = 0; k < n; k++) {
		const OEC_TARGET *t = &targets->targets[order[k]];
		
		// The cell a lone object would use.
		int own_col = (int)(t->x * (float)grid_w);
		int own_row = (int)(t->y * (float)grid_h);
		
		if (own_col < 0) own_col = 0;
		if (own_col >= grid_w) own_col = grid_w - 1;
		if (own_row < 0) own_row = 0;
		if (own_row >= grid_h) own_row = grid_h - 1;
		
		int own = own_row * grid_w + own_col;
		int best = -1;
		float best_dist = 0.0f;
		
		for (int cell = 0; cell < cells; cell++) {
			if (cell_target[cell] != -1) {
				continue;
			}
			
			float cx = ((float)(cell % grid_w) + 0.5f) / (float)grid_w;
			float cy = ((float)(cell / grid_w) + 0.5f) / (float)grid_h;
			float dx = t->x - cx;
			float dy = t->y - cy;
			float dist = dx * dx + dy * dy;
			
			if (best < 0 || dist < best_dist) {
				best = cell;
				best_dist = dist;
			}
		}
		
		if (best < 0) {
			dropped++;
			continue;
		}
		
		cell_target[best] = order[k];
		
		if (best != own) {
			moved_count++;
		}
	}
	
	if (moved != NULL) {
		*moved = moved_count;
	}
	
	return dropped;
}
