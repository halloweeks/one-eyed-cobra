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
