#ifndef OEC_TARGET_H
#define OEC_TARGET_H

#include "oec_dataset.h"

typedef struct {
	float confidence;
	float x;
	float y;
	float w;
	float h;
} OEC_TARGET;

OEC_TARGET select_target(const OEC_SAMPLE_DATA*);

#endif