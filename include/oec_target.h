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

/*
 * Multiple targets for one image.
 *
 * OEC still has a single class, so class_id in the label is ignored and
 * every box becomes an object (confidence = 1).
 */
#define OEC_MAX_TARGETS OEC_MAX_BOXES

typedef struct {
	OEC_TARGET targets[OEC_MAX_TARGETS];
	int count;
} OEC_TARGETS;

/* Every box of the sample becomes a target. count == 0 for an empty image. */
OEC_TARGETS select_targets(const OEC_SAMPLE_DATA *data);

/*
 * Decide which grid cell is responsible for which target.
 *
 * Each cell predicts one box, so two objects whose centers fall in the
 * same cell cannot both use it. Targets are placed biggest first: each takes
 * the free cell whose center is closest to its own center (its own cell when
 * free). A colliding smaller object therefore moves to the nearest free
 * neighbor instead of being discarded.
 *
 * cell_target must hold grid_w * grid_h ints. On return cell_target[cell] is
 * the index of the target assigned to that cell, or -1.
 *
 * *moved (optional) receives how many targets were placed outside their
 * own cell. Returns the number of targets that could NOT be placed because
 * there are more targets than cells, or -1 on bad arguments.
 */
int oec_target_assign(
	const OEC_TARGETS *targets,
	int grid_w,
	int grid_h,
	int *cell_target,
	int *moved
);

#endif