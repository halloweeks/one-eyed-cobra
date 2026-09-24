#include "oec_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>

#include <stdlib.h>
#include <string.h>


// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//#include "oec_dataset.h"


int oec_dataset_load_image(const char *path, OEC_TENSOR *image)
{
	if (path == NULL || image == NULL) {
		return -1;
	}
	
	if (image->data == NULL) {
		return -1;
	}
	
	if (image->c != 3) {
		return -1;
	}
	
	int src_w;
	int src_h;
	int src_c;
	
	unsigned char *src = stbi_load(path, &src_w, &src_h, &src_c, 3);
	
	if (src == NULL) {
		return -1;
	}
	
	/*
	 * Simple nearest-neighbor resize
	 * and HWC uint8 -> CHW float conversion.
	 */
	
	for (int y = 0; y < image->h; y++) {
		int sy = (y * src_h) / image->h;
		
		for (int x = 0; x < image->w; x++) {
			int sx = (x * src_w) / image->w;
			
			size_t src_index = ((size_t)sy * (size_t)src_w + (size_t)sx) * 3;
			
			float r = src[src_index + 0] / 255.0f;
			float g = src[src_index + 1] / 255.0f;
			float b = src[src_index + 2] / 255.0f;
			
			size_t pixel = (size_t)y * (size_t)image->w + (size_t)x;
			
			image->data[
				(size_t)0 * image->h * image->w + pixel
			] = r;
			
			image->data[
				(size_t)1 * image->h * image->w + pixel
			] = g;
			
			image->data[
				(size_t)2 * image->h * image->w + pixel
			] = b;
		}
	}
	
	stbi_image_free(src);
	
	return 0;
}

OEC_SAMPLE_DATA oec_dataset_load_label(const char *path)
{
	
	OEC_SAMPLE_DATA res = {0};
	
	if (path == NULL) {
		printf("failed\n");
		return res;
	}
	
	FILE *fp = fopen(path, "r");
	
	if (fp == NULL) {
		printf("failed\n");
		return res;
	}
	
	char line[512];
	size_t i = 0;
	
	
	while (fgets(line, sizeof(line), fp)) {
		if (i >= OEC_MAX_BOXES) {
			fprintf(stderr, "Maximum box count exceed!\n");
			break;
		}
		
		
		if (sscanf(
			line,
			"%d %f %f %f %f",
			&res.boxes[i].class_id,
			&res.boxes[i].x_center,
			&res.boxes[i].y_center,
			&res.boxes[i].width,
			&res.boxes[i].height
		) != 5) {
			fprintf(stderr, "Invalid label line: %s", line);
            continue;
		}
		
		i++;
	}
	
	res.box_count = i;
	
	fclose(fp);
	
	return res;
}


OEC_SAMPLE_DATA oec_dataset_load_sample(
    const OEC_SAMPLE *sample,
    OEC_TENSOR *image
)
{
    OEC_SAMPLE_DATA res = {0};

    if (sample == NULL || image == NULL)
        return res;

    /*
     * Load image into the already allocated tensor.
     */
    if (oec_dataset_load_image(
            sample->image_path,
            image) != 0) {
        return res;
    }

    /*
     * Load all bounding boxes.
     */
    FILE *fp = fopen(sample->label_path, "r");

    if (fp == NULL)
        return res;

    char line[512];

    while (fgets(line, sizeof(line), fp)) {

        if (res.box_count >= OEC_MAX_BOXES) {
            fprintf(stderr,
                    "Maximum box count exceeded!\n");
            break;
        }

        OEC_BBox *box =
            &res.boxes[res.box_count];

        if (sscanf(
                line,
                "%d %f %f %f %f",
                &box->class_id,
                &box->x_center,
                &box->y_center,
                &box->width,
                &box->height
            ) != 5) {

            fprintf(stderr,
                    "Invalid label line: %s",
                    line);

            continue;
        }

        res.box_count++;
    }

    fclose(fp);

    return res;
}

static int is_image_file(const char *name)
{
    const char *dot = strrchr(name, '.');

    if (!dot)
        return 0;

    if (strcmp(dot, ".jpg") == 0)
        return 1;

    if (strcmp(dot, ".jpeg") == 0)
        return 1;

    if (strcmp(dot, ".png") == 0)
        return 1;

    return 0;
}

static size_t count_images(const char *path)
{
    DIR *dir = opendir(path);

    if (!dir)
        return 0;

    ssize_t count = 0;
    
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG)
            continue;

        if (is_image_file(entry->d_name))
            count++;
    }

    closedir(dir);

    return count;
}

static void make_image_path(
    char *dst,
    size_t dst_size,
    const char *directory,
    const char *filename)
{
    snprintf(
        dst,
        dst_size,
        "%s/%s",
        directory,
        filename
    );
}

static void make_label_path(
    char *dst,
    size_t dst_size,
    const char *directory,
    const char *image_filename)
{
    const char *dot = strrchr(image_filename, '.');

    size_t basename_len;

    if (dot)
        basename_len = (size_t)(dot - image_filename);
    else
        basename_len = strlen(image_filename);

    snprintf(
        dst,
        dst_size,
        "%s/%.*s.txt",
        directory,
        (int)basename_len,
        image_filename
    );
}

static int load_split(OEC_DATASET_SPLIT *split,const char *images_dir, const char *labels_dir)
{
	ssize_t count = count_images(images_dir);
	
	split->samples = NULL;
	split->count = 0;
	
	if (count == 0) {
		return 0;
	}

    split->samples = calloc(
        count,
        sizeof(OEC_SAMPLE)
    );

    if (!split->samples)
        return -1;

    DIR *dir = opendir(images_dir);

    if (!dir) {
        free(split->samples);
        split->samples = NULL;
        return -1;
    }

    ssize_t index = 0;

    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {

        if (entry->d_type != DT_REG)
            continue;

        if (!is_image_file(entry->d_name))
            continue;

        if (index >= count)
            break;
         
        OEC_SAMPLE *sample = &split->samples[index];
        
       
make_image_path(
    sample->image_path,
    OEC_PATH_MAX,
    images_dir,
    entry->d_name
);

make_label_path(
    sample->label_path,
    OEC_PATH_MAX,
    labels_dir,
    entry->d_name
);

index++;
        
        
        
    }

    closedir(dir);

    split->count = index;

    return 0;
}

static void free_split(OEC_DATASET_SPLIT *split)
{
    if (!split)
        return;


    free(split->samples);

    split->samples = NULL;
    split->count = 0;
}

OEC_DATASET *oec_dataset_open(const char *root)
{
    if (!root)
        return NULL;

    OEC_DATASET *dataset =
        calloc(1, sizeof(OEC_DATASET));

    if (!dataset)
        return NULL;

    char train_images[1024];
    char train_labels[1024];

    char valid_images[1024];
    char valid_labels[1024];

    char test_images[1024];
    char test_labels[1024];

    snprintf(
        train_images,
        sizeof(train_images),
        "%s/train/images",
        root
    );

    snprintf(
        train_labels,
        sizeof(train_labels),
        "%s/train/labels",
        root
    );

    snprintf(
        valid_images,
        sizeof(valid_images),
        "%s/valid/images",
        root
    );

    snprintf(
        valid_labels,
        sizeof(valid_labels),
        "%s/valid/labels",
        root
    );

    snprintf(
        test_images,
        sizeof(test_images),
        "%s/test/images",
        root
    );

    snprintf(
        test_labels,
        sizeof(test_labels),
        "%s/test/labels",
        root
    );

    if (load_split(
            &dataset->train,
            train_images,
            train_labels) != 0)
        goto fail;

    if (load_split(
            &dataset->valid,
            valid_images,
            valid_labels) != 0)
        goto fail;

    if (load_split(
            &dataset->test,
            test_images,
            test_labels) != 0)
        goto fail;

    return dataset;

fail:
    free_split(&dataset->train);
    free_split(&dataset->valid);
    free_split(&dataset->test);

    free(dataset);

    return NULL;
}

void oec_dataset_close(OEC_DATASET *dataset)
{
	if (!dataset) 
		return;
	
	free_split(&dataset->train);
	free_split(&dataset->valid);
	free_split(&dataset->test);
	
	free(dataset);
}