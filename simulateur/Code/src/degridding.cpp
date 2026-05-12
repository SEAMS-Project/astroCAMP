#define _POSIX_C_SOURCE 199309L
#include "degridding.h"
#include "map.h"
#include <fftw3.h>
#include <time.h>
#include <stdio.h>


#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

float2 complex_mult(const float2 z1, const float2 z2) {
    float2 result;
    result.x = z1.x * z2.x - z1.y * z2.y;
    result.y = z1.y * z2.x + z1.x * z2.y;
    return result;
}

void iterator(int NUM_VISIBILITIES, int NUM_CHUNK, OUT int *out)
{

    if (NUM_VISIBILITIES % NUM_CHUNK != 0) {
        printf("Warning: NUM_VISIBILITIES (%d) is not divisible by NUM_CHUNK (%d)\n", NUM_VISIBILITIES, NUM_CHUNK);
    }

    int start = 0;
    //int num_chunks = NUM_VISIBILITIES / CHUNK;


    for (int i = 0; i < NUM_CHUNK; i++)
    {
        out[i] = start + (i * (NUM_VISIBILITIES/NUM_CHUNK));
        //printf("out[i] %d\n", out[i]);
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // ts.tv_sec = secondes depuis Epoch
    // ts.tv_nsec = nanosecondes (0..999,999,999)

    printf("Time start : %ld.%03ld seconds\n",
           ts.tv_sec, ts.tv_nsec / 1000000);
}


void std_degridding(int NUM_VISIBILITIES, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR,
		float2* input_grid, float3* vis_uvw_coords, int2* kernel_supports, float2* kernels, Config* config,
		float2* output_visibilities){
	
	memset(output_visibilities, 0, sizeof(float2) * NUM_VISIBILITIES);

	int grid_center = GRID_SIZE / 2;

	for(int i = 0; i < NUM_VISIBILITIES; ++i){

		// Calculate the w index based on the corrected z coordinate of the visibilities
		int w_idx = (int)(SQRT(ABS(vis_uvw_coords[i].z * config->w_scale)) + 0.5);

		// Retrieve the kernel thickness and offset for the w plane
		int half_support = kernel_supports[w_idx].x;
		int w_offset = kernel_supports[w_idx].y;

		// Calculate the grid position from the UV coordinates
		float2 grid_pos = {.x = vis_uvw_coords[i].x * config->uv_scale, .y = vis_uvw_coords[i].y * config->uv_scale};

		// Determine if the visibility should be conjugated based on the z coordinate
		float conjugate = (vis_uvw_coords[i].z < 0.0) ? -1.0 : 1.0;
		// Initialize the norm of the kernel coefficients for normalization
		float comm_norm = 0.f;

		// Iterate over the positions on the y-axis within the kernel window
        for(int v = CEIL(grid_pos.y - half_support); v < CEIL(grid_pos.y + half_support); ++v)
		{

        	// Correct the v index if necessary to respect the grid bounds
			int corrected_v = v;
			if(v < -grid_center){
				corrected_v = (-grid_center - v) - grid_center;
			}
			else if(v >= grid_center){
				corrected_v = (grid_center - v) + grid_center;
			}
			corrected_v = MAX(-grid_center, MIN(grid_center - 1, corrected_v));

			// Determine the kernel index on the y-axis
			int2 kernel_idx;
			kernel_idx.y = abs((int)ROUND((corrected_v - grid_pos.y) * OVERSAMPLING_FACTOR));

			// Iterate over the positions on the x-axis within the kernel window
			for(int u = CEIL(grid_pos.x - half_support); u < CEIL(grid_pos.x + half_support); ++u)
			{

				int corrected_u = u;

				if(u < -grid_center){
					corrected_u = (-grid_center - u) - grid_center;
				}
				else if(u >= grid_center){
					corrected_u = (grid_center - u) + grid_center;
				}
				corrected_u = MAX(-grid_center, MIN(grid_center - 1, corrected_u));

				// Determine the kernel index on the x-axis
				kernel_idx.x = abs((int)ROUND((corrected_u - grid_pos.x) * OVERSAMPLING_FACTOR));

				// Calculate the kernel index in the kernel array
				int k_idx = w_offset + kernel_idx.y * (half_support + 1) * OVERSAMPLING_FACTOR + kernel_idx.x;

				// Retrieve the kernel sample and apply conjugation if necessary
				float2 kernel_sample = kernels[k_idx];

				kernel_sample.y *= conjugate;

				// Calculate the input grid index to retrieve the value: we have a 2D array and access it as if it were 1D (I think)
				int grid_idx = (corrected_v + grid_center) * GRID_SIZE + (corrected_u + grid_center);

				// If the index is out of the grid, consider the input value as zero
				float2 input_value;

					input_value = input_grid[grid_idx];

				// Perform the complex product between the grid value and the kernel sample
				float2 prod = complex_mult(input_value, kernel_sample);

				// Accumulate the norm of the kernel samples for normalization
				comm_norm += kernel_sample.x * kernel_sample.x + kernel_sample.y * kernel_sample.y;

				// Add the product to the final visibility result
				output_visibilities[i].x += prod.x;
				output_visibilities[i].y += prod.y;
			}
		}
//    	printf("i:%d",i);

        // Calculate the norm of the kernels for normalization
		comm_norm = sqrt(comm_norm);

		// Normalize the visibilities to avoid values that are too small
		output_visibilities[i].x = comm_norm < 1e-5f ? output_visibilities[i].x / 1e-5f : output_visibilities[i].x / comm_norm;
		output_visibilities[i].y = comm_norm < 1e-5f ? output_visibilities[i].x / 1e-5f : output_visibilities[i].y / comm_norm;
	}

    printf("std_degridding\n");
}



void std_degridding_parallel(int GRID_SIZE, int NUM_VISIBILITIES, int NUM_KERNELS, int TOTAL_KERNELS_SAMPLES, int OVERSAMPLING_FACTOR, int NUM_CHUNK,
		float2* kernels, int2* kernel_supports, float2* input_grid, float3* vis_uvw_coords, Config* config, int* iterator,
		float2* output_visibilities){

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    printf("Time degrid start: %ld.%03ld seconds\n",
           ts.tv_sec, ts.tv_nsec / 1000000);

	memset(output_visibilities, 0, sizeof(float2)*(NUM_VISIBILITIES/NUM_CHUNK));

	int grid_center = GRID_SIZE / 2; //TODO possible to calculate once?

	for(int i = 0; i < (NUM_VISIBILITIES/NUM_CHUNK); ++i){


		// Calculate the w index based on the corrected z coordinate of the visibilities
		int w_idx = (int)(SQRT(ABS(vis_uvw_coords[i+(*iterator)].z * config->w_scale)) + 0.5);

		// Retrieve the kernel thickness and offset for the w plane
		int half_support = kernel_supports[w_idx].x;
		int w_offset = kernel_supports[w_idx].y;

		// Calculate the grid position from the UV coordinates
		float2 grid_pos = {.x = vis_uvw_coords[i+(*iterator)].x * config->uv_scale, .y = vis_uvw_coords[i+(*iterator)].y * config->uv_scale};

		// Determine if the visibility should be conjugated based on the z coordinate
		float conjugate = (vis_uvw_coords[i+(*iterator)].z < 0.0) ? -1.0 : 1.0;
		// Initialize the norm of the kernel coefficients for normalization
		float comm_norm = 0.f;

		// Iterate over the positions on the y-axis within the kernel window
        for(int v = CEIL(grid_pos.y - half_support); v < CEIL(grid_pos.y + half_support); ++v)
		{

        	// Correct the v index if necessary to respect the grid bounds
			int corrected_v = v;
			if(v < -grid_center){
				corrected_v = (-grid_center - v) - grid_center;
			}
			else if(v >= grid_center){
				corrected_v = (grid_center - v) + grid_center;
			}
			corrected_v = MAX(-grid_center, MIN(grid_center - 1, corrected_v));

			int kernel_width = (half_support + 1) * OVERSAMPLING_FACTOR;

			// Determine the kernel index on the y-axis
			int2 kernel_idx;
			kernel_idx.y = abs((int)ROUND((corrected_v - grid_pos.y) * OVERSAMPLING_FACTOR));
			kernel_idx.y = MIN(kernel_width - 1, kernel_idx.y);

			// Iterate over the positions on the x-axis within the kernel window
			for(int u = CEIL(grid_pos.x - half_support); u < CEIL(grid_pos.x + half_support); ++u)
			{

				int corrected_u = u;

				if(u < -grid_center){
					corrected_u = (-grid_center - u) - grid_center;
				}
				else if(u >= grid_center){
					corrected_u = (grid_center - u) + grid_center;
				}
				corrected_u = MAX(-grid_center, MIN(grid_center - 1, corrected_u));

				// Determine the kernel index on the x-axis
				kernel_idx.x = abs((int)ROUND((corrected_u - grid_pos.x) * OVERSAMPLING_FACTOR));
				kernel_idx.x = MIN(kernel_width - 1, kernel_idx.x);

				// Calculate the kernel index in the kernel array
				int k_idx = w_offset + kernel_idx.y * (half_support + 1) * OVERSAMPLING_FACTOR + kernel_idx.x;

                // Retrieve the kernel sample and apply conjugation if necessary
                float2 kernel_sample = {0};
                if(!(k_idx < 0 || k_idx > TOTAL_KERNELS_SAMPLES))
					kernel_sample = kernels[k_idx];

				kernel_sample.y *= conjugate;

				// Calculate the input grid index to retrieve the value: we have a 2D array and access it as if it were 1D (I think)
				int grid_idx = (corrected_v + grid_center) * GRID_SIZE + (corrected_u + grid_center);

				// If the index is out of the grid, consider the input value as zero
				float2 input_value;

					input_value = input_grid[grid_idx];

				// Perform the complex product between the grid value and the kernel sample
				float2 prod = complex_mult(input_value, kernel_sample);

				// Accumulate the norm of the kernel samples for normalization
				comm_norm += kernel_sample.x * kernel_sample.x + kernel_sample.y * kernel_sample.y;

				// Add the product to the final visibility result
				output_visibilities[i].x += prod.x;
				output_visibilities[i].y += prod.y;
			}
		}

        // Calculate the norm of the kernels for normalization
		comm_norm = sqrt(comm_norm);

		// Normalize the visibilities to avoid values that are too small
		output_visibilities[i].x = comm_norm < 1e-5f ? output_visibilities[i].x / 1e-5f : output_visibilities[i].x / comm_norm;
		output_visibilities[i].y = comm_norm < 1e-5f ? output_visibilities[i].y / 1e-5f : output_visibilities[i].y / comm_norm;
	}
    struct timespec te;
    clock_gettime(CLOCK_REALTIME, &te);

    printf("Time degrid end: %ld.%03ld seconds\n",
           te.tv_sec, te.tv_nsec / 1000000);
}

/*
void convert_vis_to_csv_test(int NUM_VISIBILITIES,float2* output_visibilities, float3* vis_uvw_coords,Config *config) {

	FILE* file = fopen("output/output_degridder_pre_grid.csv", "w");
	if (file == NULL) {
		handle_file_error(file, "Error opening file");
	}

	// Write the number of visibilities in the first line
	if (fprintf(file, "%d\n", NUM_VISIBILITIES) < 0) {
		handle_file_error(file, "Error writing number of visibilities");
	}

	for (int i = 0; i < NUM_VISIBILITIES; i++) {
		// Check for NaN values
		if (isnan(vis_uvw_coords[i].x) || isnan(vis_uvw_coords[i].y) || isnan(vis_uvw_coords[i].z) ||
			isnan(output_visibilities[i].x) || isnan(output_visibilities[i].y)) {
			fprintf(stderr, "Error: NaN at index %d.\n", i);
			handle_file_error(file, "Error: invalid data");
			}

		// Write data to the CSV file
		if (fprintf(file, "%.6f %.6f %.6f %.6f %.6f 1\n",
					vis_uvw_coords[i].x,
					vis_uvw_coords[i].y,
					vis_uvw_coords[i].z,
					output_visibilities[i].x,
					output_visibilities[i].y) < 0) {
			handle_file_error(file, "Error writing in the file");
					}
	}

	// Close the file
	if (fclose(file) != 0) {
		handle_file_error(file, "Error closing the file");
	}

	//printf("convert_vis_to_csv\n");
}
*/

void gridding_CPU(int NUM_VISIBILITIES, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNELS_SAMPLES, int OVERSAMPLING_FACTOR,
				IN float2 *kernel, IN int2 *supports,
				IN float3 *vis_uvw_coord, float2* vis,
				IN Config * config,  IN float2 *grid_in, OUT float2 *grid)
{

/*
	memcpy(grid, grid_in, GRID_SIZE * GRID_SIZE * sizeof(float2));


    int nb_skipped=0;


    for (unsigned int vis_index = 0; vis_index < NUM_VISIBILITIES; vis_index++)
    {

        // Represents index of w-projection kernel in supports array
        const int plane_index = (int) ROUND(SQRT(ABS(vis_uvw_coord[vis_index].z * config->w_scale)));

		if (plane_index < 0 || plane_index >= NUM_KERNELS){
			nb_skipped++;
			continue; //added to correct the segfault (the result is still ok and the gridding is not the actor I work on, it is used to verifiy the result)
		}

        // Scale visibility uvw into grid coordinate space
        const float2 grid_coord = make_float2(
        		vis_uvw_coord[vis_index].x * config->uv_scale,
				vis_uvw_coord[vis_index].y * config->uv_scale
        );
        const int half_grid_size = GRID_SIZE / 2;
        const int half_support = supports[plane_index].x;

        float conjugate = (vis_uvw_coord[vis_index].z < 0.0) ? -1.0 : 1.0;


        const float2 snapped_grid_coord = make_float2(
                ROUND(grid_coord.x * OVERSAMPLING_FACTOR) / OVERSAMPLING_FACTOR,
                ROUND(grid_coord.y * OVERSAMPLING_FACTOR) / OVERSAMPLING_FACTOR
        );

        const float2 min_grid_point = make_float2(
                CEIL(snapped_grid_coord.x - half_support),
                CEIL(snapped_grid_coord.y - half_support)
        );

        const float2 max_grid_point = make_float2(
                FLOOR(snapped_grid_coord.x + half_support),
                FLOOR(snapped_grid_coord.y + half_support)
        );

        float2 convolved = make_float2(0.0, 0.0);
        float2 kernel_sample = make_float2(0.0, 0.0);
        int2 kernel_uv_index = make_int2(0, 0);

        int grid_index = 0;
        int kernel_index = 0;
        int w_kernel_offset = supports[plane_index].y;

        // printf("%lf \t %lf\n", max_grid_point.x - min_grid_point.x, max_grid_point.y - min_grid_point.y);

        for(int grid_v = min_grid_point.y; grid_v <= max_grid_point.y; ++grid_v)
        {
        	if(grid_v < -half_grid_size || grid_v >= half_grid_size){
				continue;
			}

            kernel_uv_index.y = abs((int)ROUND((grid_v - snapped_grid_coord.y) * OVERSAMPLING_FACTOR));

            for(int grid_u = min_grid_point.x; grid_u <= max_grid_point.x; ++grid_u)
            {
            	if(grid_u < -half_grid_size || grid_u >= half_grid_size){
					continue;
				}

                kernel_uv_index.x = abs((int)ROUND((grid_u - snapped_grid_coord.x) * OVERSAMPLING_FACTOR));

                kernel_index = w_kernel_offset + kernel_uv_index.y * (half_support + 1)
                                                 * OVERSAMPLING_FACTOR + kernel_uv_index.x;
                if (kernel_index < 0 || kernel_index >= TOTAL_KERNELS_SAMPLES) {
                    printf("Out of bounds kernel access: %d (max: %d)\n", kernel_index, TOTAL_KERNELS_SAMPLES);

                }

                kernel_sample = make_float2(kernel[kernel_index].x, kernel[kernel_index].y  * conjugate);

                grid_index = (grid_v + half_grid_size) * GRID_SIZE + (grid_u + half_grid_size);

                convolved = complex_mult(vis[vis_index], kernel_sample);

                grid[grid_index].x += convolved.x;
                grid[grid_index].y += convolved.y;
                // atomicAdd(&(grid[grid_index].x), convolved.x);
                // atomicAdd(&(grid[grid_index].y), convolved.y);
            }
        }

    }

	printf("nb_skipped:%d\n", nb_skipped);

    printf("FINISHED GRIDDING\n");
*/

}

