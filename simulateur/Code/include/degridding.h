#ifndef DEGRIDDER_H
#define DEGRIDDER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "preesm.h"

	void subtract_image_space(int GRID_SIZE, float* measurements, float* estimate, float* result);

	void correct_to_finegrid(int NUM_VISIBILITIES, int GRID_SIZE, int OVERSAMPLING_FACTOR, int PERFORM_SIMPLIFICATION, float3* vis_uvw_coords, float2* input_visibilities,
			Config* config, float2* output_visibilities, float3* output_finegrid_vis_coords, int* num_output_visibilities);

	
	void std_degridding(int NUM_VISIBILITIES, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR,
		float2* input_grid, float3* vis_uvw_coords, int2* kernel_supports, float2* kernels, Config* config,
		float2* output_visibilities);



	void gridding_CPU(int NUM_VISIBILITIES, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR,
					IN float2 *kernel, IN int2 *supports,
	                  IN float3 *vis_uvw_coord, IN float2 *vis_uvw,
					  IN Config * config,  IN float2 *grid_in, OUT float2 *grid);

	void convert_vis_to_csv_test(int NUM_VISIBILITIES,float2* output_visibilities, float3* vis_uvw_coords,Config *config);


	void iterator(int NUM_VISIBILITIES, int NUM_CHUNK, OUT int *out);

	void std_degridding_parallel(int GRID_SIZE, int NUM_VISIBILITIES, int NUM_KERNELS, int TOTAL_KERNELS_SAMPLES, int OVERSAMPLING_FACTOR, int NUM_CHUNK,
			float2* kernels, int2* kernel_supports, float2* input_grid, float3* vis_uvw_coords, Config* config, int* iterator,
			float2* output_visibilities);

#ifdef __cplusplus
}
#endif


#endif
