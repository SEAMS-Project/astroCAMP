#ifndef TOP_H
#define TOP_H

#ifndef CPU_VERSION
#define CPU_VERSION 1
#endif


#ifdef __cplusplus
extern "C" {
#endif

	#include "preesm.h"

	void end_sink(int NUM_RECEIVERS, IN float2 *gains);

	void config_struct_set_up(int GRID_SIZE, int NUM_KERNELS, int NUM_SCENARIO, OUT Config *config_struct);

	void load_image_from_file(int GRID_SIZE, Config* config, float *output);

	//void generate_kernel(int NUM_KERNELS, int OVERSAMPLING_FACTOR, int NUM_KERNEL_SUPPORT, int TOTAL_KERNELS_SAMPLES, Config* config,
	//    float2* kernels, int2* kernel_supports);

	// void degridding_kernel_host_set_up(int TOTAL_KERNELS_SAMPLES, int NUM_KERNELS, int OVERSAMPLING_FACTOR, IN Config *config, OUT float2 *kernels, OUT int2 *kernel_supports);
	void generate_kernel(int NUM_KERNELS, int OVERSAMPLING_FACTOR, int NUM_KERNEL_SUPPORT, int TOTAL_KERNELS_SAMPLES,
	    float2* kernels, int2* kernel_supports);
	void save_degridding_kernels_to_csv(float2* degridding_kernels, int TOTAL_KERNELS_SAMPLES);
	void save_degridding_kernel_supports_to_csv(int2* degridding_kernel_supports, int NUM_KERNELS);

	float kaiser_bessel(float x, float beta);
	float I0(float x);
	float pswf(float x);

	double randn(double mean, double stddev);

	void visibility_host_set_up(int NUM_VISIBILITIES, int GRID_SIZE, IN Config *config, OUT float3 *vis_uvw_coords);

	void visibility_host_set_up2(int NUM_VISIBILITIES, Config* config,float3* vis_uvw_coords);


	void sink(float2* measured_vis);
	void save_visibilities_to_csv(const char* filename, float3* vis_uvw_coords, int num_visibilities);


	void handle_file_error(FILE *file, const char *message);

	void convert_vis_to_csv(int NUM_VISIBILITIES,PRECISION2* output_visibilities, PRECISION3* vis_uvw_coords,Config *config);


	void export_image_to_csv(IN float *image_grid, IN Config * config, int GRID_SIZE);
	void export_image_to_csv_input_grid(float2 *image_grid, int GRID_SIZE);
	void export_image_to_csv_uv_grid(float2 *image_grid, int GRID_SIZE);
	void export_image_to_csv_uv_grid_shift(float2 *image_grid, int GRID_SIZE);
	void export_image_to_csv_gridding(float2 *image_grid, int GRID_SIZE);
	void export_image_to_csv_reconstructed(float2 *image_grid, int GRID_SIZE);
	void export_image_to_csv_ifft_test(float2 *image_grid, int GRID_SIZE);

	void std_degridding(int NUM_VISIBILITIES, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR,
		float2* input_grid, float3* vis_uvw_coords, int2* kernel_supports, float2* kernels, Config* config,
		float2* output_visibilities);



	void gridding_CPU(int NUM_VISIBILITIES, int GRID_SIZE, int NUM_KERNELS, int TOTAL_KERNEL_SAMPLES, int OVERSAMPLING_FACTOR,
					IN float2 *kernel, IN int2 *supports,
	                  IN float3 *vis_uvw_coord, IN float2 *vis_uvw,
					  IN Config * config,  IN float2 *grid_in, OUT float2 *grid);


#ifdef __cplusplus
}
#endif


#endif
