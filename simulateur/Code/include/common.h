                   // Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>


#ifdef __cplusplus
extern "C" {
#endif


#ifndef COMMON_H_
#define COMMON_H_  


#include <math.h>
#include "preesm.h"

#ifdef __NVCC__
	#include <cuda.h>
	#include <cuda_runtime_api.h>
	#include <device_launch_parameters.h>
	#include <cufft.h>

	#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)
	#define CUFFT_SAFE_CALL(err) cufft_safe_call(err, __FILE__, __LINE__)
#endif

#ifndef SPEED_OF_LIGHT
	#define SPEED_OF_LIGHT 299792458.0
#endif

#ifndef SINGLE_PRECISION
	#define SINGLE_PRECISION 1
#endif

#if SINGLE_PRECISION
	#define PRECISION float
	#define PRECISION2 float2
	#define PRECISION3 float3
	#define PRECISION4 float4
	#define PI ((float) 3.141592654)

	#ifdef __NVCC__
		#define CUFFT_C2C_PLAN CUFFT_C2C
		#define CUFFT_C2P_PLAN CUFFT_C2R
	#endif

	#else
		#define PRECISION double
		#define PRECISION2 double2
		#define PRECISION3 double3
		#define PRECISION4 double4
		#define PI ((double) 3.1415926535897931)

	#ifdef __NVCC__
		#define CUFFT_C2C_PLAN CUFFT_Z2Z
		#define CUFFT_C2P_PLAN CUFFT_Z2D
	#endif

#endif

#if SINGLE_PRECISION
	#define SIN(x) sinf(x)
	#define COS(x) cosf(x)
	#define SINCOS(x, y, z) sincosf(x, y, z)
	#define ABS(x) fabsf(x)
	#define SQRT(x) sqrtf(x)
	#define ROUND(x) roundf(x)
	#define CEIL(x) ceilf(x)
	#define LOG(x) logf(x)
	#define POW(x, y) powf(x, y)
	#define FLOOR(x) floorf(x)
	#define MAKE_PRECISION2(x,y) make_float2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)

	#ifdef __NVCC__
		#define CUFFT_EXECUTE_C2P(a,b,c) cufftExecC2R(a,b,c)
		#define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecC2C(a,b,c,d)
		#define SVD_BUFFER_SIZE(a,b,c,d) cusolverDnSgesvd_bufferSize(a,b,c,d)
		#define SVD(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusolverDnSgesvd(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)
	#endif

#else
	#define SIN(x) sin(x)
	#define COS(x) cos(x)
	#define SINCOS(x, y, z) sincos(x, y, z)
	#define ABS(x) fabs(x)
	#define SQRT(x) sqrt(x)
	#define ROUND(x) round(x)
	#define CEIL(x) ceil(x)
	#define LOG(x) log(x)
	#define POW(x, y) pow(x, y)
	#define FLOOR(x) floor(x)
	#define MAKE_PRECISION2(x,y) make_double2(x,y)
	#define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
	#define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)

	#ifdef __NVCC__
	#define CUFFT_EXECUTE_C2P(a,b,c) cufftExecZ2D(a,b,c)
	#define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecZ2Z(a,b,c,d)
	#define SVD_BUFFER_SIZE(a,b,c,d) cusolverDnDgesvd_bufferSize(a,b,c,d)
	#define SVD(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) cusolverDnDgesvd(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)
	#endif
#endif

//#ifndef __cplusplus
	 typedef struct  {
		 int x;
		 int y;
	 } int2;


		 typedef struct  {
			 float x;
			 float y;
		 } float2;

		 typedef struct  {
			 float x;
			 float y;
			 float z;
		 } float3;



		 typedef struct  {
			 double x;
			 double y;
		 } double2;

		 typedef struct  {
			 double x;
			 double y;
			 double z;
		 } double3;
#ifndef __NVCC__
	int2 make_int2(int x,int y);

#if SINGLE_PRECISION

	float2 make_float2(float x, float y);
	float3 make_float3(float x, float y, float z);

#else

	double2 make_double2(double x,double y);
	double3 make_double3(double x,double y, double z);
//#endif
#endif

#endif

typedef struct Complex {
	PRECISION real;
	PRECISION imaginary;
} Complex;

typedef struct Visibility {
	PRECISION u;
	PRECISION v;
	PRECISION w;
} Visibility;

typedef struct Source {
	PRECISION l;
	PRECISION m;
	PRECISION intensity;
} Source;

	typedef struct Config {

		// general
		bool retain_device_mem;
		int num_major_cycles;
		int num_recievers;
		int num_baselines;
		int grid_size;
		double cell_size;
		const char *output_path;
		const char *dirty_image_output;
		int gpu_max_threads_per_block;
		int gpu_max_threads_per_block_dimension;
		bool right_ascension;

		bool save_dirty_image;
		bool save_residual_image;
		bool save_extracted_sources;
		bool save_predicted_visibilities;
		bool save_estimated_gains;

		// Testing
		bool perform_system_test;
		const char *system_test_image;
		const char *system_test_sources;
		const char *system_test_visibilities;
		const char *system_test_gains;

		// Gains
		const char *default_gains_file;
		const char *output_gains_file;
		bool perform_gain_calibration;
		bool use_default_gains;
		int max_calibration_cycles;
		int number_cal_major_cycles;

		// griddingf
		bool force_weight_to_one;
		double frequency_hz;
		int oversampling;
		double uv_scale;
		int num_visibilities;
		int num_kernels;
		int total_kernel_samples;
		double max_w;
		double w_scale;
		const char *input;
		const char *degridding_kernel_real_file;
		const char *degridding_kernel_imag_file;
		const char *degridding_kernel_support_file;
		const char *kernel_real_file;
		const char *kernel_imag_file;
		const char *kernel_support_file;
		const char *visibility_source_file;
		const char *output_degridder;
		const char *output_ifft;
		const char *output_fft_test;

		// deconvolution
		unsigned int number_minor_cycles_cal;
		unsigned int number_minor_cycles_img;
		double loop_gain;
		double weak_source_percent_gc;
		double weak_source_percent_img;
		double noise_factor;
		double psf_max_value;
		const char *model_sources_output;
		const char *residual_image_output;
		const char *psf_input_file;
		int num_sources;

		// Direct Fourier Transform
		const char *predicted_vis_output;

	} Config;

	void consume_loop_token(IN char *loop_token);

	void loop_iterator(int NB_ITERATION, OUT int *cycle_count);



	#ifdef __NVCC__
		void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

		void cufft_safe_call(cufftResult err, const char *file, const int line);

		const char* cuda_get_error_enum(cufftResult error);
	#endif


	#define MAX(x, y) (((x) > (y)) ? (x) : (y))
	#define MIN(x, y) (((x) < (y)) ? (x) : (y))

	#endif /* COMMON_H_ */

	#ifdef __cplusplus
	}
	#endif
