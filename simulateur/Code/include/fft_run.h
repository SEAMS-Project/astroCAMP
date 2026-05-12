#ifndef FFT_RUN_H
#define FFT_RUN_H

//#include <cufft.h>
//#include <cuda_runtime.h>


#ifdef __cplusplus
#include <complex>
extern "C" {
#endif
	#include "preesm.h"
    #include <fftw3.h>

	void CUFFT_EXECUTE_INVERSE_C2C_actor(int GRID_SIZE, float2 *uv_grid_in, float2 *uv_grid_out);


	void fft_shift_complex_to_complex_actor(int GRID_SIZE, float2 *uv_grid_in, float2 *uv_grid_out);

	void CUFFT_EXECUTE_FORWARD_C2C_actor(int GRID_SIZE, IN float2* uv_grid_in, OUT float2* uv_grid_out);

	void fft_shift_real_to_complex_actor(int GRID_SIZE, float *input_grid, Config *config, float2 *fourier);
	void fft_shift_complex_to_real_actor(int GRID_SIZE, float2 *uv_grid, float * dirty_image);


	void fft1d(int N, float2 *Input, float2 *Output);
	void fft2d_parallel(int N, int nbCore, float2* in, float2* out);
	void transpose(int N, float2* in, float2* out);


	#ifdef __cplusplus
	void transpose_ff(int N, float2* in, float2* out);
	void transpose_fc(int N, float2* in, std::complex<float>* out);
	void transpose_cf(int N, std::complex<float>* in, float2* out);
	void transpose_cc(int N, std::complex<float>* in, std::complex<float>* out);

	void fft1d_ff(int N, float2 *Input, float2 *Output);
	void fft1d_fc(int N, float2 *Input, std::complex<float> *Output);
	void fft1d_cf(int N, std::complex<float> *Input, float2 *Output);
	void fft1d_cc(int N, std::complex<float> *Input, std::complex<float> *Output);

	void maxArray(int N, std::complex<float>* in, std::complex<float>* max_s);
	void downscale(int N, std::complex<float>* inf_s, std::complex<float>* max_s, std::complex<float>* out_s);
	void upscale(int N, std::complex<float>* in_s, std::complex<float>* max_s, std::complex<float>* outf);
	#else
	#endif



#ifdef __NVCC__
	__global__ void fft_shift_complex_to_complex(float2 *grid, const int width);

	__global__ void fft_shift_complex_to_real(float2 *grid, float *image, const int width);
#endif


#ifdef __cplusplus
}
#endif




#endif
