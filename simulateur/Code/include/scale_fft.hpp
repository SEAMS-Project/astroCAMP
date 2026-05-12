#ifndef SCALE_FFT_HPP
#define SCALE_FFT_HPP

#include "fft.hpp"

// Comment syntax:

// In this code whenever you see a variable with _s it means it's a stream, _r it's a real scalar, _i it's an imaginary scalar
// Sometimes in functions where we mix floats and ap_fixed, f_ means it's a float

// Stream consumed by actor ../A means that it's consumed by an actor A prior to the current function scope
// Stream has to be consumed by actor ++/A means the stream is intended to be consumed by an actor A that comes after the current function scope

template<int N>
void maxStream(hls::stream<std::complex<float>> &in,
		hls::stream<float> &max_s) {
	float max = 0;
	loop_copyStream: for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE
		std::complex<float> temp = in.read();
		float tempf_r = temp.real();
		float tempf_i = temp.imag();
		if (std::abs(tempf_r) > max)
			max = std::abs(tempf_r);       //Calculating max
		if (std::abs(tempf_i) > max)
			max = std::abs(tempf_i);       //---------------
	}
	if (max == 0)
		max = 1;     //Avoid division by 0
	max_s.write(max); //Each line has it's own max now and it is ordered in the stream
}

// -inf_s is an INPUT stream that contains float values of the data to scale down, this stream is produced by ../copyStreamAndCalcMax
// -out_s is an OUTPUT stream that contains fixed-point values of the data after scaling it down, this stream should be consumed by myfftwrapper
// -max_s is an INPUT stream that contains the max value of each line, this stream is produced by ../copyStreamAndCalcMax
// -tmax2 is the max of a line that should be transfered to the upscale after the fft
// This function downscales the data by tmax2 (which is read from tmax stream) it also implicitly converts float to ap_fixed for the fft
// We could read the tmax in rcfft function directly and keep the variable instead of transferrin its reference,
// But to make the canonical region as clean as possible ((only sequential function calls)) we do read it here
template<int N>
void downscale(hls::stream<std::complex<float>> &inf_s,
		hls::stream<cmpxDataIn> &out_s, hls::stream<float> &max_s) {
#pragma HLS inline off
	float max = max_s.read();
	loop_downscale: for (int i = 0; i < N; ++i) { //For each line we downscale every element by the max of the line * FFT_LENGTH (We dont need to divide by FFT_LENGTH again because the ip does it implicitly)
		std::complex<float> temp = inf_s.read();
		float tempf_r = (temp.real() / max) / N;
		float tempf_i = (temp.imag() / max) / N;
		data_in_t temp_r = tempf_r;
		data_in_t temp_i = tempf_i;
		out_s.write(cmpxDataIn(temp_r, temp_i));
	}
}

// -in_s  is an INPUT stream that contains the data after FFT, this stream is produced by myfftwrapper
// -outf  in an OUTPUT array that contains the data after being scaled up to normal range, this array should be used by ++/copyToMem
// -max   is the value that we downscaled with
// This function upscales the data back to normal range and also implicitly converts ap_fixed to float
template<int N>
void upscale(hls::stream<cmpxDataIn> &in_s, hls::stream<std::complex<float>> &outf,
		hls::stream<float> &max) {
#pragma HLS inline off
	const float SQRT_N = sqrt(FFT_LENGTH);
	float coeff = 2 * max.read() * N * SQRT_N; // Don't ask how i got it, only god knows, fft ip does some shitty weird scaling down on it's own
	loop_upscale: for (int i = 0; i < N; ++i) { //For each line upscale each element by a const factor we got from downscale func
		cmpxDataIn temp = in_s.read();
		float temp_r = temp.real();
		float temp_i = temp.imag();
		temp_r = temp_r * coeff;
		temp_i = temp_i * coeff;
		outf.write(std::complex<float>(temp_r, temp_i));
	}
}

/**
 * Naive implementation of transposition using local storage (BRAM or REG) in FPGAs
 */
template<int N, typename T>
void transposeStream(hls::stream<T> &in_s, hls::stream<T> &out_s) {
	T buffer[N][N];
	// Read data from input stream and store them in local buffer
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			buffer[i][j] = in_s.read();
		}
	}
	// Write data to output stream with transposition
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			out_s.write(buffer[j][i]);
		}
	}
}


#endif //SCALE_FFT_HPP
