#ifndef FFT_HPP
#define FFT_HPP

#include "hls_fft.h"
#include "hls_stream.h"
#include <complex>

// configurable params
const char FFT_INPUT_WIDTH = 32; // nombre de bits des données
const char FFT_OUTPUT_WIDTH = 32; // nombre de bits des données
const char FFT_CONFIG_WIDTH = 8; // taille de la struct (ne pas toucher sauf si vitis râle. possibles : 8 ou 16)
const char FFT_NFFT_MAX = 6; // log du nombre de points
constexpr int  FFT_LENGTH = 1 << FFT_NFFT_MAX; // nombre de points. DOIT être un multiple de MAX_BURST (ou MAX_BURST un diviseur de FFT_LENGTH).
const int  FWD_INV = 0; // direction de fft (directe ou inverse)
//constexpr int  MAX_BURST = FFT_LENGTH > 256 ? 256:FFT_LENGTH; // MAX_BURST semble devoir être inférieur ou égal à FFT_LENGTH. Je le cappe arbitrairement à 256 pour le moment, mais ça peut changer.
//const int  PARALLEL = 4; // taille du burst en nombre de tokens


struct config1: hls::ip_fft::params_t {
	static const unsigned ordering_opt = hls::ip_fft::natural_order;
	static const unsigned config_width = FFT_CONFIG_WIDTH;
	static const unsigned input_width = FFT_INPUT_WIDTH;
	static const unsigned output_width = FFT_OUTPUT_WIDTH;
	static const unsigned max_nfft = FFT_NFFT_MAX;
	static const unsigned scaling_opt = hls::ip_fft::scaled;
	static const unsigned stages_block_ram = 0;
};

typedef hls::ip_fft::config_t<config1> config_t;
typedef hls::ip_fft::status_t<config1> status_t;
typedef ap_fixed<FFT_INPUT_WIDTH, 1> data_in_t;
typedef ap_fixed<FFT_OUTPUT_WIDTH, FFT_OUTPUT_WIDTH - FFT_INPUT_WIDTH + 1> data_out_t;
typedef std::complex<data_in_t> cmpxDataIn;
typedef std::complex<data_out_t> cmpxDataOut;

inline void myfftwrapper(hls::stream<cmpxDataIn> &xn, hls::stream<cmpxDataOut> &xk) {
#pragma HLS dataflow
#pragma HLS INLINE recursive
	config_t config;
	config.setDir(FWD_INV); //For each line we push a config var in a config stream
	config.setSch(0x2AB); // 0x2AB
	hls::stream<config_t> config_s;
	config_s.write(config);
	status_t status;
	hls::stream<status_t> status_s;
	hls::fft<config1>(xn, xk, status_s, config_s);
	status_s.read();
}

#endif //FFT_HPP
