//
// Created by orenaud on 4/14/25.
//
#include "casacore_wrapper.h"
#include "common.h"

#include <iostream>
#include <cmath>
#include <limits>

#define SPEED_OF_LIGHT 299792458.0

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif


/**
 * Actually loads data from a csv, but name us kept for compatibility reasons
 * @param ms_path_c
 * @param num_vis
 * @param config
 * @param uvw_coords
 */

extern "C" void load_visibilities_from_ms(const char* ms_path_c, int num_vis,
                                          Config* config, float3* uvw_coords)
{
    std::string ms_path(ms_path_c);
    FILE* uvw_csv = fopen("uvw_64_vec.csv", "r");
    FILE* info_file = fopen("info.csv", "r");

    if(uvw_csv == NULL) {
        std::cerr << "ERROR >>> csv file " << "uvw_64_vec.csv" << " does not exist!\n";
        std::exit(EXIT_FAILURE);
    }
    if(info_file == NULL) {
        std::cerr << "ERROR >>> csv file " << "info.csv" << " does not exist!\n";
        std::exit(EXIT_FAILURE);
    }

    // extract information
    int total_rows;
    double D;
    double freq_hz;
    fscanf(info_file, "%d, %lf, %lf", &total_rows, &D, &freq_hz);

    //casacore::MeasurementSet ms(ms_path);
    //casacore::MSColumns msCols(ms);

    //int total_rows = ms.nrow();
    if (num_vis > total_rows) {
        std::cerr << "WARNING >>> Requested " << num_vis << " visibilities, but only " << total_rows << " available. Truncating to " << total_rows << ".\n";
        num_vis = total_rows;
    }

    double maxW = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < num_vis; ++i)
    {
        double u;
        double v;
        double w;

        fscanf(uvw_csv, "%lf,%lf,%lf", &u, &v, &w);

        if (w > maxW) {
            maxW = w;
        }

        if (config->right_ascension) {
            u *= -1.0;
            w *= -1.0;
        }

        uvw_coords[i].x = u ;
        uvw_coords[i].y = v ;
        uvw_coords[i].z = w ;
    }

    config->max_w = maxW;
    int NUM_KERNELS = 17;

    config->w_scale = pow(NUM_KERNELS - 1, 2.0) / config->max_w;

    // Get central frequency (first value of the array)
    config->frequency_hz = freq_hz;

    // ANTENNA subtable

    // Compute wavelength
    double wavelength = SPEED_OF_LIGHT / freq_hz;

    // Compute field of view (in radians then degrees)
    double fov_rad = 1.22 * wavelength / D;
    double fov_deg = fov_rad * 180.0 / M_PI;

    int GRID_SIZE = 1024;
    config->cell_size = (fov_deg * PI) / (180.0 * GRID_SIZE);
    config->uv_scale =  config->cell_size*GRID_SIZE;
}


/**
 * Former ms data loading function. Kept for history.
 * @param ms_path_c
 * @param num_vis
 * @param config
 * @param uvw_coords
 */
/*
extern "C" void load_visibilities_from_ms_save(const char* ms_path_c, int num_vis,
                                               Config* config, float3* uvw_coords)
{
    std::string ms_path(ms_path_c);

    if (!casacore::File(ms_path).exists()) {
        std::cerr << "ERROR >>> MeasurementSet " << ms_path << " does not exist!\n";
        std::exit(EXIT_FAILURE);
    }

    casacore::MeasurementSet ms(ms_path);
    casacore::MSColumns msCols(ms);

    int total_rows = ms.nrow();
    if (num_vis > total_rows) {
        std::cerr << "WARNING >>> Requested " << num_vis << " visibilities, but only " << total_rows << " available. Truncating to " << total_rows << ".\n";
        num_vis = total_rows;
    }

    casacore::ArrayColumn<casacore::Double> uvwCol(ms, "UVW");
    size_t nrows = ms.nrow();


    auto uvw_col = msCols.uvw().getColumn();
    auto data_col = msCols.data().getColumn();
    auto weight_col = msCols.weight().getColumn();

    auto uvw_vec = uvw_col.tovector();
    auto data_vec = data_col.tovector();
    auto weight_vec = weight_col.tovector();

    auto shape = data_col.shape();


    int nchan = shape[0];
    int npol = shape[1];


    double maxW = -std::numeric_limits<double>::infinity();


    for (int i = 0; i < num_vis; ++i)
    {
        casacore::Vector<casacore::Double> uvw;
        uvwCol.get(i, uvw);
        double u = uvw[0];
        double v = uvw[1];
        double w = uvw[2];

        if (w > maxW) {
            maxW = w;
        }

        if (config->right_ascension) {
            u *= -1.0;
            w *= -1.0;
        }

        uvw_coords[i].x = u ;
        uvw_coords[i].y = v ;
        uvw_coords[i].z = w ;

        auto cpx = data_vec[i * nchan * npol];  // premier canal/pola
        float weight = config->force_weight_to_one ? 1.0f : weight_vec[i];

    }

    config->max_w = maxW;
    int NUM_KERNELS = 17;

    config->w_scale = pow(NUM_KERNELS - 1, 2.0) / config->max_w;


    casacore::MSSpectralWindow spwTable = ms.spectralWindow();
    casacore::ArrayColumn<casacore::Double> chanFreqCol(spwTable, "CHAN_FREQ");

    // Get central frequency (first value of the array)
    casacore::Array<casacore::Double> freqs;
    chanFreqCol.get(0, freqs);
    double freq_hz = freqs(casacore::IPosition(1, 0));
    config->frequency_hz = freq_hz;

    // ANTENNA subtable
    casacore::MSAntenna antTable = ms.antenna();
    casacore::ScalarColumn<casacore::Double> dishDiameterCol(antTable, "DISH_DIAMETER");

    // Get the diameter of the first antenna
    double D = dishDiameterCol(0); // in meters

    // Compute wavelength
    double wavelength = SPEED_OF_LIGHT / freq_hz;

    // Compute field of view (in radians then degrees)
    double fov_rad = 1.22 * wavelength / D;
    double fov_deg = fov_rad * 180.0 / M_PI;
    //std::cout << "📡 FoV ≈ " << fov_deg << " degrees" << std::endl;
    int GRID_SIZE = 1024;
    config->cell_size = (fov_deg * PI) / (180.0 * GRID_SIZE);
    config->uv_scale =  config->cell_size*GRID_SIZE;

    std::cout << "spectralWindow : " << spwTable << std::endl;
    std::cout << "freq_hz : " << freq_hz << std::endl;

}
*/