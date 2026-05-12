#define _POSIX_C_SOURCE 199309L
#include "top.h"
#include <math.h>
#include "casacore_wrapper.h"
#include "map.h"
#include <time.h>
#include <stdio.h>



void config_struct_set_up(int GRID_SIZE, int NUM_KERNELS, int NUM_SCENARIO, OUT Config *config_struct)
{
	int FOV_DEGREES = 1; // field of view

	switch (NUM_SCENARIO) {

		case 1:
			config_struct->input = "../data/input/input.csv";
			config_struct->visibility_source_file = "../data/sim_small.ms";
			break;

	    case 2:
	        config_struct->input = "../data/input/cycle_0_deconvolved_2560.csv";
	        config_struct->visibility_source_file = "../data/sim_medium.ms";
	        break;

	    case 3:
	        config_struct->input = "../data/input/cycle_0_deconvolved_5120.csv";
	        config_struct->visibility_source_file = "../data/sim_large.ms";
	        break;

	    default:
			config_struct->input = "../data/input/cycle_0_deconvolved_2560.csv";
			config_struct->visibility_source_file = "../data/sim_small.ms";
	        break;

	}

	config_struct->frequency_hz = SPEED_OF_LIGHT/0.21;// SPEED_OF_LIGHT

	config_struct->degridding_kernel_support_file = "../data/input/kernels/small/w-proj_supports_x16_2458_image.csv";

	int baseline_max = 800;
	config_struct->max_w = baseline_max*config_struct->frequency_hz/SPEED_OF_LIGHT;

	config_struct->w_scale = pow(NUM_KERNELS - 1, 2.0) / config_struct->max_w;

	config_struct->cell_size = (FOV_DEGREES * PI) / (180.0 * GRID_SIZE);//lower than 1/2.f_max//1.84945e-05;

	config_struct->uv_scale = config_struct->cell_size * GRID_SIZE;

	config_struct->degridding_kernel_imag_file = "../data/input/kernels/small/w-proj_kernels_imag_x16_2458_image.csv";
	config_struct->degridding_kernel_real_file = "../data/input/kernels/small/w-proj_kernels_real_x16_2458_image.csv";


	config_struct->right_ascension = true;
	config_struct->force_weight_to_one	= true;
	config_struct->output_degridder= "../output/output_degridder.csv";
	config_struct->output_ifft="../output/reconstructed_shift.csv";
	config_struct->output_fft_test="../output/output_image_fft_test.csv";

}

void load_image_from_file(int GRID_SIZE, Config* config, float *output) {
	const char *file_name = config->input;
	FILE *f = fopen(file_name, "r");
	if (f == NULL) {
		perror(">>> ERROR");
		printf(">>> ERROR: Unable to open file %s...\n\n", file_name);
		return;
	}
	else {
		printf("opened file %s successfully \n", file_name);
	}

	int i = 0;
	 while (fscanf(f, "%f%*[, \n]", &output[i]) == 1) {
		i++;
		if (i >= GRID_SIZE*GRID_SIZE+1) { //TODO : truncating for 1 element, to be checked. if +1 no problem
			printf(">>> WARNING: Too many elements in file, truncating\n");
			break;
		}
	}
	fclose(f);
	//printf("load_image_from_file\n");
}

float pswf(float x) {
	if (fabs(x) > 1.0) return 0.0;
	return powf(cosf(PI * x / 2.0f), 4);
}
float I0(float x) {
	float sum = 1.0f;
	float term = 1.0f;
	for (int n = 1; n < 20; ++n) {
		term *= (x / (2.0f * n));
		sum += term * term;
	}
	return sum;
}

float kaiser_bessel(float x, float beta) {
	if (fabs(x) > 1.0f) return 0.0f;
	float t = sqrtf(1.0f - x * x);
	return powf(I0(beta * t) / I0(beta), 2.0f);
}

void generate_kernel(int NUM_KERNELS, int OVERSAMPLING_FACTOR, int NUM_KERNEL_SUPPORT, int TOTAL_KERNELS_SAMPLES,
    float2* kernels, int2* kernel_supports) {
    bool bessel_boolean = false;
    int kernel_half = NUM_KERNEL_SUPPORT/2;
    //int kernel_size = 2 * kernel_half;

    int kernel_1d_len = (kernel_half + 1) * OVERSAMPLING_FACTOR;
    int kernel_2d_len = kernel_1d_len * kernel_1d_len;

    int kernel_offset = 0;

    for (int w = 0; w < NUM_KERNELS; ++w)
    {
       kernel_supports[w].x = kernel_half;
       kernel_supports[w].y = kernel_offset;

       // Generate 1D kernel
       float* kernel_1d = (float*) malloc(sizeof(float) * kernel_1d_len);
       for (int i = 0; i < kernel_1d_len; ++i) {
          float x = (float)i / (float)OVERSAMPLING_FACTOR;
          float norm_x = x / (float)(kernel_half + 1);
          if (bessel_boolean) {
             float beta  =3.0f; // bessel parameter to control lobe shape: large β -> thinner lobe (better concentration but increase secondary lobes), tradeoff resolution and noise
             kernel_1d[i] = kaiser_bessel(norm_x, beta);
          }else {
             kernel_1d[i] = pswf(norm_x);
          }
       }

       // NOrmalize the 1D kernel
       float sum = 0.f;
       for (int i = 0; i < kernel_1d_len; ++i) sum += kernel_1d[i];
       for (int i = 0; i < kernel_1d_len; ++i) kernel_1d[i] /= sum;

       for (int y = 0; y < kernel_1d_len; ++y) {
          for (int x = 0; x < kernel_1d_len; ++x) {
             int idx = kernel_offset + y * kernel_1d_len + x;
             float val = kernel_1d[x] * kernel_1d[y];
             kernels[idx].x = val;
             kernels[idx].y = 0.0f;
          }
       }

       kernel_offset += kernel_2d_len;
       free(kernel_1d);
    }
    //printf("generate_kernel\n");
}

void save_degridding_kernels_to_csv(float2* degridding_kernels, int TOTAL_KERNELS_SAMPLES) {
    FILE* file = fopen("kernels_prod.csv", "w");
    if (file == NULL) {
        printf("ERROR: Unable to open file %s for writing kernels.\n", "kernels_prod.csv");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "real,imag\n");
    for (int i = 0; i < TOTAL_KERNELS_SAMPLES; ++i) {
        fprintf(file, "%.10f,%.10f\n", degridding_kernels[i].x, degridding_kernels[i].y);
    }

    fclose(file);
}

void save_degridding_kernel_supports_to_csv(int2* degridding_kernel_supports, int NUM_KERNELS) {
    FILE* file = fopen("kernels_support_prod.csv", "w");
    if (file == NULL) {
        printf("ERROR: Unable to open file %s for writing supports.\n", "kernels_support_prod.csv");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "support,offset\n");
    for (int i = 0; i < NUM_KERNELS; ++i) {
        fprintf(file, "%d,%d\n", degridding_kernel_supports[i].x, degridding_kernel_supports[i].y);
    }

    fclose(file);
}


double randn(double mean, double stddev) {
	double u1 = ((double) rand() / RAND_MAX);  // Generate a number between 0 and 1
	double u2 = ((double) rand() / RAND_MAX);  // Generate another number between 0 and 1
	double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);  // Box-Muller transform
	return z0 * stddev + mean;  // Transform to have a given mean and standard deviation
}


void visibility_host_set_up2(int NUM_VISIBILITIES, Config* config,float3* vis_uvw_coords){
    //Case if the visibilities are saved in a csv file
    if (strstr(config->visibility_source_file, ".csv") != NULL) {
		//printf("UPDATE >>> Loading visibilities from file %s...\n\n", config->visibility_source_file);
		FILE *vis_file = fopen(config->visibility_source_file, "r");
		if(vis_file == NULL)
		{   printf("ERROR >>> Unable to open visibility file %s...\n\n", config->visibility_source_file);
			exit(EXIT_FAILURE);
			// return false; // unsuccessfully loaded data
		}

		// Configure number of visibilities from file
		int num_vis = 0;
		fscanf(vis_file, "%d", &num_vis);
		if(vis_uvw_coords == NULL)
		{
			printf("ERROR >> Unable to allocate memory for visibility information...\n\n");
			fclose(vis_file);
			exit(EXIT_FAILURE);
			// return false;
		}

		// Load visibility uvw coordinates into memory
		double vis_u = 0.0;
		double vis_v = 0.0;
		double vis_w = 0.0;
		double vis_real = 0.0;
		double vis_imag = 0.0;
		double vis_weight = 0.0;
		double meters_to_wavelengths = config->frequency_hz / SPEED_OF_LIGHT; // lambda = c/f

		for(int vis_index = 0; vis_index < NUM_VISIBILITIES; ++vis_index)
		{
			fscanf(vis_file, "%lf %lf %lf %lf %lf %lf\n", &vis_u, &vis_v,&vis_w, &vis_real, &vis_imag, &vis_weight);

			if(config->right_ascension)
			{
				vis_u *= -1.0;
				vis_w *= -1.0;
			}

			if(!config->force_weight_to_one)
			{
				vis_real *= vis_weight;
				vis_imag *= vis_weight;
			}

			vis_uvw_coords[vis_index].x = (vis_u * meters_to_wavelengths);
			vis_uvw_coords[vis_index].y = (vis_v * meters_to_wavelengths);
			vis_uvw_coords[vis_index].z = (vis_w * meters_to_wavelengths);

		}
		//printf("UPDATE >>> Successfully loaded %d visibilities from file...\n\n", NUM_VISIBILITIES);
		// Clean up
		fclose(vis_file);

    }else if(strstr(config->visibility_source_file, ".ms") != NULL) {

		load_visibilities_from_ms(config->visibility_source_file, NUM_VISIBILITIES, config, vis_uvw_coords);
	    //save_visibilities_to_csv("exported_visibilities.csv", vis_uvw_coords, NUM_VISIBILITIES);

    }else{
        printf("Error reading the visibility file: %s\n", config->visibility_source_file);
        exit(EXIT_FAILURE);
    }
	//printf("visibility_host_set_up\n");

}

void save_visibilities_to_csv(const char* filename, float3* vis_uvw_coords, int num_visibilities) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file for writing");
        return;
    }

    for (int i = 0; i < num_visibilities; ++i) {
        fprintf(file, "%f %f %f 1\n",
                vis_uvw_coords[i].x,
                vis_uvw_coords[i].y,
                vis_uvw_coords[i].z);
    }

    fclose(file);
}


void handle_file_error(FILE *file, const char *message) {
	if (file) {
		fclose(file); // Close the file if open
	}
	perror(message);
	exit(EXIT_FAILURE);
}

void convert_vis_to_csv(int NUM_VISIBILITIES,float2* output_visibilities, float3* vis_uvw_coords,Config *config) {

	FILE* file = fopen(config->output_degridder, "w");
	if (file == NULL) {
		handle_file_error(file, "Error opening file convert_vis_to_csv");
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
	//printf("Visibilities successfully saved\n");
}




void export_image_to_csv(IN float *image_grid, IN Config * config, int GRID_SIZE){

	if (remove(config->output_ifft) == 0) {
		printf("Old file deleted: %s\n", config->output_ifft);
	}
    FILE *f = fopen(config->output_ifft, "w");
    if (f == NULL) {
        perror(">>> ERROR");
        return;
    }
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            fprintf(f, "%.6f", image_grid[i * GRID_SIZE + j]);
            if (j < GRID_SIZE - 1)
                fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    //printf("export_image_to_csv\n");
}

void export_image_to_csv_input_grid(float2 *image_grid, int GRID_SIZE) {

	FILE *f = fopen("../output/input_grid.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y);
	}
	fclose(f);
    //printf("export_image_to_csv_input_grid\n");
}

void export_image_to_csv_uv_grid(float2 *image_grid, int GRID_SIZE) {
	FILE *f = fopen("../output/uv_grid.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}
	fclose(f);
	//printf("export_image_to_csv_uv_grid\n");
}

void export_image_to_csv_uv_grid_shift(float2 *image_grid, int GRID_SIZE) {

	FILE *f = fopen("../output/uv_grid_shift.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}
	fclose(f);
	//printf("export_image_to_csv_uv_grid_shift\n");

}

void export_image_to_csv_gridding(float2 *image_grid, int GRID_SIZE) {

	FILE *f = fopen("../output/gridding.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}
	fclose(f);
	//printf("export_image_to_csv_gridding\n");
}


void export_image_to_csv_ifft_test(float2 *image_grid, int GRID_SIZE) {

	FILE *f = fopen("../output/ifft_test.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}
	fclose(f);
	//printf("export_image_to_csv_ifft_test\n");
}

void export_image_to_csv_reconstructed(float2 *image_grid, int GRID_SIZE) {

	FILE *f = fopen("../output/reconstructed.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}
	fclose(f);
	//printf("export_image_to_csv_reconstructed\n");
}


