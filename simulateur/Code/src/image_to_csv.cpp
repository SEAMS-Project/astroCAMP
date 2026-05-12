#include "image_to_csv.h"

#include "map.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



//void save_heatmap_png(const float* image, int size, const char* filename) {
	/*
	unsigned char* pixels = malloc(size * size * sizeof(unsigned char));
	if (!pixels) {
		fprintf(stderr, "Erreur allocation mémoire heatmap\n");
		return;
	}

	// Find min/max to normalise
	float min_val = FLT_MAX, max_val = -FLT_MAX;
	for (int i = 0; i < size * size; ++i) {
		if (image[i] < min_val) min_val = image[i];
		if (image[i] > max_val) max_val = image[i];
	}

	float range = max_val - min_val;
	if (range == 0.0f) range = 1.0f; //avoid dividing by 0

	// Normalisation + conversion in gris shades (0-255)
	for (int i = 0; i < size * size; ++i) {
		float norm = (image[i] - min_val) / range;
		pixels[i] = (unsigned char)(norm * 255.0f);
	}

	// Save PNG
	int success = stbi_write_png(filename, size, size, 1, pixels, size);

	free(pixels);
	*/
//	return;
//}
/*
void export_image_to_csv(IN float *image_grid, IN Config * config, int GRID_SIZE){

	if (remove(config->output_ifft) == 0) {
		printf("Old file deleted: %s\n", config->output_ifft);
	}
    FILE *f = fopen(config->output_ifft, "w");
    if (f == NULL) {
        perror(">>> ERROR output_ifft");
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
    printf("export_image_to_csv\n");

}

void export_image_to_csv_input_grid(float2 *image_grid, int GRID_SIZE) {

	FILE *f = fopen("../output/input_grid.csv", "w");
	if (f == NULL) {
		perror(">>> ERROR");
		return;
	}

	//Write the complex data in the csv file
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++) {
		fprintf(f, "%.6f, %.6f\n", image_grid[i].x, image_grid[i].y); // Partie réelle, partie imaginaire
	}
	fclose(f);
    printf("export_image_to_csv_input_grid\n");

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
	printf("export_image_to_csv_uv_grid\n");

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
	printf("export_image_to_csv_gridding\n");


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
	printf("export_image_to_csv_ifft_test\n");

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
	save_heatmap_png(image_grid, 64, "../output/heatmap_reconstructed.png");
	fclose(f);
	printf("export_image_to_csv_reconstructed\n");



}
*/
