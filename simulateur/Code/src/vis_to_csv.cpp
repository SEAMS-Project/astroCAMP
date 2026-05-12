#include "vis_to_csv.h"

#include "map.h"

/*
void handle_file_error(FILE *file, const char *message) {
	if (file) {
		fclose(file); // Close the file if open
	}
	perror(message);
	exit(EXIT_FAILURE);
}*/

/*
void convert_vis_to_csv(int NUM_VISIBILITIES,float2* output_visibilities, float3* vis_uvw_coords,Config *config) {


	//if (remove(config->output_degridder) == 0) {
	//	printf("Old file deleted: %s\n", config->output_degridder);
	//}

	printf("Trying to open: %s\n", config->output_degridder);
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

	printf("Visibilities successfully saved\n");
}
*/

