#include "input_grid.h"

#include <stdio.h>
#include <stdlib.h>

void input_grid_setup(int GRID_SIZE, IN Config *config, OUT float2 *output) {
    const char *file_name = config->input;
    FILE *f = fopen(file_name, "r");
    if (f == NULL) {
        perror(">>> ERROR");
        printf(">>> ERROR: Unable to open file %s...\n\n", file_name);
        return;
    }

    // Calcul du nombre d'éléments dans le fichier
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);

    // On suppose qu'il y a un nombre pair de valeurs dans le fichier (x, y)
    long num_elements = file_size / sizeof(float);  // Nombre total d'éléments (x et y compris)
    if (num_elements % 2 != 0) {
//        printf("Nombre impair de valeurs détecté, suppression du dernier élément.\n");
        num_elements--;
    }

    // Lecture des valeurs dans le fichier
    int i = 0;
    while (fscanf(f, "%f, %f,", &output[i].x, &output[i].y) == 2) {
        i++;
    }

    fclose(f);

//    // Affichage des valeurs non nulles
//    printf("Affichage des valeurs non nulles :\n");
//    for (int j = 0; j < i; j++) {
//        if (output[j].x != 0.0f || output[j].y != 0.0f) {
//            printf("output[%d] = (%f, %f)\n", j, output[j].x, output[j].y);
//        }
//    }
    printf("input_grid_setup\n");

}
