// Roya Elisa Eskandani
// 30. March 2023

/*
  Calculation of Haar-like feature values and labels of the dataset.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "../Headers/image.h"
#include "../Headers/haarfeatures.h"

#define N_FEATURES 162335


int main(int argc, char *argv[]) {
  if (argc != 2) {fprintf(stderr, "argc != 2\n"); return 1;}

// 0. Parameter  
  char *path_dataset = argv[1];

  
// 1. Create Dataset (Haar-like Features and Label)
  DIR* dir;
  struct dirent* entry;

  if ((dir = opendir(path_dataset)) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
      if (strstr(entry->d_name, ".bmp")) {
        char filename[256];
        sprintf(filename, "%s/%s", path_dataset, entry->d_name);

// 2. Haar-like Feature Values
        FILE* bmp = fopen(filename, "rb");
        if (!bmp) {fprintf(stderr, "%s konnte nicht geöffnet werden.\n", filename); return 1;}

// 2.1. Label
        char* prefix = "nonface";
        int label = 1;
        if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0)
          label = 0;

// 2.2. Calculate Integral image
        struct image* img = calculate_RGB_to_grayscale(bmp);
        fclose(bmp);

        struct image* integral_img = calculate_integral_image(img);
  
// 2.3. Load Haar-like Features
        FILE* haarfeatures = fopen("../Dataset/haarfeatures.txt", "r");
        if (!haarfeatures) {printf("haarfeatures.txt konnte nicht geöffnet werden.\n"); return 1;}


// 3. Print Haar-like Feature Values and Label
        for (int i = 0; i < N_FEATURES; i++) {
          struct pattern* haarfeature = read_haarfeatures(haarfeatures);
          int haarvalue = calculate_haarvalue(haarfeature, integral_img);
          printf("%d ", haarvalue); 
        }
        printf("%d\n", label);


// 4. Free / Close
        free_image(img);
        free_image(integral_img);
        fclose(haarfeatures);
      }
    }
    closedir(dir);
  }

  return 0;
}
