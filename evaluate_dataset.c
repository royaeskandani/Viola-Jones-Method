// Roya Elisa Eskandani
// 16. March 2023

/*
  Classification of BMP files into one of the "nonface" or "face" categories
  
  Methode: Cascade Classifier with Haar-like Features
*/


#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "../Headers/haarfeatures.h"
#include "../Headers/image.h"


struct confusion_matrix {
  int tp;
  int fn;
  int fp;
  int tn;
} confusion_matrix;


int main() {
// 1. Parameter
  struct confusion_matrix cm = {0, 0, 0, 0}; // tp fn fp tn
  char* path_validation = "../Dataset/Validation";


// 2. Caclulate Confusion Matrix
  DIR* dir;
  struct dirent* entry;

  if ((dir = opendir(path_validation)) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
      if (strstr(entry->d_name, ".bmp")) {
        char filename[256];
        sprintf(filename, "%s/%s", path_validation, entry->d_name);

// 2.1. Haar-like Feature Values
        FILE* bmp = fopen(filename, "rb");
        if (!bmp) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * bmp\n");
          return 1;
        }

// 2.2. Label
        char* prefix = "nonface";
        int label = 1;
        if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0)
          label = 0;

// 2.3. Calculate Integral image
        image* img = calculate_RGB_to_grayscale(bmp);
        if (!img) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * img\n");
          fclose(bmp);
          return 1;
        }

        image* integral_img = calculate_integral_image(img);
        if (!integral_img) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * integral_img\n");
          free_image(img);
          return 1;
        }


// 3. Classification with Haar Cascade Classifier
        FILE* haarcascade = fopen("temp_haarcascade.txt", "r");
        if (!haarcascade) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * haarcascade\n");
          free(integral_img);
          return 1;
        }

        struct haarcascade* cascade = read_haarcascade(haarcascade);
        if (!cascade) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * cascade\n");
          fclose(haarcascade);
          free_image(integral_img);
          return 1;
        }

        int flag_predicted = 0;
        do {
          float value_cascade = 0.;
      
          for (int i_stump = 0; i_stump < cascade->n_stumps; i_stump++) {
            int haarvalue = calculate_haarvalue(cascade->haarfeature, integral_img);

            if (haarvalue < cascade->threshold_stump)
              value_cascade += cascade->value_left_child;
            else
              value_cascade += cascade->value_right_child;

            if (i_stump != cascade->n_stumps - 1) {
              cascade = read_haarcascade(haarcascade);
              if (!cascade)
                break;
            }
          }

          if (value_cascade < cascade->threshold_forest) { // predict nonface
            flag_predicted = 1;
            if (label == 0)
              cm.tn++;
            else
              cm.fn++;
            break;
          }

          cascade = read_haarcascade(haarcascade);
          if (!cascade)
            break;
        } while(cascade->n_stumps != 0);

        if (flag_predicted == 0) { // not predicted -> predict face
          if (label == 0)
            cm.fp++;
          else
            cm.tp++;
        }

        fclose(haarcascade);
        free_haarcascade(cascade);
        free_image(integral_img);
        free_image(img);
        fclose(bmp);
      }
    }
  }


printf("%d %d %d %d\n", cm.tp, cm.fn, cm.fp, cm.tn);
  return 0;
}