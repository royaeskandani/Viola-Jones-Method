// Roya Elisa Eskandani
// 30. March 2023


//! FREE!!!!

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "../Headers/haarfeatures.h"


#define DECREASE_FAC 0.05


struct confusion_matrix {
  int tp;
  int fn;
  int fp;
  int tn;
} confusion_matrix;

const char* create_layer_dir(const char* path_source, int i);
void calulcate_dataset_layer(const char* path_source, int i);
void calculate_strong_classifier(const char* path_source, int n_stumps);
float get_layer_threshold();
struct confusion_matrix evaluate_dataset();
float decrease_threshold_layer(float fac);
void empty_nonfaces(const char* path_source);
void classify_nonfaces(char* path_source, const char* path_destination);


int main(int argc, char *argv[]) {
fprintf(stderr, "\ncalculate cascade: start\n");
  if (argc != 4) {fprintf(stderr, "argc != 4\n"); return 1;}


// 0. Input parameters
  float f_max = atof(argv[1]); // maximal false-positive-rate
  float d_min = atof(argv[2]); // minimal true-positive-rate
  float F_target = atof(argv[3]);


// 1. Parameters
  int i = 0;
  float F = 1.0; // false-positive-rate
  float D = 1.0; // true-positive-rate
  // char path_source[256] = "../Dataset/Train";
  char path_source[256] = "../Dataset/Test";
  system("touch haarcascade.txt");


// 2. Calculate Cascade Classifier
  while (F > F_target) {
// 2.1. Parameters
    i++;
    int n = 0;
    float F_prev = F;
    float threshold_layer = 0.0;
    float D_prev = D;

// 2.2. Calculate Dataset (Layer)
fprintf(stderr, "calculating dataset ...");
    const char* path_destination = create_layer_dir(path_source, i);
   calulcate_dataset_layer(path_destination, i);
fprintf(stderr, " done\n");
fprintf(stderr, "F > F_target: %f > %f\n", F, F_target);
// 2.3. Calculate Strong Classifier
    while (F > (f_max * F_prev)) {
      F = F_prev;
      D = D_prev;
      n++;

// 2.3.1. AdaBoost for n Haar-like Features
fprintf(stderr, "\ncalculating strong classifier ...");
      calculate_strong_classifier(path_destination, n);
      threshold_layer = get_layer_threshold();
fprintf(stderr, " done\n");
fprintf(stderr, "%d: ", n);
fprintf(stderr, "F > (f_max * F_old): %f > %f\n", F, f_max * F_prev);

// 2.3.2. Calculate confusion matrix (Evaluation on validation set)
      struct confusion_matrix cm = evaluate_dataset();

      float tpr = (float) cm.tp / (cm.tp + cm.fn);
fprintf(stderr, "confusion matrix: %d %d %d %d\n", cm.tp, cm.fn, cm.fp, cm.tn);
// fprintf(stderr, "tpr: %f\n", tpr);
// fprintf(stderr, "fpr: %f\n", fpr);
fprintf(stderr, "checking (d_min * D) <= (tpr * D_prev): %f <= %f\n", d_min * D, tpr * D_prev);
      while ((d_min * D_prev) > (tpr * D_prev)) {
fprintf(stderr, "\tthreshold decrease: ");
        threshold_layer = decrease_threshold_layer(DECREASE_FAC);
        if (threshold_layer <= 1.0) break;
        cm = evaluate_dataset();
        tpr = (float) cm.tp / (cm.tp + cm.fn);
      }
fprintf(stderr, "-> (d_min * D) <= (tpr * D_prev): %f <= %f\n", d_min * D, tpr * D_prev);

      float fpr = (float) cm.fp / (cm.fp + cm.tn);
      F *= fpr;
      D *= tpr;

fprintf(stderr, "   F: %f\n", F);
    }
    system("cat temp_layer.txt >> haarcascade.txt");

    empty_nonfaces(path_destination);
    // Setze N = NULLMENGE


    if (F > F_target)
      classify_nonfaces(path_source, path_destination);
    // if (F > F_target)
      // aktuellen Haarcades zum Filtern der true NEGATIVEN Trainingsbeispiele.. -> Neue Menge N

fprintf(stderr, "path_source: %s\n", path_source);
    strcpy(path_source, path_destination);
fprintf(stderr, "path_source: %s\n", path_source);

  }

fprintf(stderr, "end.\n");
  return 0;
}












const char* create_layer_dir(const char* path_source, int i) {
  static char path_destination[256];
  sprintf(path_destination, "Cascade/%02d_Layer/Dataset", i);

  char cmd_mkdir[256];
  sprintf(cmd_mkdir, "mkdir -p %s", path_destination);
  system(cmd_mkdir);

  char cmd_cp_bmps[256];
  sprintf(cmd_cp_bmps, "cp %s/*.bmp %s", path_source, path_destination);
  system(cmd_cp_bmps);

  return path_destination;
}


void calulcate_dataset_layer(const char* path_source, int i) {
  char cmd_calculate_dataset[256];
  sprintf(cmd_calculate_dataset, "./calculate_dataset %s > %s/dataset.txt", path_source, path_source);
  system(cmd_calculate_dataset);
}


void calculate_strong_classifier(const char* path_source, int n_stumps) {
  // 1. Calculate strong classifier
  char cmd_adaboost[256];
  sprintf(cmd_adaboost,"./calculate_strong_classifier %s %d > temp_layer.txt", path_source, n_stumps);
  system(cmd_adaboost);

  // 2. Create temporary files for layer
  system("cat haarcascade.txt > temp_haarcascade.txt");
  system("cat temp_layer.txt >> temp_haarcascade.txt");
}


float get_layer_threshold() {
  //1. Read strong classifier
  FILE* strong_classifier_file = fopen("temp_layer.txt", "r");
  float threshold_layer = 0.0;
  if (strong_classifier_file != NULL) {
    fscanf(strong_classifier_file, "%*s %f", &threshold_layer);
    fclose(strong_classifier_file);
  }

  return threshold_layer;
}


struct confusion_matrix evaluate_dataset() {
  // 1. Evaluate dataset
  system("./evaluate_dataset > confusion_matrix.txt");

  // 2. Calulcate confusion matrix
  struct confusion_matrix cm = {0, 0, 0, 0};

  FILE* file = fopen("confusion_matrix.txt", "rb");
  fscanf(file, "%d %d %d %d", &cm.tp, &cm.fn, &cm.fp, &cm.tn);
  fclose(file);

  return cm;
}


float decrease_threshold_layer(float fac) {
  // 1. Read strong classifier
  struct haarcascade** temp_cascade = NULL;
  int n_stumps = 0;

  FILE* file = fopen("temp_layer.txt", "r");
  if (!file) {
    fprintf(stderr, "Error: calculate_cascade.c -> decrease_threshold_layer(), * file - read\n");
    return 0.0;
  }

  while (!feof(file)) {
    struct haarcascade* cascade = read_haarcascade(file);
    if (cascade) {
      n_stumps++;
      temp_cascade = (struct haarcascade**) realloc(temp_cascade, sizeof(struct haarcascade*) * n_stumps);
      temp_cascade[n_stumps-1] = cascade;
    }
  }
  fclose(file);


  // 2. Update strong classifer threshold
fprintf(stderr, " %f", temp_cascade[0]->threshold_forest);
  for (int i = 0; i < n_stumps; i++)
    // temp_cascade[i]->threshold_forest *= (1 - fac);
    temp_cascade[i]->threshold_forest -= fac;
fprintf(stderr, "-> %f\n", temp_cascade[0]->threshold_forest);
  file = fopen("temp_layer.txt", "w");
  if (!file) {
    fprintf(stderr, "Error: calculate_cascade.c -> decrease_threshold_layer(), * file - write\n");
    return 0.0;
  }

  for (int i = 0; i < n_stumps; i++)
    fprinft_cascade(file, temp_cascade[i]);
  fclose(file);

  system("cat haarcascade.txt > temp_haarcascade.txt");
  system("cat temp_layer.txt >> temp_haarcascade.txt");


  // 3. Free
  for (int i = 0; i < n_stumps; i++) {
    free(temp_cascade[i]->haarfeature);
    free(temp_cascade[i]);
  }
  free(temp_cascade);

  return get_layer_threshold();
}


void empty_nonfaces(const char* path_source) {
  char cmd_rm_nonfaces[256];
  sprintf(cmd_rm_nonfaces, "rm %s/nonface_*.bmp", path_source);
  system(cmd_rm_nonfaces);
}


void classify_nonfaces(char* path_source, const char* path_destination) {
  DIR* dir;
  struct dirent* entry;

  if ((dir = opendir(path_source)) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
      if (strstr(entry->d_name, "nonface_") == entry->d_name && strstr(entry->d_name, ".bmp")) {

        char filename[256];
        sprintf(filename, "%s/%s", path_source, entry->d_name);

// 2.1. Haar-like Feature Values
        FILE* bmp = fopen(filename, "rb");
        if (!bmp) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * bmp\n");
          return;
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
          return;
        }

        image* integral_img = calculate_integral_image(img);
        if (!integral_img) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * integral_img\n");
          free_image(img);
          return;
        }


// 3. Classification with Haar Cascade Classifier
        FILE* haarcascade = fopen("temp_haarcascade.txt", "r");
        if (!haarcascade) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * haarcascade\n");
          free(integral_img);
          return;
        }

        struct haarcascade* cascade = read_haarcascade(haarcascade);
        if (!cascade) {
          fprintf(stderr, "Error: evaluate_dataset.c -> main(), * cascade\n");
          fclose(haarcascade);
          free_image(integral_img);
          return;
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
fprintf(stderr, "pred(nonface): %s\n", filename);
            break;
          }

          cascade = read_haarcascade(haarcascade);
          if (!cascade)
            break;
        } while(cascade->n_stumps != 0);

        if (flag_predicted == 0) { // false prediction
          char cmd_cp_bmp[256];
fprintf(stderr, "pred(nonface): %s\n", filename);
          sprintf(cmd_cp_bmp, "cp %s %s", filename, path_destination);
          system(cmd_cp_bmp);
        }

        fclose(haarcascade);
        free_haarcascade(cascade);
        free_image(integral_img);
        free_image(img);
        fclose(bmp);
      }
    }
  }
}