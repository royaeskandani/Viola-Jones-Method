// Roya Elisa Eskandani
// 16. March 2023

// Input: siehe _run_adaboost.py
// Output: alle Stumps einer Cascade - Format wie bei OpenCV


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../Headers/image.h"
#include "../Headers/haarfeatures.h"
#include "../Headers/adaboost.h"


int main(int argc, char *argv[]) {
  if (argc != 3) {fprintf(stderr, "argc != 3\n"); return 1;}

// 0. Input parameters
  char* path_layer = argv[1];
  int n_stumps = atoi(argv[2]);

 
// 1. Create Dataset (Haar-like Feature Values and Image Labels)
  int n_samples_face = 0, n_samples_nonface = 0;
  struct dataset* dataset = create_dataset(path_layer, &n_samples_face, &n_samples_nonface);
  if (!dataset) {
    fprintf(stderr, "Error: adaboost.c -> main(), * dataset\n");
    return 1;
  }
  int n_samples = n_samples_face + n_samples_nonface;
  if (n_samples == 0) {fprintf(stderr, "....");}


// 2. AdaBoost
  struct stump** classifier = init_stumps(dataset->samples, n_stumps);
  if (!classifier) {
    fprintf(stderr, "Error: adaboost.c -> main(), * classifier\n");
    free_dataset(dataset);
    return 1;
  }

// 2.1. Training
// 2.1.1. Initialize weights
  float* weights = init_weights(dataset->samples, n_samples, n_samples_face, n_samples_nonface);
  if (!weights) {
    fprintf(stderr, "Error: adaboost.c -> main(), * weights\n");
    free_stumps(classifier, n_stumps);
    free_dataset(dataset);
    return 1;
  }

// 2.1.2. Select weak classifier
  for (int t = 0; t < n_stumps; t++) {
    //  Normalize weights
    normalize_weights(weights, n_samples);
    if (!weights) {
      fprintf(stderr, "Error: adaboost.c -> main(), * weights\n");
      free_stumps(classifier, n_stumps);
      free_dataset(dataset);
      return 1;
    }

    struct stump** stumps = init_stumps(dataset->samples, N_FEATURES);
    if (!stumps) {
      fprintf(stderr, "Error: adaboost.c -> main(), * stumps\n");
      free(weights);
      free_stumps(classifier, n_stumps);
      free_dataset(dataset);
      return 1;
    }

    // Select the best weak classifier with respect to the weighted error
    struct stump* best_stump = stumps[0];
    if (!best_stump) {
      fprintf(stderr, "Error: adaboost.c -> main(), * stumps\n");
      free_stumps(stumps, N_FEATURES);
      free(weights);
      free_stumps(classifier, n_stumps);
      free_dataset(dataset);
      return 1;
    }

    // T+ = sum of positive example weights
    // T- = sum of negative example weights 
    float weight_faces = 0., weight_nonfaces = 0.;
    for (int i = 0; i < n_samples; i++) {
      if (dataset->samples[i]->label == 1) weight_faces += weights[i];
      else weight_nonfaces += weights[i];
    }
  
    float minimal_error = INFINITY;
    for (int j = 0; j < N_FEATURES; j++) {
//?(siehe nächste Zeile) weak classifier h(x,feature, polarity, threshold) = (p*f(x) < p*threshold) ? 1 : 0 ;
//? Im Folgenden wird die Polarität nicht betrachtet.

    // error for a threshold which splits the range between the current and previous example in the sorted list ist:
    // e = min(S^+ + (T^- - S^-), S^- + (T^+ - S^+))
      int* sorted_idx = dataset->sorted_idx[j];

      int best_threshold = INFINITY;
      float minimal_error_threshold = INFINITY; // Minimum for specific threshold (selected from the Haar values)

      float weights_posivite = 0., weights_negative = 0.;
      int current_val = dataset->samples[sorted_idx[0]]->haar_value[j]; 
      for (int i = 1; i < n_samples - 1; i++) { //TODO Stimmt die Anpassung hier?! threshold != minimal or  maximal value
        if (dataset->samples[sorted_idx[i]]->haar_value[j] != current_val) {
          current_val = dataset->samples[sorted_idx[i]]->haar_value[j];
          int threshold = dataset->samples[sorted_idx[i]]->haar_value[j];

          // S+ = sum of posivite weights below the current example
          // S- = sum of negative weights below the current example
          if (dataset->samples[sorted_idx[i-1]]->label == 1) weights_posivite += weights[sorted_idx[i-1]];
          else weights_negative += weights[sorted_idx[i-1]];

          // e = min(S^+ + (T^- - S^-), S^- + (T^+ - S^+))
          //   = min(T^- + (S^+ - S^-), T^+ + (S^- - S^+))
          float left_value = weights_posivite + weight_nonfaces - weights_negative;
          float right_value = weights_negative + weight_faces - weights_posivite;

          float error = (left_value < right_value) ? left_value : right_value;
          if (error < minimal_error_threshold) {
            minimal_error_threshold = error;
            best_threshold = threshold;
          }        
        } else
          if (dataset->samples[sorted_idx[i-1]]->label == 1) weights_posivite += weights[sorted_idx[i-1]];
          else weights_negative += weights[sorted_idx[i-1]];
          continue;
      }
      
      stumps[j]->threshold = best_threshold;

      if (minimal_error_threshold == 0) minimal_error_threshold = __DBL_EPSILON__;
      stumps[j]->alpha = log((float) (1 - minimal_error_threshold) / minimal_error_threshold);
      
      float error = calculate_error(dataset->samples, n_samples, weights, j, best_threshold);
      if (error == INFINITY) return 1;
      stumps[j]->error = error;
      
      // overfitting
      if (error == 0) error = INFINITY;
      
      if (error < minimal_error) {
        minimal_error = error;
        best_stump = stumps[j];  // save weak classifier
      }
    }

  // 2.1.3. Update weights
    copy_stump(classifier[t], best_stump);
    classifier[t]->sigma = dataset->sigma[classifier[t]->haar_idx];
    if (!classifier[t]) {
      fprintf(stderr, "Error: adaboost.c -> main(), * classifier[t]\n");
      free_stumps(stumps, N_FEATURES);
      free(weights);
      free_stumps(classifier, n_stumps);
      free_dataset(dataset);
      return 1;
    }

//* Test: Skalierung mit /(n_stumps+1)
//    if (n_stumps == 2) classifier[t]->threshold -= (int) dataset->sigma[classifier[t]->haar_idx] / (n_stumps+1);
    weights = update_weights(weights, dataset->samples, n_samples, best_stump, minimal_error);
    if (!weights) {
      fprintf(stderr, "Error: adaboost.c -> main(), * weights\n");
      free_stumps(stumps, N_FEATURES);
      free_stumps(classifier, n_stumps);
      free_dataset(dataset);
      return 1;
    }
    
    free_stumps(stumps, N_FEATURES);
  }

  // 2.2. Strong classifier
  save_classifier(path_layer, classifier, n_stumps);
  

  // 3. Free 
  free_stumps(classifier, n_stumps);
  free(weights);
  free_dataset(dataset);

  return 0;
}