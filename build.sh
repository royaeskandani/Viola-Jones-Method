#!/bin/bash

# 0. Parameters
F_MAX=0.1 # ! Achtung ! Abhängig von der Größe der Validierungsmenge
D_MIN=0.95
F_TARGET=0


gcc -Wall -O0 calculate_dataset.c ../Headers/*.c -o calculate_dataset
echo "-> calculate_dataset kompiliert"
gcc -Wall -O0 calculate_strong_classifier.c ../Headers/*.c -o calculate_strong_classifier
echo "-> calculate_strong_classifier kompiliert"
gcc -Wall -O0 evaluate_dataset.c ../Headers/*.c -o evaluate_dataset
echo "-> evaluate_dataset kompiliert"


gcc -Wall -O0 calculate_cascade.c ../Headers/*.c -o calculate_cascade
echo "-> calculate_cascade kompiliert"

# Vorher noch die Samples aus "Train in Cascades/00_Layer transformieren"
./calculate_cascade $F_MAX $D_MIN $F_TARGET

# mv haarcascade.txt Cascade/
# rm calculate_dataset
# rm calculate_strong_classifier
# rm calculate_cascade
# rm evaluate_dataset
# echo "-> reset"

echo "Ende."