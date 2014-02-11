#ifndef TRAINING_H
#define TRAINING_H
#include "include.h"

using std::vector;

//Function to load training data from a file and then stores it in the trainingDataVector
void loadTrainingData(std::string trainingDataFile, vector<std::tuple<float*, int*> > &trainingDataVector);
#endif