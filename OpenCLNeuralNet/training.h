#ifndef INCLUDED_STRING_VECTOR_STREAM
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#define INCLUDED_STRING_VECTOR_STREAM
#else
#endif

using std::vector;

//Function to load training data from a file and then stores it in the trainingDataVector
void loadTrainingData(std::string trainingDataFile, vector<std::tuple<float*, int*> > &trainingDataVector);