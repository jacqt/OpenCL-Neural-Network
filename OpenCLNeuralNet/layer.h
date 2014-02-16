#ifndef LAYER_H
#define LAYER_H
#include "include.h"

#define MAXSIZE 3400
#define MAXFILTERDIM 15
#define MAXFILTERS 20
#define MAXPOOLDIM 2
//Define a maxsize because pointers are not allowed to be passed to the kernel
//Note that this number _must_ be the same as the MAXSIZE defined under neuralnet.cl


//We define a layer struct along with node struct because we want to pass these structs to the
//kernel running on the GPU which does not support C++ types
typedef struct Node
{
    int numberOfWeights;
    float weights[MAXSIZE]; 
    float output;
    float input;
    float errorGradient;
} Node;

typedef struct Filter
{
    int filterDim;
    int filterNumber;
    float weights[MAXFILTERDIM*MAXFILTERDIM]; 
    float bias;
    float costs[MAXFILTERDIM];
    float errorGradient;//sum of the error gradients
} Filter;

typedef struct Layer
{
    int numberOfNodes;
    Node nodes[MAXSIZE];
} Layer;

typedef struct ConvolutionalLayer
{
    int numberOfFilters;
    Filter filters[MAXFILTERS];
} ConvolutionalLayer;

float getRandomFloat(float lowerbound, float upperbound);

Layer* layer_newInputLayer(int numberOfNodes);

Layer* layer_new(int numberOfNodes, int numberOfWeights);

ConvolutionalLayer* layer_newConvolutionalLayer(unsigned int filterDim, unsigned int filterNumber);
#endif