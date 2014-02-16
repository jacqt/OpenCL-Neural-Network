#include "layer.h"
//Function to return a random float between the supplied lower bound and upper bound
float getRandomFloat(float lowerbound, float upperbound)
{
    float f = (float)rand() / RAND_MAX;
    f = lowerbound + f * (upperbound - lowerbound);
    return f;
}

//Creates the input layer that has no nodes feeding into it
Layer* layer_newInputLayer(int numberOfNodes)
{
    Layer* netLayer = new Layer();
    netLayer->numberOfNodes = numberOfNodes;
    for (int i = 0; i != numberOfNodes; ++i)
    {
        Node node;
        node.numberOfWeights = 0;
        node.output = 0;
        netLayer->nodes[i] = node;
    }
    return netLayer;
}

//Creates a layer with nodes feeding into it
Layer* layer_new(int numberOfNodes, int numberOfWeights)
{
    Layer* netLayer = new Layer();
    (*netLayer).numberOfNodes = numberOfNodes;
    for (int i = 0; i != numberOfNodes; ++i)
    {
        Node* node = new Node();
        node->numberOfWeights = numberOfWeights;
        node->output = 0;
        for (int j = 0; j != numberOfWeights; ++j)
            node->weights[j] = getRandomFloat(-0.1,0.1);

        netLayer->nodes[i] = *node;
    }
    return netLayer;
}

//Creates a convolutional layer
ConvolutionalLayer* layer_newConvolutionalLayer(unsigned int filterDim, unsigned int filterNumberSize)
{
    ConvolutionalLayer* newCLayer = new ConvolutionalLayer();
    newCLayer->numberOfFilters = filterNumberSize;
    
    for (unsigned int i = 0; i != filterNumberSize; ++i)
    {
        Filter* filter = new Filter;
        filter->filterDim = filterDim;
        for (int k = 0; k != filterDim*filterDim; ++k)
            filter->weights[k] = getRandomFloat(-0.1,0.1);
        filter->bias = getRandomFloat(-0.1,0.1);
        filter->filterNumber = i;
        newCLayer->filters[i] = *filter;
    }
    return newCLayer;
}
