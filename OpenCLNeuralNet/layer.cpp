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
        Node node;
        node.numberOfWeights = numberOfWeights;
        node.output = 0;
        for (int j = 0; j != numberOfWeights; ++j)
            node.weights[j] = getRandomFloat(-0.15,0.15);

        netLayer->nodes[i] = node;
    }
    return netLayer;
}

//Creates a convolutional layer
ConvolutionalLayer* layer_newConvolutionalLayer(unsigned int filterDim, unsigned int filterNumber)
{
    ConvolutionalLayer* newCLayer = new ConvolutionalLayer();
    
    for (unsigned int i = 0; i != filterNumber; ++i)
    {
        Filter filter;
        filter.filterDim = filterDim;
        filter.filterNumber = i;
        (*newCLayer).filters[i] = filter;
    }
    return newCLayer;
}
