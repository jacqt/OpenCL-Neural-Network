#include "cl_neuralnet.h"

//Used to find the (row,nodeNumber) pair that corresponds to the n'th input/errorGradient node
void inline getPosition(int n, constant int* restrict netSpec, int* restrict row, int* restrict nodeNumber)
{
    for (unsigned int i = 1; ;++i)//Termination is determined by the break statement
    {
        int k = netSpec[i];
        bool comparison = k <= n;
        if (comparison)
            n += -k;
        else
        {
            *row = i;
            *nodeNumber = n;
            break;
        }
    }
}

kernel void writeOutputToBuffer(global Layer* restrict layers, global float* restrict outputs, int lastLayer)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0);
    outputs[i] = layers[lastLayer].nodes[i].output;
}

//Rolled kernel that computes a layer's output
kernel void computeLayerOutput_Rolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0); //There will be an offset depending on the layer we are operating on

    int layer, nodeNumber, numberOfWeights;
    float t;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;

    t = 0;
    for (unsigned int j = 0; j != numberOfWeights; ++j)
        t += layers[layer].nodes[nodeNumber].weights[j] * layers[layer-1].nodes[j].output;

    layers[layer].nodes[nodeNumber].output = sigmoid(t);
}

//Unrolled kernel that computes a layer's output
kernel void computeLayerOutput_Unrolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0); //There will be an offset depending on the layer we are operating on

    int layer, nodeNumber, numberOfWeights;
    float t;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;

    t = 0;
    for (unsigned int j = 0; j != numberOfWeights; j+=5)
    {
        t += layers[layer].nodes[nodeNumber].weights[j] * layers[layer-1].nodes[j].output;
        t += layers[layer].nodes[nodeNumber].weights[j+1] * layers[layer-1].nodes[j+1].output;
        t += layers[layer].nodes[nodeNumber].weights[j+2] * layers[layer-1].nodes[j+2].output;
        t += layers[layer].nodes[nodeNumber].weights[j+3] * layers[layer-1].nodes[j+3].output;
        t += layers[layer].nodes[nodeNumber].weights[j+4] * layers[layer-1].nodes[j+4].output;
    }

    layers[layer].nodes[nodeNumber].output = sigmoid(t);
}

kernel void setInputs(global Layer* layers, constant float* inputs)
{
    const int i = get_global_id(0);
    layers[0].nodes[i].output = inputs[i];
}

//Implements the _online_ backwards propgatiaon algorithm that computes the errorGradient, then uses that value to compute the weights
//And then apply the weights immediately. This function is for all other non-input and non-output nodes
kernel void computeErrorGradient_ApplyWeightChange_Rolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0);

    //Useful variables
    int layer, nodeNumber, numberOfWeights, numberOfNodes_NextLayer;
    float errorGradient,  weightChange, output;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    numberOfNodes_NextLayer = layers[layer+1].numberOfNodes;
    output = layers[layer].nodes[nodeNumber].output;
       
    //Compute errorGradient
    errorGradient = 0;
    for (uint j = 0; j != numberOfNodes_NextLayer; ++j)
        errorGradient += layers[layer+1].nodes[j].errorGradient * layers[layer+1].nodes[j].weights[nodeNumber];
    errorGradient *= output*(1-output);

    //Use the errorGradient to compute and apply the weight change
    float NerrorGradient = N*errorGradient;
    for (uint j = 0; j != numberOfWeights; ++j)
        layers[layer].nodes[nodeNumber].weights[j] += NerrorGradient*layers[layer-1].nodes[j].output;
    layers[layer].nodes[nodeNumber].errorGradient = errorGradient;
}

//Unrolled version of the above
kernel void computeErrorGradient_ApplyWeightChange_Unrolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0);

    //Useful variables
    int layer, nodeNumber, numberOfWeights, numberOfNodes_NextLayer;
    float errorGradient, output, weightChange;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    output = layers[layer].nodes[nodeNumber].output;
    numberOfNodes_NextLayer = layers[layer+1].numberOfNodes;
       
    //Compute errorGradient
    errorGradient = 0;
    for (uint j = 0; j != numberOfNodes_NextLayer; j+=5)
    {
        errorGradient += layers[layer+1].nodes[j].errorGradient * layers[layer+1].nodes[j].weights[nodeNumber];
        errorGradient += layers[layer+1].nodes[j+1].errorGradient * layers[layer+1].nodes[j+1].weights[nodeNumber];
        errorGradient += layers[layer+1].nodes[j+2].errorGradient * layers[layer+1].nodes[j+2].weights[nodeNumber];
        errorGradient += layers[layer+1].nodes[j+3].errorGradient * layers[layer+1].nodes[j+3].weights[nodeNumber];
        errorGradient += layers[layer+1].nodes[j+4].errorGradient * layers[layer+1].nodes[j+4].weights[nodeNumber];
    }
    errorGradient *= output*(1-output);

    //Use the errorGradient to compute and apply the weight change
    float NerrorGradient = N*errorGradient;
    for (uint j = 0; j != numberOfWeights; j+=5)
    {
        //Partial unroll
        layers[layer].nodes[nodeNumber].weights[j] += NerrorGradient * layers[layer-1].nodes[j].output;
        layers[layer].nodes[nodeNumber].weights[j+1] += NerrorGradient * layers[layer-1].nodes[j+1].output;
        layers[layer].nodes[nodeNumber].weights[j+2] += NerrorGradient * layers[layer-1].nodes[j+2].output;
        layers[layer].nodes[nodeNumber].weights[j+3] += NerrorGradient * layers[layer-1].nodes[j+3].output;
        layers[layer].nodes[nodeNumber].weights[j+4] += NerrorGradient * layers[layer-1].nodes[j+4].output;
    }
    layers[layer].nodes[nodeNumber].errorGradient = errorGradient;
}

//Implements the _online_ backwards propgatiaon algorithm that computes the errorGradient, then uses that value to compute the weights
//And then apply the weights immediately. This function is for the output nodes
kernel void computeErrorGradient_ApplyWeightChange_OutputNode(global Layer* restrict layers, constant int* restrict netSpec, constant int* restrict targets)
{
    const int n = get_global_size(0); //Also the size of the target array
    const int i = get_global_id(0); //Offset tells us which layer we are operating on

    //Useful variables
    int layer, nodeNumber, numberOfWeights;
    float errorGradient, output, weightChange;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    output = layers[layer].nodes[nodeNumber].output;
    
    //Compute errorGradient
    errorGradient = (targets[nodeNumber] - output)*output*(1-output);
    
    //Use the errorGradient to compute and apply the weight change
    float NerrorGradient = N*errorGradient;
    for (uint j = 0; j != numberOfWeights; j+=5)
    {
        //Partial unroll
        layers[layer].nodes[nodeNumber].weights[j] += NerrorGradient * layers[layer-1].nodes[j].output;
        layers[layer].nodes[nodeNumber].weights[j+1] += NerrorGradient * layers[layer-1].nodes[j+1].output;
        layers[layer].nodes[nodeNumber].weights[j+2] += NerrorGradient * layers[layer-1].nodes[j+2].output;
        layers[layer].nodes[nodeNumber].weights[j+3] += NerrorGradient * layers[layer-1].nodes[j+3].output;
        layers[layer].nodes[nodeNumber].weights[j+4] += NerrorGradient * layers[layer-1].nodes[j+4].output;
    }
    layers[layer].nodes[nodeNumber].errorGradient = errorGradient;
}
