#include "cl_neuralnet.h"

//Rolled kernel that computes the result of performing max pooling after convolving the filters
kernel void computeConvolveResult(
    global ConvolutionalLayer* cLayer,
    global float *outputArray,
    constant float* inputArray,
    global float* convolvingResult,
    uint inputDim)
{
    //Assume the Layer array has already calculated the apprproiate erorrs
    const uint id0 = get_global_id(0);
    const uint id1 = get_global_id(1);
    const uint res_dim = get_global_size(0);
    const uint globalSize = get_global_size(1);
    const uint filterNumber = id1/res_dim;

    const uint res_row = id0;
    const uint res_col = id1 % res_dim;
    const uint filterDim = cLayer->filters[filterNumber].filterDim;
    const uint numberOfFilters = globalSize / res_dim;

    float output = 0;
    //Perform filtering operation and write it to the local output array
    for (uint i_row = 0; i_row != filterDim; ++i_row)
    {
        for (uint i_col = 0; i_col != filterDim; ++ i_col)
        {
            float a = twoD_access(cLayer->filters[filterNumber].weights, i_row, i_col, filterDim);
            float b = twoD_access(inputArray, res_row + i_row, res_col + i_col, inputDim);
            output += a*b;
        }
    }
    output += cLayer->filters[filterNumber].bias;
    convolvingResult[id0*globalSize + id1] = sigmoid(output);
    //twoD_access(convolvingResult, res_row, res_col + res_dim * filterNumber , res_dim * numberOfFilters) = sigmoid(output);
}

//Performs max pooling over 2x2 non overlapping squares if appropriate
kernel void poolConvolveResult(
    global ConvolutionalLayer* cLayer,
    global float* convolvingResult,
    global float* outputArray,
    uint inputDim)
{
    uint id0 = get_global_id(0);
    uint id1 = get_global_id(1);
    uint res_dim = MAXPOOLDIM * get_global_size(0);
    uint globalSize = get_global_size(1);
    uint filterNumber = id1/res_dim;

    uint res_row = MAXPOOLDIM * id0;
    uint res_col = (MAXPOOLDIM * id1) % res_dim;
    uint filterDim = cLayer->filters[filterNumber].filterDim;
    uint numberOfFilters = globalSize /get_global_size(0);
    uint offset = res_dim * filterNumber;
    uint width = globalSize * MAXPOOLDIM;

    float max = 0;
    for (uint i = 0; i != MAXPOOLDIM; ++i)
    {
        for (uint j = 0; j != MAXPOOLDIM; ++j)
        {
            float output = twoD_access(convolvingResult, res_row + i, res_col + j + offset, width);
            if (output > max)
                max = output;
           // if (output == 0)
                //printf("an output : %f, gsize %d\n", output, id0*globalSize + id1);
        }
    }
    outputArray[(id0*globalSize + id1)] = max;
}

//Given error values, trains the neural net
kernel void trainConvolutionalNetworkPortion(
    global ConvolutionalLayer* restrict cLayer,
    global Layer* restrict layers,
    constant float* inputArray,
    global float* convolvingResult,
    uint inputDim)
{
    //Assume the Layer array has already calculated the apprproiate erorrs
    const uint id0 = get_global_id(0);
    const uint id1 = get_global_id(1);
    const uint globalSize = get_global_size(0);
    const uint filterNumber = id1/inputDim;

    const uint res_row = id0;
    const uint res_col = id1 % inputDim;
    const uint filterDim = cLayer->filters[filterNumber].filterDim;
    const uint res_dim = inputDim - filterDim + 1;
    const uint numberOfFilters = globalSize / res_dim;

    float output = twoD_access(convolvingResult, res_row, res_col + res_dim * filterNumber , res_dim * numberOfFilters);
//    printf("output %f\n", output);

    //Calculate the sum of all error gradients
    float errorGradient = 0;
    uint subOutputDim = res_dim/MAXPOOLDIM;
    uint outputWidth = subOutputDim * numberOfFilters;
    uint outputCol = (res_col/MAXPOOLDIM) + subOutputDim * filterNumber;
    uint outputRow = res_row/MAXPOOLDIM;
    uint outputIndex = twoD_index(outputRow, outputCol, outputWidth);

    for (int i = 0; i != layers[1].numberOfNodes; ++i)
        errorGradient += layers[1].nodes[i].weights[outputIndex] * layers[1].nodes[i].errorGradient;
    errorGradient *= output * (1 - output); //the deriviative of the sigmoid function applied

    //Now find the associated costs
    for (uint i_row = 0; i_row != filterDim; ++i_row)
    {
        for (uint i_col = 0; i_col != filterDim; ++ i_col)
        {
            twoD_access(cLayer->filters[filterNumber].weights, i_row, i_col, filterDim) +=
                N * errorGradient * twoD_access(inputArray, res_row + i_row, res_col + i_col, inputDim);
            mem_fence(CLK_LOCAL_MEM_FENCE);
        }
    }
    cLayer->filters[filterNumber].bias += N * errorGradient;
}