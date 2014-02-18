#include "cl_neuralnet.h"

//Rolled kernel that computes the result of performing max pooling after convolving the filters
kernel void computeConvolveResult(
    global ConvolutionalLayer* cLayer,
    global float *outputArray,
    constant float* inputArray,
    global float* convolvingResult,
    uint inputDim,
    uint inputVectorNumberSize)
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

    const uint inputArrayWidth = inputDim * inputVectorNumberSize;

    float output = 0;
    uint offset =  0;
    //Perform filtering operation on each input vector and write it to the local output array
    for (uint inputVectorNumber = 0; inputVectorNumber != inputVectorNumberSize; ++inputVectorNumber)
    {
        offset = inputVectorNumber * inputDim;
        for (uint i_row = 0; i_row != filterDim; ++i_row)
        {
            for (uint i_col = 0; i_col != filterDim; ++ i_col)
            {
                float a = twoD_access(cLayer->filters[filterNumber].weights, i_row, i_col, filterDim);
                float b = twoD_access(inputArray, res_row + i_row, res_col + i_col + offset, inputArrayWidth);
                output += a*b;
            }
        }
    }
    output += cLayer->filters[filterNumber].bias;
    convolvingResult[id0*globalSize + id1] = sigmoid(output);
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

kernel void trainConvolutionalNetworkPortion(
    global ConvolutionalLayer* restrict cLayer,
    global Layer* restrict layers,
    constant float* inputArray,
    global float* convolvingResult,
    uint inputDim)
{
    //Assume the Layer array has already calculated the apprproiate erorrs
    const uint globalID0 = get_global_id(0);
    const uint globalID1 = get_global_id(1);
    const uint globalSize0 = get_global_size(0);
    const uint globalSize1 = get_global_size(0);

    const uint localID0 = get_local_id(0);
    const uint localID1 = get_local_id(1);
    const uint localSize = get_local_size(0); //we know that the workgroup dimension is square

    const uint offset0 = get_global_offset(0); //0th index offset
    const uint offset1 = get_global_offset(1); //1st index offset

    const uint numberOfFilters = globalSize1 / localSize; //total number of filters
    const uint filterNumber = (globalID1 - offset1) / localSize; //the filter to operate on
    const uint filterDim = cLayer->filters[filterNumber].filterDim; //dimension of filter
    const uint convolveResultDim = inputDim - filterDim + 1;  //dimension of the result after convolving

    const uint id0 = offset0 + localID0; //0th index in the convolvingResult array
    const uint id1 = offset1 + localID1 + filterNumber * convolveResultDim;//1st index in the convolvingResult array
    float output = twoD_access(convolvingResult, id0, id1, convolveResultDim * numberOfFilters);//the output of the convolvingResult

    const uint input0 = id0;
    const uint input1 = offset1 + localID1;

    //Calculate the index the output of the current (filter, image sub section) pair has in the 
    //sucneural network following the current layer
    const int subOutputDim = convolveResultDim/MAXPOOLDIM;
    const int outputWidth = subOutputDim * numberOfFilters;
    const int output0 = (offset0 + localID0)/MAXPOOLDIM;
    const int output1 = ((offset1 + localID1)/MAXPOOLDIM) + subOutputDim * filterNumber;
    const int outputIndex = twoD_index(output0, output1, outputWidth);

    //Calculate the sum of all error gradients
    float errorGradient = 0;
    for (int i = 0; i != layers[1].numberOfNodes; ++i)
        errorGradient += layers[1].nodes[i].weights[outputIndex] * layers[1].nodes[i].errorGradient;
    errorGradient *= output * (1 - output); //the deriviative of the sigmoid function applied

    //Now find the associated costs
    for (uint i_row = 0; i_row != filterDim; ++i_row)
    {
        for (uint i_col = 0; i_col != filterDim; ++ i_col)
        {
            twoD_access(cLayer->filters[filterNumber].weights, i_row, i_col, filterDim) +=
                N * errorGradient * twoD_access(inputArray, input0 + i_row, input1 + i_col, inputDim) * (1.0 / (float) (convolveResultDim * convolveResultDim));
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
    cLayer->filters[filterNumber].bias += N * errorGradient * (1.0 / (float) (convolveResultDim * convolveResultDim));
}