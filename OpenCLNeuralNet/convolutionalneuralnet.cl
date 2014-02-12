#include "cl_neuralnet.h"

//Rolled kernel that computes the result of performing max pooling after convolving the filters
kernel void computeConvolveResult(
    global ConvolutionalLayer* cLayer,
    global float *outputArray,
    constant float* inputArray,
    local float* convolvingResult,
    uint filterNumber)
{
    const uint res_row = get_local_id(0);
    const uint res_col = get_local_id(1);
    const uint res_dim = get_local_size(0);
    const uint numberOfFilters = get_num_groups(0);
    
    //Perform filtering operation and write it to the local output array
    const uint filterDim = cLayer->filters[filterNumber].filterDim;
    float output = 0;
    for (uint i_row = 0; i_row != filterDim; ++i_row)
    {
        for (uint i_col = 0; i_col != filterDim; ++ i_col)
            output += twoD_access(cLayer->filters[filterNumber].weights, i_row, i_col, filterDim) * 
                twoD_access(inputArray, res_row + i_row, res_col + i_col, res_dim + filterDim);
    }
    output = sigmoid(output);
    twoD_access(convolvingResult, res_row, res_col, res_dim) = output;

    //Now perform max pooling over 2x2 non overlapping squares if appropriate
    if ((res_row % MAXPOOLDIM  == 0) && (res_col % MAXPOOLDIM == 0))
    {
        float output1 = output;
        float output2 = twoD_access(convolvingResult, res_row, res_col + 1, res_dim);
        float output3 = twoD_access(convolvingResult, res_row + 1, res_col, res_dim);
        float output4 = twoD_access(convolvingResult, res_row + 1, res_col + 1, res_dim);
        float k1 = fmax(output1, output2);
        float k2 = fmax(output3, output4);
        uint subOutputDim = res_dim/MAXPOOLDIM;
        uint outputWidth = subOutputDim * numberOfFilters
        uint outputCol = (res_col/MAXPOOLDIM) + subOutputDim * filterNumber;
        uint outputRow = res_row/MAXPOOLDIM;
        twoD_access(outputArray,outputRow, outputCol, outputWidth) = fmax(k1, k2);
    }
}

//Given an error, trains the neural net
kernel void trainConvolutionalNeuralNet()
{
}