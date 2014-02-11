#include "convolutionalneuralent.h"

void ConvolutionalNeuralNetwork::createConvolutionalNetwork(
    unsigned int newFilterDim,
    unsigned int newFilterNumber,
    unsigned int newInputDim)
{
    convolutionalLayer = *layer_newConvolutionalLayer(newFilterDim, newFilterNumber);
    filterDim = newFilterDim;
    filterNumber = newFilterNumber;
}

void ConvolutionalNeuralNetwork::createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program)
{
    sizeOfNet = getSizeOfNet();
    sizeOfInput = sizeof(cl_int)*netSpec[0];

    //Create memory buffers

    layersBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeOfNet, &convolutionalLayer);

    inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfInput);

}

int ConvolutionalNeuralNetwork::getSizeOfNet()
{
    return sizeof(ConvolutionalLayer);
}
void ConvolutionalNeuralNetwork::computeOutput(vector<vector<float> > &inputs)
{

}
