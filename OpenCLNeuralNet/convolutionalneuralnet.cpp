#include "convolutionalneuralent.h"

void ConvolutionalNetworkPortion::createConvolutionalNetwork(
    unsigned int newFilterDim,
    unsigned int newFilterNumberSize,
    unsigned int newInputDim,
    cl::Buffer* newOutputLayerBuffer)
{
    convolutionalLayer = *layer_newConvolutionalLayer(newFilterDim, newFilterNumberSize);
    filterDim = newFilterDim;
    filterNumberSize = newFilterNumberSize;
    inputDim = newInputDim;
    outputLayersBuffer = newOutputLayerBuffer;
}

void ConvolutionalNetworkPortion::createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program)
{
    convolveResultDim = (inputDim - filterDim + 1); //Assume it is divisible by 2
    outputDim = convolveResultDim / 2;

    sizeOfNet = getSizeOfNet();
    sizeOfInput = sizeof(cl_int)*inputDim;
    sizeOfConvolveResult = sizeof(cl_float) * (convolveResultDim * convolveResultDim);
    sizeOfOutput = sizeOfConvolveResult / 4;

    //Create memory buffers
    cLayerBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeOfNet, &convolutionalLayer);

    inputsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfInput);

    //We pass this buffer to the next NN to act as an inputs buffer
    outputsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

    //Create kernels
    computeConvolveResult = cl::Kernel(program, "computeConvolveResult");
    computeConvolveResult.setArg(0,cLayerBuffer);
    computeConvolveResult.setArg(1,outputsBuffer);
    computeConvolveResult.setArg(2,inputsBuffer);
    computeConvolveResult.setArg(3,sizeOfConvolveResult, NULL);
  //computeConvolveResult.setArg(4, the index of the filter to use)

    trainConvolutionalNetworkPortion = cl::Kernel(program, "trainConvolutionalNetworkPortion");
    trainConvolutionalNetworkPortion.setArg(0, cLayerBuffer);
    trainConvolutionalNetworkPortion.setArg(1, *outputLayersBuffer);
    trainConvolutionalNetworkPortion.setArg(2, inputsBuffer);
    trainConvolutionalNetworkPortion.setArg(3, sizeOfConvolveResult, NULL);
  //trainConvolutionalNeuralNet.setArg(4, the index of the filter to use)
}


size_t ConvolutionalNetworkPortion::getSizeOfNet()
{
    return sizeof(ConvolutionalLayer);
}

//Computes the output of the CNN; i.e. writes the output values to the output buffer on the 
//GPU memory
void ConvolutionalNetworkPortion::computeOutput(float* inputs, cl::CommandQueue *queue)
{
    //Assume the inputs are flattened
    (*queue).enqueueWriteBuffer(inputsBuffer, CL_TRUE, 0, sizeOfInput, inputs);
    
    //Set final arg for computeConvolveResult
    for (unsigned int i = 0; i != filterNumberSize; ++i)
    {
        computeConvolveResult.setArg(4, i);
        (*queue).enqueueNDRangeKernel(computeConvolveResult,cl::NullRange,
            cl::NDRange(convolveResultDim, filterNumberSize * convolveResultDim),
            cl::NDRange(convolveResultDim,convolveResultDim));
    }
}

/*Trains the network. We assume we have already written the inputs to the inputbuffer
 *We do this because in a CNN the convoluted layer is never the output layer
 *Thus when we train the whole NN we first compute the output of the CNN and the NN.
 *Following that, we calculate the errors of the NN, then the errors of the CNN, applying
 *the appropriate weight change at each backprop step
 */
void ConvolutionalNetworkPortion::trainConvolutionalPortion(cl::CommandQueue *queue)
{
    for (unsigned int i = 0; i != filterNumberSize; ++i)
    {
        computeConvolveResult.setArg(4, i);
        (*queue).enqueueNDRangeKernel(trainConvolutionalNetworkPortion,cl::NullRange,
            cl::NDRange(convolveResultDim, filterNumberSize * convolveResultDim),
            cl::NDRange(convolveResultDim,convolveResultDim));
    }
}
