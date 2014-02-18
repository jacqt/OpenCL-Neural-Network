#include "convolutionalneuralent.h"
#include "layer.h"

void ConvolutionalNetworkPortion::createConvolutionalNetwork(
    unsigned int newFilterDim,
    unsigned int newFilterNumberSize,
    unsigned int newInputDim,
    unsigned int newInputVectorNumberSize,
    cl::Buffer* newOutputLayerBuffer)
{
    convolutionalLayer = layer_newConvolutionalLayer(newFilterDim, newFilterNumberSize);
    filterDim = newFilterDim;
    filterNumberSize = newFilterNumberSize;
    inputDim = newInputDim;
    inputVectorNumberSize = newInputVectorNumberSize;
    outputLayersBuffer = newOutputLayerBuffer;
}

void ConvolutionalNetworkPortion::createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program)
{
    convolveResultDim = (inputDim - filterDim + 1); //Assume it is divisible by 2
    outputDim = convolveResultDim / MAXPOOLDIM;

    sizeOfNet = getSizeOfNet();
    sizeOfInput = sizeof(cl_float) * inputDim * inputDim * inputVectorNumberSize; 
    sizeOfConvolveResult = sizeof(cl_float) * (convolveResultDim * convolveResultDim) * filterNumberSize;
    sizeOfOutput = sizeof(cl_float) * outputDim * outputDim * filterNumberSize;
    float zeroArray [2000] = { 0 };

    //Create memory buffers
    inputsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfInput);

    cLayerBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeOfNet, convolutionalLayer);

    convolveResultBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeOfConvolveResult);

//    costBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeOfConvolveResult);

    //We pass this buffer to the next NN to act as an inputs buffer
    outputsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeOfOutput, zeroArray);

    //Create kernels
    computeConvolveResult = cl::Kernel(program, "computeConvolveResult");
    computeConvolveResult.setArg(0,cLayerBuffer);
    computeConvolveResult.setArg(1,outputsBuffer);
    computeConvolveResult.setArg(2,inputsBuffer);
    computeConvolveResult.setArg(3,convolveResultBuffer);
    computeConvolveResult.setArg(4, inputDim);
    computeConvolveResult.setArg(5, inputVectorNumberSize);

    poolConvolveResult = cl::Kernel(program, "poolConvolveResult");
    poolConvolveResult.setArg(0, cLayerBuffer);
    poolConvolveResult.setArg(1, convolveResultBuffer);
    poolConvolveResult.setArg(2, outputsBuffer);
  //poolConvolveResult.setArg(3, the index of the filter);
    

    trainConvolutionalNetworkPortion = cl::Kernel(program, "trainConvolutionalNetworkPortion");
    trainConvolutionalNetworkPortion.setArg(0, cLayerBuffer);
    trainConvolutionalNetworkPortion.setArg(1, *outputLayersBuffer);
    trainConvolutionalNetworkPortion.setArg(2, inputsBuffer);
    trainConvolutionalNetworkPortion.setArg(3, convolveResultBuffer);
//    trainConvolutionalNetworkPortion.setArg(4, costBuffer);
  //trainConvolutionalNetworkPortion.setArg(4, the index of the filter to use)
}


size_t ConvolutionalNetworkPortion::getSizeOfNet()
{
    return sizeof(*convolutionalLayer);
}

//Computes the output of the CNN; i.e. writes the output values to the output buffer on the 
//GPU memory
void ConvolutionalNetworkPortion::computeOutput(cl_float* inputs, cl::CommandQueue *queue)
{

    //Assume the inputs are flattened
    (*queue).enqueueWriteBuffer(inputsBuffer, CL_TRUE, 0, sizeOfInput, inputs);
    
    (*queue).enqueueNDRangeKernel(computeConvolveResult,
        cl::NullRange,
        cl::NDRange(convolveResultDim, convolveResultDim* filterNumberSize),
        cl::NullRange);

    poolConvolveResult.setArg(3, inputDim);
    (*queue).enqueueNDRangeKernel(poolConvolveResult,
		cl::NullRange,
        cl::NDRange(outputDim, outputDim * filterNumberSize),
        cl::NullRange);

#ifdef _DEBUG
    cl_float *outputArray = new cl_float[720];
    queue->enqueueReadBuffer(outputsBuffer,
        CL_TRUE,
        0,
        sizeof(cl_float)*720, 
        outputArray);
    for (int c = 0; c < 720; c+=100)
        cout << c << " " << outputArray[c] << " ";
#else
#endif
}

/*Trains the network. We assume we have already written the inputs to the inputbuffer
 *We do this because in a CNN the convoluted layer is never the output layer
 *Thus when we train the whole NN we first compute the output of the CNN and the NN.
 *Following that, we calculate the errors of the NN, then the errors of the CNN, applying
 *the appropriate weight change at each backprop step
 */
void ConvolutionalNetworkPortion::trainConvolutionalPortion(cl::CommandQueue *queue)
{
    trainConvolutionalNetworkPortion.setArg(4, inputDim);
    //Mini sized ndrages
    int miniRange1 = convolveResultDim/2;
    int miniRange2 = (convolveResultDim/2) * filterNumberSize;
    for (unsigned int j = 0; j != 4; ++j)
    {
        (*queue).enqueueNDRangeKernel(trainConvolutionalNetworkPortion,
            cl::NDRange((j%2)*(miniRange1), (j/2)*(miniRange1)),
            cl::NDRange(miniRange1, miniRange2),
            cl::NDRange(miniRange1, miniRange1));
    }
    /*
    (*queue).enqueueNDRangeKernel(trainConvolutionalNetworkPortion,
        cl::NullRange,
        cl::NDRange(convolveResultDim, convolveResultDim* filterNumberSize),
        cl::NDRange(24, 24));
    */
}

void ConvolutionalNeuralNetwork::createConvolutionalNeuralNetwork(
	const vector<vector<int> > &newConvolutionalNetSpec,
	const cl::Program program)
{
    //Assign value to private variable
	convolutionalNetSpec = newConvolutionalNetSpec;

    for (auto convIt = convolutionalNetSpec.begin(); convIt != convolutionalNetSpec.end(); ++convIt)
	{
        ConvolutionalNetworkPortion* newLayer = new ConvolutionalNetworkPortion;
		newLayer->createConvolutionalNetwork(
            (*convIt)[0],
            (*convIt)[1],
            (*convIt)[2],
            (*convIt)[3],
            NULL); //Assign the output buffer later

        //newLayer->createMemoryBuffersAndKernels(context, program);
		//layers.push_back(*newLayer);
    }
}
