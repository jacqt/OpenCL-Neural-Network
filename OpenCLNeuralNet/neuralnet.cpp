#include "neuralnet.h"

std::string getFileContents(const char* fileName)
{
    std::ifstream in(fileName, std::ios::in | std::ios::binary);
    if (in)
        return(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    throw(errno);
}

/////FUNCTIONS FOR TESTING//////

//Creates some training data for the neural net
int sampleFunction(int x, int y)
{
    if (x > y)
        return 1;
    else 
        return 0;
}

//Gets some test data
vector<std::tuple<float*, int*> > getTestData ()
{
    vector<std::tuple<float*,int*> > trainingDataSet;
    for (int x = -50; x != 50; ++x)
    {
        for (int y = -50; y != 50; ++y)
        {
            float* featureVector = new float[2];
            featureVector[0] = ((float) x)/30.0; //normalize the data
            featureVector[1] = ((float) y)/30.0; //normalize the data
            int* targets  = new int [1];
            targets[0] = sampleFunction(x,y);;
            trainingDataSet.push_back(std::make_tuple(featureVector, targets));
        }
    }
    std::random_shuffle(trainingDataSet.begin(), trainingDataSet.end());
    return trainingDataSet;
}


void NeuralNetwork::createNeuralNetwork(
        cl::Context &context,
        cl::Program &fullyConnectedProgram,
        cl::Program &convolutionalProgram,
        vector<cl_int> &fullyConnectedNetSpec,
        unsigned int newFilterDim, 
        unsigned int newFilterNumberSize,
        unsigned int newInputDim)
{
    //Create the fully connected portion first
    fullyConnectedPortion.createFullyConnectedNeuralNet(fullyConnectedNetSpec);
    
    //Create the buffers of the fully connected portion
    fullyConnectedPortion.createMemoryBuffersAndKernels(context, fullyConnectedProgram);

    //Create the convolutional portion
    convolutionalPortion.createConvolutionalNetwork(
        newFilterDim,
        newFilterNumberSize,
        newInputDim,
        &(fullyConnectedPortion.layersBuffer));

    convolutionalPortion.createMemoryBuffersAndKernels(context, convolutionalProgram);
}

void NeuralNetwork::computeOutput(cl_float* inputs, cl::CommandQueue *queue)
{
    //First compute output of the convolutional portion
    convolutionalPortion.computeOutput(inputs, queue);

    //Now compute the output of the fully connected portion
    fullyConnectedPortion.computeOutputWithInputNet(&convolutionalPortion.outputsBuffer, queue);
}

void NeuralNetwork::trainNeuralNet(
    vector<std::tuple<float*, int*> > *trainingData,
    cl::CommandQueue *queue,
    int trainingIterations)
{
    for (int c = 0; c != trainingIterations; ++c)
    {
        cout << "Training iteration " << c << endl;
        //Training once over all the data samples
        for (auto dataPairIt = (*trainingData).begin(); dataPairIt != (*trainingData).end(); ++dataPairIt)
        {
            float* featureVector= std::get<0>(*dataPairIt);
            int* targetVector = std::get<1>(*dataPairIt);

            computeOutput(featureVector, queue);

            fullyConnectedPortion.trainFullyConnectedPortion(
                &convolutionalPortion.outputsBuffer,
                targetVector,
                queue);

            convolutionalPortion.trainConvolutionalPortion(queue);
        }
    }
}