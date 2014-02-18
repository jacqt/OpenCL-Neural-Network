#include "neuralnet.h"

std::string getFileContents(const char* fileName)
{
    std::ifstream in(fileName, std::ios::in | std::ios::binary);
    if (in)
        return(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    throw(errno);
}

//Reads a .cl file and creates a program from the .cl file
//Throws exceptions if there are build errors and logs the errors
cl::Program createProgram(cl::Context &context, std::string fname, const std::string params) 
{
    cl::Program::Sources sources;
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES> ();
    
    std::string sourceCode = getFileContents(fname.c_str());
    sources.insert(sources.end(), std::make_pair(sourceCode.c_str(),
        sourceCode.length()));
    cl::Program* program = new cl::Program(context,sources);

    try
    {
        (*program).build(devices, params.c_str());
    }
    catch (cl::Error e)
    {
        cout << "Compilation build error log: " << endl <<
            (*program).getBuildInfo <CL_PROGRAM_BUILD_LOG> (devices [0]) << endl;
    }

    return (*program);
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

//Deconstructor
NeuralNetwork::~NeuralNetwork()
{
    delete convolutionalPortion;
    delete fullyConnectedPortion;
}

//Rewrite to allow for multi convolutional layers
void NeuralNetwork::loadNeuralNetworkFromFile(
    std::string netFileName,
    cl::Context &context,
    cl::Program &fullyConnectedProgram,
    cl::Program &convolutionalProgram)
{
    /*Format
     *InputDim FilterDim Maxpool filterNumberSize Netspec
     *Weights... BIAS
     *Weights... BIAS
     *....
     *Weights...
     *Weights...
     *...
     */
    //first parse the data
    std::ifstream netFile (netFileName, std::ios::in | std::ios::binary);
    string line;

    int newInputDim;
    int newFilterDim;
    int newMaxPool; //we discard this
    int newFilterNumberSize;
    vector<vector<float> > weightVector; 
    vector<int> fullyConnectedNetSpec;
    
    unsigned int lineN = 0;
    while (std::getline(netFile,line))
    {
        std::stringstream lineStream;
        lineStream << line;
        if (lineN == 0)
        {
            int val;
            lineStream >> val;
            newInputDim = val;
            
            lineStream >> val;
            newFilterDim = val;

            lineStream >> val;
            newMaxPool = val;

            lineStream >> val;
            newFilterNumberSize = val;

            while (lineStream >> val)
                fullyConnectedNetSpec.push_back(val);
            lineN = 1;
        }
        else
        {
            float val;
            vector<float> weights;
            while (lineStream >> val)
                weights.push_back(val);
            weightVector.push_back(weights);
        }
    }

    //Create the convolutional portion first. We set the output layers buffer later
    convolutionalPortion = new ConvolutionalNetworkPortion();
    convolutionalPortion->createConvolutionalNetwork(
        newFilterDim,
        newFilterNumberSize,
        newInputDim,
        1,
        NULL);
    //Set the weights
    int filterDim = newFilterDim;
    int numberOfWeights = filterDim * filterDim;
    int weightVectorIndex = 0;

    for (unsigned int i = 0; i != convolutionalPortion->filterNumberSize; ++i)
    {
        for (unsigned int w = 0; w != numberOfWeights; ++w)
            convolutionalPortion->convolutionalLayer->filters[i].weights[w] = weightVector[weightVectorIndex][w];

        convolutionalPortion->convolutionalLayer->filters[i].bias = weightVector[weightVectorIndex][numberOfWeights];
        ++weightVectorIndex;
    }

    //Create the fully connected portion 
    fullyConnectedPortion = new FullyConnectedNeuralNet();
    fullyConnectedPortion->createFullyConnectedNeuralNet(fullyConnectedNetSpec);

    //Set the weights
    for (unsigned int i = 1; i != fullyConnectedPortion->netSpec.size(); ++i)
    {
        for (unsigned int n = 0; n != fullyConnectedPortion->layers[i].numberOfNodes; ++n)
        {
            for (int w = 0; w != fullyConnectedPortion->layers[i].nodes[n].numberOfWeights; ++w)
            {
                fullyConnectedPortion->layers[i].nodes[n].weights[w] = weightVector[weightVectorIndex][w];
            }
            ++weightVectorIndex;
        }
    }
    
    //Create the buffers of the fully connected portion
    fullyConnectedPortion->createMemoryBuffersAndKernels(context, fullyConnectedProgram);
    convolutionalPortion->outputLayersBuffer = &(fullyConnectedPortion->layersBuffer);
    convolutionalPortion->createMemoryBuffersAndKernels(context, convolutionalProgram);
    writeFileCounter = 0;
}

//Need to be rewritten to allow for multi convolutional layers
void NeuralNetwork::writeNeuralNetworkToFile(cl::CommandQueue &queue)
{
    /*Format
     *InputDim FilterDim Maxpool filterNumberSize Netspec
     *Weights... BIAS
     *Weights... BIAS
     *....
     *Weights...
     *Weights...
     *...
     */

    //Implement a persistent method to store the neural net
    std::ofstream netFile;
    std::ostringstream fileNameStream;
    fileNameStream << "CNN-" << writeFileCounter << ".net";
    netFile.open(fileNameStream.str());
    writeFileCounter += 3;

    netFile << convolutionalPortion->inputDim << " " << convolutionalPortion->filterDim << " ";
    netFile << MAXPOOLDIM << " " << convolutionalPortion->filterNumberSize;
    for (unsigned int i = 0; i != fullyConnectedPortion->netSpec.size(); ++i)
        netFile << " " << fullyConnectedPortion->netSpec[i];
    netFile << endl;

    //Write the weights of the convolutional part
    queue.finish();
    ConvolutionalLayer* cLayer = new ConvolutionalLayer();
    queue.enqueueReadBuffer(convolutionalPortion->cLayerBuffer,
        CL_TRUE, 0,
        convolutionalPortion->sizeOfNet, 
        cLayer);

    unsigned int maxWeightIndex = (cLayer->filters[0].filterDim) * (cLayer->filters[0].filterDim);
    for (unsigned int i = 0; i != cLayer->numberOfFilters; ++i)
    {
        netFile << cLayer->filters[i].weights[0] << " ";
        for (unsigned int w = 1; w != maxWeightIndex; ++w)
            netFile << cLayer->filters[i].weights[w] << " ";
        netFile << cLayer->filters[i].bias << " " << endl;
    }

    //Get the weights for the fully connected part from the GPU
    Layer *layerArray = new Layer[fullyConnectedPortion->netSpec.size()];
    queue.enqueueReadBuffer(fullyConnectedPortion->layersBuffer,
        CL_TRUE,0,
        fullyConnectedPortion->sizeOfNet,
        layerArray);

    //Write the weights of the fully connected part
    for (unsigned int i = 1; i != fullyConnectedPortion->netSpec.size(); ++i)
    {
        for (unsigned int j = 0; j != layerArray[i].numberOfNodes; ++j)
        {
            netFile << layerArray[i].nodes[j].weights[0] << " ";
            for (unsigned int w = 1; w != layerArray[i].nodes[j].numberOfWeights; ++w)
                netFile << layerArray[i].nodes[j].weights[w] << " ";
            netFile << endl;
        }
    }
    netFile.close();
    delete layerArray;
    delete cLayer;
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
    fullyConnectedPortion = new FullyConnectedNeuralNet();
    fullyConnectedPortion->createFullyConnectedNeuralNet(fullyConnectedNetSpec);
    
    //Create the buffers of the fully connected portion
    fullyConnectedPortion->createMemoryBuffersAndKernels(context, fullyConnectedProgram);

    //Create the convolutional portion
    convolutionalPortion = new ConvolutionalNetworkPortion();
    convolutionalPortion->createConvolutionalNetwork(
        newFilterDim,
        newFilterNumberSize,
        newInputDim,
        1,
        &(fullyConnectedPortion->layersBuffer));

    convolutionalPortion->createMemoryBuffersAndKernels(context, convolutionalProgram);
    writeFileCounter = 0;
}

//Overloaded expression
void NeuralNetwork::calculateError(
    vector<float*> &trainingData,
    vector<int*>  &trainingLabels,
    cl::CommandQueue* queue)
{
    if (trainingData.size() != trainingLabels.size())
    {
        throw (std::invalid_argument("Data and vector label must be of the same size"));
    }
    cout << "Calculating error" << endl;
    time_t start, end;
    time(&start);
    float errors = 0;
    float total = trainingData.size();

    unsigned int dataSetSize = trainingData.size();
    for (unsigned int i = 0; i != dataSetSize; ++i)
    {
        if (i%500 == 0)
            cout << "    Testing over the " << i << "th input. # Errors: " << errors << endl;
        float* featureVector = trainingData[i];
        int* targetVector = trainingLabels[i];
        int outputSize = fullyConnectedPortion->netSpec[fullyConnectedPortion->lastLayerIndex];

        //First compute output
        computeOutput(featureVector, queue);

        //Write output to output buffer
        (*queue).enqueueNDRangeKernel(fullyConnectedPortion->writeOutputToBuffer,
            cl::NullRange,
            cl::NDRange(outputSize),
            cl::NullRange);

        //Get the result from the buffer
        float *outputArray = new float[outputSize];
        (*queue).enqueueReadBuffer(fullyConnectedPortion->outputBuffer,
            CL_TRUE,
            0,
            fullyConnectedPortion->sizeOfOutput,
            outputArray);

        //Calculate if it is an error or not
        for (int index = 0; index != outputSize; ++ index)
        {
            int target = targetVector[index];
            if (outputArray[index] > 0.5 && target == 0)
            {
                errors += 1;
                break;
            }
            else if (outputArray[index] < 0.5 && target == 1)
            {
                errors += 1;
                break;
            }
        }
        delete outputArray;

    }
    time(&end);
    cout << "    Completed in " << difftime(end ,start) << " seconds" << endl;
    cout << "NUMBER OF ERRORS: " << errors << " ERROR RATE: " << 100*(errors / total) << "%" << endl;
    writeNeuralNetworkToFile(*queue);

}
void NeuralNetwork::computeOutput(cl_float* inputs, cl::CommandQueue *queue)
{
    //First compute output of the convolutional portion
    
    convolutionalPortion->computeOutput(inputs, queue);

    //Now compute the output of the fully connected portion
    fullyConnectedPortion->computeOutputWithInputNet(&(convolutionalPortion->outputsBuffer), queue);

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

            fullyConnectedPortion->trainFullyConnectedPortion(
                targetVector,
                queue);

            convolutionalPortion->trainConvolutionalPortion(queue);
        }
    }
}

void NeuralNetwork::trainNeuralNet(
    vector<float*> &trainingData,
    vector<int*> &trainingLabels,
    cl::CommandQueue* queue,
    int trainingIterations)
{
    if (trainingData.size() != trainingLabels.size())
    {
        throw (std::invalid_argument("Data and vector label must be of the same size"));
    }
    
    unsigned int dataSetSize = trainingData.size();

    for (int c = 0; c != trainingIterations; ++c)
    {
        time_t start, end;
        time(&start);
        cout << "Training iteration " << c << endl;

        //Training once over all the data samples
        for (unsigned int i = 0; i != dataSetSize; ++i)
        {
            if (i % 2000 == 0)
                cout << "    Training over the " << i << "th data member..." << endl;
            float* featureVector = trainingData[i];
            int* targetVector = trainingLabels[i];

            computeOutput(featureVector, queue);

            fullyConnectedPortion->trainFullyConnectedPortion(
                targetVector,
                queue);

            convolutionalPortion->trainConvolutionalPortion(queue);
        }
        time(&end);
        cout << "    Completed in " << difftime(end ,start) << " seconds" << endl;
        if (c % 3 == 0)
            calculateError(trainingData, trainingLabels, queue);
    }

}