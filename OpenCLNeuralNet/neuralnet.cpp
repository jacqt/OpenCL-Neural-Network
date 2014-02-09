#include "neuralnet.h"

std::string getFileContents(const char* fileName)
{
    std::ifstream in(fileName, std::ios::in | std::ios::binary);
    if (in)
        return(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    throw(errno);
}

//Create the neural net as a vector of layers
void NeuralNet::createNeuralNet(vector<cl_int> &newNetSpec)
{
    //Create the input layer
    netSpec = newNetSpec;
    Layer inputLayer = layer_newInputLayer(netSpec[0]);
    layers.push_back(inputLayer);

    //Create the rest of the layers
    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        Layer layer = layer_new(netSpec[i],netSpec[i-1]);
        layers.push_back(layer);
    }
}

void NeuralNet::createMemoryBuffers(cl::Context &context)
{
    sizeOfNet = getSizeOfNet();
    sizeOfInput = sizeof(cl_int)*netSpec[0];
    sizeOfTarget = sizeof(cl_int)*netSpec[netSpec.size()-1]; 

    //Create memory buffers
    netSpecBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_int)*netSpec.size(), &netSpec[0]);

    layersBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeOfNet, &layers[0]);

    inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfInput);

    targetBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfTarget);
}

//Loads a neural net from a file
void NeuralNet::loadNeuralNetFromFile(std::string netFileName)
{
    //first parse the data
    std::ifstream netFile ("C:/cpp/cppNeuralNet/neuralNet.net", std::ios::in | std::ios::binary);
    string line;
    int numberOfInputs;
    vector<vector<double> > weightVector; 

    unsigned int lineN = 0;
    while (std::getline(netFile,line))
    {
        std::stringstream lineStream;
        lineStream << line;
        if (lineN == 0)
        {
            int val;
            int i = 0;
            while (lineStream >> val)
                netSpec.push_back(val);
            lineN = 1;
        }
        else
        {
            double val;
            vector<double> weights;
            while (lineStream >> val)
                weights.push_back(val);
            weightVector.push_back(weights);
        }
    }
    //Create the input layer
    Layer inputLayer = layer_newInputLayer(netSpec[0]);
    layers.push_back(inputLayer);

    //Create the rest of the layers
    int w = 0;
    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        Layer layer = layer_new(netSpec[i],netSpec[i-1]);
        for (int j = 0; j != layer.numberOfNodes; ++j)
        {
            for (int k = 0; k != layer.nodes[j].numberOfWeights; ++k)
                layer.nodes[j].weights[k] = weightVector[w][k];
            ++w;
        }
        layers.push_back(layer);
    }
}


//Returns the size of the neural net in bytes
int NeuralNet::getSizeOfNet()
{
    int size = 0;
    for (auto it = layers.begin(); it != layers.end(); ++it)
        size += layer_size(&(*it));

    return size;
}

//Computes the output of the net given an array of inputs
void NeuralNet::computeOutput(
    cl::Context *context,
    cl_float *inputs,
    cl::Program *program,
    cl::CommandQueue *queue)
{
    cl::Kernel setInputKernel;
    setInputKernel = cl::Kernel(*program, "setInputs");
    setInputKernel.setArg(0,layersBuffer);
    setInputKernel.setArg(1,inputBuffer);

    (*queue).enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, sizeOfInput, inputs);
    (*queue).enqueueNDRangeKernel(setInputKernel,cl::NullRange, cl::NDRange(netSpec[0]), cl::NullRange);

    //Now compute the output
    unsigned int offset = 0;
    cl::Kernel kernel;
    kernel = cl::Kernel(*program, "computeLayerOutput");
    kernel.setArg(0, layersBuffer);
    kernel.setArg(1, netSpecBuffer);
    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        (*queue).enqueueNDRangeKernel(kernel, cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        offset += netSpec[i];
    }
}

void NeuralNet::calculateError(
    cl::Context *context,
    vector<std::tuple<float*,int*> > *trainingData,
    cl::Program *program,
    cl::CommandQueue *queue)
{
    float errors = 0;
    float total = (*trainingData).size();
    int i = 0;
    for (auto dataPairIt = (*trainingData).begin(); dataPairIt != (*trainingData).end(); ++dataPairIt)
    {
        float *featureVector = std::get<0>(*dataPairIt);
        int *targetVector = std::get<1>(*dataPairIt);

        //First compute output
        computeOutput(context, featureVector, program, queue);

        //Get the result from the buffer
        Layer *layerArray = new Layer[netSpec.size()];
        (*queue).enqueueReadBuffer(layersBuffer, CL_TRUE,0,sizeOfNet,layerArray);
        int lastLayerIndex = netSpec.size()-1;
        float output = layerArray[lastLayerIndex].nodes[layerArray[lastLayerIndex].numberOfNodes-1].output;

        //Calculate if it is an error or not
        int target = targetVector[0];
        if (output > 0.5 && target == 0)
            errors += 1;
        else if (output < 0.5 && target == 1)
            errors += 1;

        ++i;
        delete layerArray;
    }
    cout << "NUMBER OF ERRORS: " << errors << " ERROR RATE: " << 100*(errors / total) << endl;
}

//Trains the neural net given a vector of tuples containing the feature vector
//  and target vector
void NeuralNet::trainNeuralNet(
    cl::Context *context,
    vector<std::tuple<float*,int*> > *trainingData,
    cl::Program *program,
    cl::CommandQueue *queue,
    int trainingIterations)
{
    //Create the kernel to compute the deltas and apply the weight changes for the rest of the layers
    cl::Kernel calcLayerDeltas;
    calcLayerDeltas = cl::Kernel(*program, "computeDelta_ApplyWeightChange");
    calcLayerDeltas.setArg(0,layersBuffer);
    calcLayerDeltas.setArg(1,netSpecBuffer);

    int lastLayerIndex = netSpec.size()-1;
    int *targetVector = std::get<1>((*trainingData)[0]);

    //Now create the kernel to calculate the deltas in the output layer and apply the appropriate weight changes
    cl::Kernel calcOutputLayerDeltas;
    calcOutputLayerDeltas = cl::Kernel(*program, "computeDelta_ApplyWeightChange_OutputNode");
    calcOutputLayerDeltas.setArg(0, layersBuffer);
    calcOutputLayerDeltas.setArg(1, netSpecBuffer);
    calcOutputLayerDeltas.setArg(2, targetBuffer);

    for (int c = 0; c != trainingIterations; ++c)
    {
        if (c%11 == 5)
            calculateError(context, trainingData, program, queue);

        cout << "Training iteration " << c << endl;
        //Training once over all the data samples
        for (auto dataPairIt = (*trainingData).begin(); dataPairIt != (*trainingData).end(); ++dataPairIt)
        {
            float *featureVector = std::get<0>(*dataPairIt);
            targetVector = std::get<1>(*dataPairIt);

            //First compute output
            computeOutput(context, featureVector, program, queue);

            //Calculate the appropriate offset
            int offset = 0;
            for (unsigned int i = 1; i != netSpec.size()-1 ; ++i)
                offset += netSpec[i];
            
            //Queue the writebuffer
            (*queue).enqueueWriteBuffer(targetBuffer,CL_TRUE,0,sizeOfTarget,&targetVector[0]);

            //Queue the kernel
            (*queue).enqueueNDRangeKernel(calcOutputLayerDeltas, cl::NDRange(offset), cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

            //Queue the kernels 
            for (unsigned int i = netSpec.size()-2; i != 0; --i)
            {
                /*
                cl_ulong queued;
                cl_ulong submit;
                cl_ulong start;
                cl_ulong end;
                */
                cl::Event event;
                offset += -netSpec[i];
                (*queue).enqueueNDRangeKernel(calcLayerDeltas, cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange,NULL,&event);
                (*queue).flush();
                event.wait();
                /*
                event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_SUBMIT, &submit);
                event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
                event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end);
                event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_QUEUED, &queued);

                cout << "queued" << queued << " dT=0" << endl;
                cout << "submit" << submit << " dT=" << submit - queued << endl;
                cout << "start" << start << " dT=" << start - submit << endl;
                cout << "end" << end << " dT=" << end - start << endl;
                */
            }
        }
    }
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
            featureVector[0] = ((float) x)/30.0;
            featureVector[1] = ((float) y)/30.0;
            int* targets  = new int [1];
            targets[0] = sampleFunction(x,y);;
            trainingDataSet.push_back(std::make_tuple(featureVector, targets));
        }
    }
    std::random_shuffle(trainingDataSet.begin(), trainingDataSet.end());
    return trainingDataSet;
}