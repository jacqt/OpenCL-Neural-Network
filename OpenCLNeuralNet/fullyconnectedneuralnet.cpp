#include "fullyconnectedneuralnet.h"

//Create the neural net as a vector of layers
void FullyConnectedNeuralNet::createFullyConnectedNeuralNet(vector<cl_int> &newNetSpec)
{
    //Create the input layer
    netSpec = newNetSpec;
    Layer* inputLayer = layer_newInputLayer(netSpec[0]);
    layers.push_back(*inputLayer);

    //Create the rest of the layers
    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        Layer* layer = layer_new(netSpec[i],netSpec[i-1]);
        layers.push_back(*layer);
    }

    writeFileCounter = 0;
}

void FullyConnectedNeuralNet::writeFullyConnectedNeuralNetToFile(cl::CommandQueue &queue)
{
    //Implement a persistent method to store the neural net
    std::ofstream netFile;
    std::ostringstream fileNameStream;
    fileNameStream << "NN-" << writeFileCounter << ".net";
    netFile.open(fileNameStream.str());
    ++writeFileCounter;

    /*File format:
     * N_INPUTS N_NODES_0 N_NODES_1 ... 
     * WEIGHT1 WEIGHT2 ...
     * WEIGHT1 WEIGHT2 ...
     * etc.
     */

    netFile << netSpec[0];
    for (unsigned int i = 1; i != netSpec.size(); ++i)
        netFile << " " << netSpec[i];
    netFile << endl;

    //Get the weights from the GPU
    Layer *layerArray = new Layer[netSpec.size()];
    queue.enqueueReadBuffer(layersBuffer, CL_TRUE,0,sizeOfNet,layerArray);

    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        for (unsigned int j = 0; j != layerArray[i].numberOfNodes; ++j)
        {
            netFile << layerArray[i].nodes[j].weights[0] << " ";
            for (unsigned int w = 1; w != layerArray[i].nodes[j].numberOfWeights; ++w)
                netFile << layerArray[i].nodes[j].weights[w] << " ";
            netFile << endl;
        }
    }
    delete layerArray;
    netFile.close();
}

void FullyConnectedNeuralNet::createMemoryBuffersAndKernels(cl::Context &context, cl::Program &program)
{
    sizeOfNet = getSizeOfNet();
    sizeOfInput = sizeof(cl_int)*netSpec[0];
    sizeOfTarget = sizeof(cl_int)*netSpec[netSpec.size()-1]; 
    sizeOfOutput = sizeof(cl_float)*netSpec[netSpec.size()-1];
    lastLayerIndex = netSpec.size()-1;

    //Create memory buffers
    netSpecBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_int)*netSpec.size(), &netSpec[0]);

    layersBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeOfNet, &layers[0]);

    inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfInput);

    targetBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeOfTarget);

    outputBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeOfOutput);

    //Create input kernels
    setInputKernel = cl::Kernel(program, "setInputs");
    setInputKernel.setArg(0,layersBuffer);
    setInputKernel.setArg(1,inputBuffer);

    //Create compute output kernels
    computeOutputRolled = cl::Kernel(program, "computeLayerOutput_Rolled");
    computeOutputRolled.setArg(0, layersBuffer);
    computeOutputRolled.setArg(1, netSpecBuffer);

    computeOutputUnrolled = cl::Kernel(program, "computeLayerOutput_Unrolled");
    computeOutputUnrolled.setArg(0, layersBuffer);
    computeOutputUnrolled.setArg(1, netSpecBuffer);

    //Create the kernel to compute the errorGradients
    //and apply the weight changes for the rest of the layers
    calcLayerErrorGradientsRolled = cl::Kernel(program, 
        "computeErrorGradient_ApplyWeightChange_Rolled");
    calcLayerErrorGradientsRolled.setArg(0,layersBuffer);
    calcLayerErrorGradientsRolled.setArg(1,netSpecBuffer);

    calcLayerErrorGradientsUnrolled= cl::Kernel(program,
        "computeErrorGradient_ApplyWeightChange_Unrolled");
    calcLayerErrorGradientsUnrolled.setArg(0,layersBuffer);
    calcLayerErrorGradientsUnrolled.setArg(1,netSpecBuffer);

    //Now create the kernel to calculate the errorGradients in the output layer
    //and apply the appropriate weight changes
    calcOutputLayerErrorGradients = cl::Kernel(program,
        "computeErrorGradient_ApplyWeightChange_OutputNode");
    calcOutputLayerErrorGradients.setArg(0, layersBuffer);
    calcOutputLayerErrorGradients.setArg(1, netSpecBuffer);
    calcOutputLayerErrorGradients.setArg(2, targetBuffer);

    //Create kernel to write outputs to the output buffer
    writeOutputToBuffer = cl::Kernel(program, "writeOutputToBuffer");
    writeOutputToBuffer.setArg(0, layersBuffer);
    writeOutputToBuffer.setArg(1, outputBuffer);
    writeOutputToBuffer.setArg(2, lastLayerIndex);
}

//Loads a neural net from a file
void FullyConnectedNeuralNet::loadFullyConnectedNeuralNetFromFile(std::string netFileName)
{
    //first parse the data
    std::ifstream netFile (netFileName, std::ios::in | std::ios::binary);
    string line;
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
    Layer* inputLayer = layer_newInputLayer(netSpec[0]);
    layers.push_back(*inputLayer);

    //Create the rest of the layers
    int w = 0;
    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        Layer* layer = layer_new(netSpec[i],netSpec[i-1]);
        for (int j = 0; j != (*layer).numberOfNodes; ++j)
        {
            for (int k = 0; k != (*layer).nodes[j].numberOfWeights; ++k)
                (*layer).nodes[j].weights[k] = weightVector[w][k];
            ++w;
        }
        layers.push_back(*layer);
    }
    writeFileCounter = 0;
}

//Returns the size of the neural net in bytes
int FullyConnectedNeuralNet::getSizeOfNet()
{
    return layers.size()*sizeof(Layer);
}

//Computes the output of the net given an array of inputs
void FullyConnectedNeuralNet::computeOutput(
    cl_float *inputs,
    cl::CommandQueue *queue)
{
    //Change the arg so that the setInputKernel uses the correct buffer
    setInputKernel.setArg(1,inputBuffer);
    (*queue).enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, sizeOfInput, inputs);
    (*queue).enqueueNDRangeKernel(setInputKernel,cl::NullRange, cl::NDRange(netSpec[0]), cl::NullRange);

    //Now compute the output
    unsigned int offset = 0;

    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        if (layers[i].nodes[0].numberOfWeights % 5 == 0)
            (*queue).enqueueNDRangeKernel(computeOutputUnrolled, cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        else
            (*queue).enqueueNDRangeKernel(computeOutputRolled, cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        offset += netSpec[i];
    }
}

void FullyConnectedNeuralNet::computeOutputWithInputNet(
    cl::Buffer* outputFromPreviousNetBuffer,
    cl::CommandQueue* queue)
{
//    Layer *layerArray = new Layer[netSpec.size()];
//    queue->enqueueReadBuffer(layersBuffer, CL_TRUE,0,4*sizeof(Layer),layerArray);
//    Layer *layerArray2 = &layerArray[1];

    //Change the arg so that the setInputKernel uses the correct buffer
    setInputKernel.setArg(1,*outputFromPreviousNetBuffer);
    (*queue).enqueueNDRangeKernel(setInputKernel,cl::NullRange, cl::NDRange(netSpec[0]), cl::NullRange);

    //Now compute the output
    unsigned int offset = 0;

    for (unsigned int i = 1; i != netSpec.size(); ++i)
    {
        if (layers[i].nodes[0].numberOfWeights % 5 == 0)
            (*queue).enqueueNDRangeKernel(computeOutputUnrolled, cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        else
            (*queue).enqueueNDRangeKernel(computeOutputRolled, cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        offset += netSpec[i];
    }
}

void FullyConnectedNeuralNet::calculateError(
    vector<std::tuple<float*,int*> > *trainingData,
    cl::CommandQueue *queue)
{
    cout << "Calculating error" << endl;
    float errors = 0;
    float total = (*trainingData).size();
    for (auto dataPairIt = (*trainingData).begin(); dataPairIt != (*trainingData).end(); ++dataPairIt)
    {
        float *featureVector = std::get<0>(*dataPairIt);
        int *targetVector = std::get<1>(*dataPairIt);

        //First compute output
        computeOutput(featureVector, queue);

        //Write output to output buffer
        (*queue).enqueueNDRangeKernel(writeOutputToBuffer,cl::NullRange, cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

        //Get the result from the buffer
        float *outputArray = new float[netSpec[lastLayerIndex]];
        (*queue).enqueueReadBuffer(outputBuffer, CL_TRUE,0,sizeOfOutput,outputArray);

        //Calculate if it is an error or not
        for (int index = 0; index != netSpec[lastLayerIndex]; ++ index)
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
    cout << "NUMBER OF ERRORS: " << errors << " ERROR RATE: " << 100*(errors / total) << endl;
}

void FullyConnectedNeuralNet::calculateError(
    vector<float*> &trainingData,
    vector<int*> &trainingLabel,
    cl::CommandQueue *queue)
{
    if (trainingLabel.size() != trainingData.size())
    {
        throw std::invalid_argument("training label and training data must have the same size!");
    }
    cout << "Calculating error" << endl;
    time_t start, end;
    time(&start);

    float errors = 0;
    unsigned int dataSetSize = trainingLabel.size();
    for (int i = 0; i != dataSetSize; ++i)
    {
        if (i%10000 == 0)
        {
            cout << "    Checking input " << i << ". # Errors: " << errors << endl;
        }
        float* featureVector = trainingData[i];
        int* targetVector = trainingLabel[i];

        //First compute output
        computeOutput(featureVector, queue);

        //Write output to output buffer
        (*queue).enqueueNDRangeKernel(writeOutputToBuffer,cl::NullRange, cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

        //Get the result from the buffer
        float *outputArray = new float[netSpec[netSpec.size()-1]];
        (*queue).enqueueReadBuffer(outputBuffer, CL_TRUE,0,sizeOfOutput,outputArray);

        //Calculate if it is an error or not
        for (int index = 0; index != netSpec[lastLayerIndex]; ++ index)
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
    cout << "NUMBER OF ERRORS: " << errors << " ERROR RATE: " << 100*(errors / (float)dataSetSize) << "%" << endl;
}

void FullyConnectedNeuralNet::calcQuickError(
    vector<float*> &trainingData,
    vector<int*> &trainingLabel,
    cl::CommandQueue *queue)
{
    if (trainingLabel.size() != trainingData.size())
    {
        throw std::invalid_argument("training label and training data must have the same size!");
    }
    cout << "Calculating quick error estimate" << endl;
    time_t start, end;
    time(&start);

    float errors = 0;
    unsigned int dataSetSize = trainingLabel.size();
    for (int i = 0; i < dataSetSize; i += 10)
    {
        if (i%10000 == 0)
        {
            cout << "    Checking input " << i << ". # Errors: " << errors << endl;
        }
        float* featureVector = trainingData[i];
        int* targetVector = trainingLabel[i];

        //First compute output
        computeOutput(featureVector, queue);

        //Write output to output buffer
        (*queue).enqueueNDRangeKernel(writeOutputToBuffer,cl::NullRange, cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

        //Get the result from the buffer
        float *outputArray = new float[netSpec[netSpec.size()-1]];
        (*queue).enqueueReadBuffer(outputBuffer, CL_TRUE,0,sizeOfOutput,outputArray);

        //Calculate if it is an error or not
        for (int index = 0; index != netSpec[lastLayerIndex]; ++ index)
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
    cout << "NUMBER OF ERRORS: " << errors << " ERROR RATE: " << 1000*(errors / (float)dataSetSize) << "%" << endl;
}

//Trains the neural net given a vector of tuples containing the feature vector
//  and target vector
void FullyConnectedNeuralNet::trainFullyConnectedNeuralNet(
    vector<std::tuple<float*,int*> > *trainingData,
    cl::CommandQueue *queue,
    int trainingIterations)
{
    for (int c = 0; c != trainingIterations; ++c)
    {
        if (c%11 == 5)
            calculateError(trainingData, queue);

        cout << "Training iteration " << c << endl;
        //Training once over all the data samples
        for (auto dataPairIt = (*trainingData).begin(); dataPairIt != (*trainingData).end(); ++dataPairIt)
        {
            float* featureVector = std::get<0>(*dataPairIt);
            int* targetVector = std::get<1>(*dataPairIt);

            //First compute output
            computeOutput(featureVector, queue);

            //Calculate the appropriate offset
            int offset = 0;
            for (unsigned int i = 1; i != netSpec.size()-1 ; ++i)
                offset += netSpec[i];
            
            //Queue the writebuffer
            (*queue).enqueueWriteBuffer(targetBuffer,CL_TRUE,0,sizeOfTarget,&targetVector[0]);

            //Queue the kernel
            (*queue).enqueueNDRangeKernel(calcOutputLayerErrorGradients, 
                cl::NDRange(offset), cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

            //Queue the kernels 
            for (unsigned int i = netSpec.size()-2; i != 0; --i)
            {
                offset += -netSpec[i];
                if (layers[i+1].numberOfNodes % 5 == 0 && layers[i].nodes[0].numberOfWeights % 5 == 0)
                {
                    (*queue).enqueueNDRangeKernel(calcLayerErrorGradientsUnrolled,
                        cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
                }
                else
                {
                    (*queue).enqueueNDRangeKernel(calcLayerErrorGradientsRolled,
                        cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
                }
            }
        }
    }
}

void FullyConnectedNeuralNet::trainFullyConnectedNeuralNet (
    vector<float*> &trainingData,
    vector<int*> &trainingLabel,
    cl::CommandQueue *queue,
    int trainingIterations)
{
    if (trainingLabel.size() != trainingData.size())
    {
        throw std::invalid_argument("training label and training data must be of the same size!");
    }
    unsigned int dataSetSize = trainingLabel.size();
    for (int c = 1; c != trainingIterations; ++c)
    {
        if (c%5 == 0)
        {
            calculateError(trainingData, trainingLabel, queue);
            writeFullyConnectedNeuralNetToFile(*queue);
        }
        else
            calcQuickError(trainingData, trainingLabel, queue);

        time_t start, end;
        time(&start);
        cout << "Training iteration " << c << "..." << endl;
        //Training once over all the data samples
        for (int i = 0; i != dataSetSize; ++i)
        {
            if (i%10000 == 0)
                cout << "    Training over the " << i << "th input" << endl;
            float* featureVector = getRandomDistortion(trainingData[i]);
            int* targetVector = trainingLabel[i];

            //First compute output
            computeOutput(featureVector, queue);

            //Garbage collect
            delete featureVector;


            //Calculate the appropriate offset
            int offset = 0;
            for (unsigned int i = 1; i != netSpec.size()-1 ; ++i)
                offset += netSpec[i];
            
            //Queue the writebuffer
            (*queue).enqueueWriteBuffer(targetBuffer,CL_TRUE,0,sizeOfTarget,&targetVector[0]);

            //Queue the kernel
            (*queue).enqueueNDRangeKernel(calcOutputLayerErrorGradients, 
                cl::NDRange(offset), cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

            //Queue the kernels 
            for (unsigned int i = netSpec.size()-2; i != 0; --i)
            {
                offset += -netSpec[i];
                if (layers[i+1].numberOfNodes % 5 == 0 && layers[i].nodes[0].numberOfWeights % 5 == 0)
                {
                    (*queue).enqueueNDRangeKernel(calcLayerErrorGradientsUnrolled,
                        cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
                }
                else
                {
                    (*queue).enqueueNDRangeKernel(calcLayerErrorGradientsRolled,
                        cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
                }
            }
        }
        time(&end);
        cout << "    Completed in " << difftime(end ,start) << "seconds" << endl;
    }
}
void FullyConnectedNeuralNet::trainFullyConnectedPortion(
    int* targetVector,
    cl::CommandQueue *queue)
{
    //Calculate the appropriate offset
    int offset = 0;
    for (unsigned int i = 1; i != netSpec.size()-1 ; ++i)
        offset += netSpec[i];
    
    //Queue the writebuffer
    (*queue).enqueueWriteBuffer(targetBuffer,CL_TRUE,0,sizeOfTarget,targetVector);

    //Queue the kernel
    (*queue).enqueueNDRangeKernel(calcOutputLayerErrorGradients, 
        cl::NDRange(offset), cl::NDRange(netSpec[lastLayerIndex]), cl::NullRange);

    //Queue the kernels 
    for (unsigned int i = netSpec.size()-2; i != 0; --i)
    {
        offset += -netSpec[i];
        if (layers[i+1].numberOfNodes % 5 == 0 && layers[i].nodes[0].numberOfWeights % 5 == 0)
        {
            (*queue).enqueueNDRangeKernel(calcLayerErrorGradientsUnrolled,
                cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        }
        else
        {
            (*queue).enqueueNDRangeKernel(calcLayerErrorGradientsRolled,
                cl::NDRange(offset), cl::NDRange(netSpec[i]), cl::NullRange);
        }
    }
}