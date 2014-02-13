#ifndef NEURALNET_H
#define NEURALNET_H
#include "include.h"
#include "fullyconnectedneuralnet.h"
#include "convolutionalneuralent.h"

//Reads a file
std::string getFileContents(const char* fileName);

//Gets some test data
vector<std::tuple<float*, int*> > getTestData ();

//Class combining the CNN and fully connected neural networks into one structure
class NeuralNetwork
{
public:
    ConvolutionalNetworkPortion convolutionalPortion;
    FullyConnectedNeuralNet fullyConnectedPortion;

    //Writes the network to a file
    void writeNeuralNetworkToFile();

    //Loads a network from file assuming the instance has not already initalized a network
    void loadNeuralNetworkFromFile();

    //Creates and intializes everything that is needed for the neural network
    void createNeuralNetwork(
        cl::Context &context,
        cl::Program &fullyConnectedProgram,
        cl::Program &convolutionalProgram,
        vector<cl_int> &fullyConnectedNetSpec,
        unsigned int newFilterDim, 
        unsigned int newFilterNumberSize,
        unsigned int newInputDim);

    //Computes the output of the neural net given a feature vector
    void computeOutput(cl_float* inputs, cl::CommandQueue* queue);

    //Computes the error rate of the neural network on some data
    void computeError(
        vector<std::tuple<float*, int*> >* testData,
        cl::CommandQueue* queue);
    
    //Trains the neural net given a vector of tuples containing the feature vector
    //  and target vector
    void trainNeuralNet(
        vector<std::tuple<float*, int*> >* trainingData,
        cl::CommandQueue* queue,
        int trainingIterations);
};

#endif