#ifndef NEURALNET_H
#define NEURALNET_H
#include "include.h"
#include "fullyconnectedneuralnet.h"
#include "convolutionalneuralent.h"

//Reads a file
std::string getFileContents(const char* fileName);

//Reads a .cl file and creates a program from the .cl file
//Throws exceptions if there are build errors and logs the errors
cl::Program createProgram(cl::Context &context, std::string fname, const std::string params = "") ;

//Gets some test data
vector<std::tuple<float*, int*> > getTestData ();

//Class combining the CNN and fully connected neural networks into one structure
class NeuralNetwork
{
public:
    ConvolutionalNetworkPortion* convolutionalPortion;
    FullyConnectedNeuralNet* fullyConnectedPortion;
    int writeFileCounter;

    ~NeuralNetwork();

    //Loads a network from file assuming the instance has not already initalized a network
    void loadNeuralNetworkFromFile(
        std::string netFileName,
        cl::Context &context,
        cl::Program &fullyConnectedProgram,
        cl::Program &convolutionalProgram);

    //Writes the network to a file
    void writeNeuralNetworkToFile(cl::CommandQueue &queue);


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
    void calculateError(
        vector<std::tuple<float*, int*> >* testData,
        cl::CommandQueue* queue);

    //Overloaded expression
    void calculateError(
        vector<float*> &testData,
        vector<int*>  &testLabel,
        cl::CommandQueue* queue);
    
    //Trains the neural net given a vector of tuples containing the feature vector
    //  and target vector
    void trainNeuralNet(
        vector<std::tuple<float*, int*> >* trainingData,
        cl::CommandQueue* queue,
        int trainingIterations);

    //Overloaded
    void trainNeuralNet(
        vector<float*> &trainingData,
        vector<int*>  &trainingLabels,
        cl::CommandQueue* queue,
        int trainingIterations);
private:
    float prevOutput[100];
};

#endif