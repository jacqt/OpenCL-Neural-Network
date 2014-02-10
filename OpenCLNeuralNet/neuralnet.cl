#define MAXSIZE 200
#define N 0.02

typedef struct Node
{
    int numberOfWeights;
    float weights[MAXSIZE];
    float output;
    float input;
    float delta;
} Node;

typedef struct Layer
{
    int numberOfNodes;
    Node nodes[MAXSIZE];
} Layer;

//Returns the result of passing n through the sigmoid function; f(x) = 1/(1+exp(-x))
float inline sigmoid(float n)
{

    //To deal with overflow rounding errors and the such
    if (n < -100)
        return 0;
    if (n > 100)
        return 1;
    return 1/(1 + exp(-n));
}

//Returns the result of passing n through the deriviative of the sigmoid function
float sigmoidDerivative(float n)
{
    float k = sigmoid(n);
    return k*(1-k);
}

//Used to find the (row,nodeNumber) pair that corresponds to the n'th input/delta node
void inline getPosition(int n, constant int* restrict netSpec, int* restrict row, int* restrict nodeNumber)
{
    for (unsigned int i = 1; ;++i)//Termination is determined by the break statement
    {
        int k = netSpec[i];
        bool comparison = k <= n;
        if (comparison)
            n += -k;
        else
        {
            *row = i;
            *nodeNumber = n;
            break;
        }
    }
}

//Rolled kernel that computes a layer's output
kernel void computeLayerOutput_Rolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0); //There will be an offset depending on the layer we are operating on

    int layer, nodeNumber, numberOfWeights;
    float t;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;

    t = 0;
    for (unsigned int j = 0; j != numberOfWeights; ++j)
        t += layers[layer].nodes[nodeNumber].weights[j] * layers[layer-1].nodes[j].output;

    layers[layer].nodes[nodeNumber].input = t;
    layers[layer].nodes[nodeNumber].output = sigmoid(t);
}

//Unrolled kernel that computes a layer's output
kernel void computeLayerOutput_Unrolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0); //There will be an offset depending on the layer we are operating on

    int layer, nodeNumber, numberOfWeights;
    float t;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;

    t = 0;
    for (unsigned int j = 0; j != numberOfWeights; j+=5)
    {
        t += layers[layer].nodes[nodeNumber].weights[j] * layers[layer-1].nodes[j].output;
        t += layers[layer].nodes[nodeNumber].weights[j+1] * layers[layer-1].nodes[j+1].output;
        t += layers[layer].nodes[nodeNumber].weights[j+2] * layers[layer-1].nodes[j+2].output;
        t += layers[layer].nodes[nodeNumber].weights[j+3] * layers[layer-1].nodes[j+3].output;
        t += layers[layer].nodes[nodeNumber].weights[j+4] * layers[layer-1].nodes[j+4].output;
    }

    layers[layer].nodes[nodeNumber].input = t;
    layers[layer].nodes[nodeNumber].output = sigmoid(t);
}

kernel void setInputs(global Layer* layers, constant float* inputs)
{
    const int i = get_global_id(0);
    layers[0].nodes[i].output = inputs[i];
}

//Implements the _online_ backwards propgatiaon algorithm that computes the delta, then uses that value to compute the weights
//And then apply the weights immediately. This function is for all other non-input and non-output nodes
kernel void computeDelta_ApplyWeightChange_Rolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0);

    //Useful variables
    int layer, nodeNumber, numberOfWeights, numberOfNodes_NextLayer;
    float delta, input, weightChange;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    input = layers[layer].nodes[nodeNumber].input;
    numberOfNodes_NextLayer = layers[layer+1].numberOfNodes;
       
    //Compute delta
    delta = 0;
    for (uint j = 0; j != numberOfNodes_NextLayer; ++j)
        delta += layers[layer+1].nodes[j].delta * layers[layer+1].nodes[j].weights[nodeNumber];
    delta *= sigmoidDerivative(input);

    //Use the delta to compute and apply the weight change
    float NDelta = N*delta;
    for (uint j = 0; j != numberOfWeights; ++j)
        layers[layer].nodes[nodeNumber].weights[j] += NDelta*layers[layer-1].nodes[j].output;
    layers[layer].nodes[nodeNumber].delta = delta;
}

//Unrolled version of the above
kernel void computeDelta_ApplyWeightChange_Unrolled(global Layer* restrict layers, constant int* restrict netSpec)
{
    const int n = get_global_size(0);
    const int i = get_global_id(0);

    //Useful variables
    int layer, nodeNumber, numberOfWeights, numberOfNodes_NextLayer;
    float delta, input, weightChange;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    input = layers[layer].nodes[nodeNumber].input;
    numberOfNodes_NextLayer = layers[layer+1].numberOfNodes;
       
    //Compute delta
    delta = 0;
    for (uint j = 0; j != numberOfNodes_NextLayer; j+=5)
    {
        delta += layers[layer+1].nodes[j].delta * layers[layer+1].nodes[j].weights[nodeNumber];
        delta += layers[layer+1].nodes[j+1].delta * layers[layer+1].nodes[j+1].weights[nodeNumber];
        delta += layers[layer+1].nodes[j+2].delta * layers[layer+1].nodes[j+2].weights[nodeNumber];
        delta += layers[layer+1].nodes[j+3].delta * layers[layer+1].nodes[j+3].weights[nodeNumber];
        delta += layers[layer+1].nodes[j+4].delta * layers[layer+1].nodes[j+4].weights[nodeNumber];
    }
    delta *= sigmoidDerivative(input);

    //Use the delta to compute and apply the weight change
    float NDelta = N*delta;
    for (uint j = 0; j != numberOfWeights; j+=5)
    {
        //Partial unroll
        layers[layer].nodes[nodeNumber].weights[j] += NDelta * layers[layer-1].nodes[j].output;
        layers[layer].nodes[nodeNumber].weights[j+1] += NDelta * layers[layer-1].nodes[j+1].output;
        layers[layer].nodes[nodeNumber].weights[j+2] += NDelta * layers[layer-1].nodes[j+2].output;
        layers[layer].nodes[nodeNumber].weights[j+3] += NDelta * layers[layer-1].nodes[j+3].output;
        layers[layer].nodes[nodeNumber].weights[j+4] += NDelta * layers[layer-1].nodes[j+4].output;
    }
    layers[layer].nodes[nodeNumber].delta = delta;
}

//Implements the _online_ backwards propgatiaon algorithm that computes the delta, then uses that value to compute the weights
//And then apply the weights immediately. This function is for the output nodes
kernel void computeDelta_ApplyWeightChange_OutputNode(global Layer* restrict layers, constant int* restrict netSpec, constant int* restrict targets)
{
    const int n = get_global_size(0); //Also the size of the target array
    const int i = get_global_id(0); //Offset tells us which layer we are operating on

    //Useful variables
    int layer, nodeNumber, numberOfWeights;
    float delta, output, input, weightChange;
    getPosition(i, netSpec, &layer, &nodeNumber);
    numberOfWeights = layers[layer].nodes[nodeNumber].numberOfWeights;
    output = layers[layer].nodes[nodeNumber].output;
    input = layers[layer].nodes[nodeNumber].input;
    
    //Compute delta
    delta = (targets[nodeNumber] - output)*sigmoidDerivative(input);
    
    //Use the delta to compute and apply the weight change
    float NDelta = N*delta;
    for (uint j = 0; j != numberOfWeights; j+=5)
    {
        //Partial unroll
        layers[layer].nodes[nodeNumber].weights[j] += NDelta * layers[layer-1].nodes[j].output;
        layers[layer].nodes[nodeNumber].weights[j+1] += NDelta * layers[layer-1].nodes[j+1].output;
        layers[layer].nodes[nodeNumber].weights[j+2] += NDelta * layers[layer-1].nodes[j+2].output;
        layers[layer].nodes[nodeNumber].weights[j+3] += NDelta * layers[layer-1].nodes[j+3].output;
        layers[layer].nodes[nodeNumber].weights[j+4] += NDelta * layers[layer-1].nodes[j+4].output;
    }
    layers[layer].nodes[nodeNumber].delta = delta;
}
