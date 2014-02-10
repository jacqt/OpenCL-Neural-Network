#define MAXSIZE 200
//Define a maxsize because pointers are not allowed to be passed to the kernel
//Note that this number _must_ be the same as the MAXSIZE defined under neuralnet.cl

//We define a layer struct along with node struct because we want to pass these structs to the
//kernel running on the GPU which does not support C++ types
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

float getRandomFloat(float lowerbound, float upperbound);

Layer layer_newInputLayer(int numberOfNodes);

Layer layer_new(int numberOfNodes, int numberOfWeights);

void layer_setLayerOutputs(Layer *layer, float *outputs);

int layer_size(Layer *layer);

int node_size(Node *node);