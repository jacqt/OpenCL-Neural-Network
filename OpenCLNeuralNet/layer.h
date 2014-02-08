#define MAXSIZE 150

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