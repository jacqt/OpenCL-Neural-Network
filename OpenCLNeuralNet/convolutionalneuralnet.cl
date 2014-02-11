#define MAXFILTERDIM 15
#define MAXFILTERS 15

typedef struct Filter
{
    int filterDim;
    int filterNumber;
    float weights[MAXFILTERDIM]; 
    float errorGradientSum;//sum of the error gradients
} Filter;

typedef struct ConvolutionalLayer
{
    int numberOfFilters;
    Filter filters[MAXFILTERS];
} ConvolutionalLayer;

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


//Rolled kernel that computes the result of convolving the filters 
kernel void computeConvolveResult(global ConvolutionalLayer* restrict layers, global float* convolvingResult, constant double* inputs)
{
    const int row = get_global_id(0);
    const int column = get_global_id(1);
}