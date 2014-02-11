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