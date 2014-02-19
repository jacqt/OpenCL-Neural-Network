#include "MNIST.h"
//#define FILENAME "CNN-27.net" //Name of the CNN to load
#define DATASUBSET 1000

#include <istream>
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readMNIST(vector<float*> &inputs, vector<int*> &targets, int &inputSize)
{
    //Read the targets
    std::ifstream targetFile("data/train-labels.idx1-ubyte", std::ios::binary);
    if (targetFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        targetFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        targetFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        for(int i=0;i != number_of_images;++i)
        {
            unsigned char temp=0;
            targetFile.read((char*)&temp,sizeof(temp));
            int* target = new int[10];
            std::fill_n(target,10,0);
            target[(int)temp] = 1;
            targets.push_back(target);

            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
    std::ifstream trainingFeatureFile ("data/train-images.idx3-ubyte", std::ios::binary);
    if (trainingFeatureFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        trainingFeatureFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        trainingFeatureFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        trainingFeatureFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        trainingFeatureFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        inputSize = n_rows * n_cols;
        for(int i=0;i != number_of_images; ++i)
        {
            float* inputData = new float[n_rows*n_cols];
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    trainingFeatureFile.read((char*)&temp,sizeof(temp));
                    inputData[r*n_cols + c] = ((float)temp)/255.0;
                }
            }
            inputs.push_back(inputData);
            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
}

void readMNISTTest(vector<float*> &inputs, vector<int*> &targets, int &inputSize)
{
    //Read the targets
    std::ifstream targetFile("data/t10k-labels.idx1-ubyte", std::ios::binary);
    if (targetFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        targetFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        targetFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);


        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            targetFile.read((char*)&temp,sizeof(temp));
            int* target = new int[10];
            std::fill_n(target,10,0);
            target[(int)temp] = 1;
            targets.push_back(target);

            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
    std::ifstream trainingFeatureFile ("data/t10k-images.idx3-ubyte", std::ios::binary);
    if (trainingFeatureFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        trainingFeatureFile.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        trainingFeatureFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        trainingFeatureFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        trainingFeatureFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        inputSize = n_rows * n_cols;
        for(int i=0;i<number_of_images;++i)
        {
            float* inputData = new float[n_rows*n_cols];
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    trainingFeatureFile.read((char*)&temp,sizeof(temp));
                    inputData[r*n_cols + c] = ((float)temp)/255.0;
                }
            }
            inputs.push_back(inputData);
            if (i%10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
}

//Prints out an image matrix form
void printInput(float* inputs)
{
    cout << "BELOW IS AN IMAGE" << endl;
    int c = 0;
    for (int i = 0; i != 28; ++i)
    {
        cout << "    " << endl;
        for (int j = 0; j != 28; ++j)
        {
            if (inputs[c] > 0)
                cout << 1;
            else
                cout << 0;
            ++c;
        }
        cout << endl;
    }
}

void trainMNISTFullyConnectedNN()
{

    vector<float*> inputs;
    vector<int*> targets;
    int inputSize;
    cout << "Reading MNIST data set" << endl;
    readMNIST(inputs,targets,inputSize);
    cout << "Finished reading. Now creating neural net" << endl;
    for (int i = 0; i != 10; ++i)
        cout << targets[0][i];
    cout << endl;

    //Get the platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //Create a context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platforms.begin())(), 0};
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    //Create and build the program
    cl::Program fullyConnectedNeuralNetProgram;
    fullyConnectedNeuralNetProgram = createProgram(context, "fullyconnectedneuralnet.cl");

    //Create the command queue from the first device in context
    cl::CommandQueue queue;
    queue = cl::CommandQueue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

    //Create the neural network as a vector of layers
    //We include the input layer in the netSpec, which means that we will have to perform some offsets
    cl_int netSpecArray[] = {inputSize, 1500, 1000, 500, 10};
    vector<cl_int> netSpec (netSpecArray, netSpecArray + sizeof(netSpecArray)/sizeof(int)); 

    //Need to allocate the net to the heap as neural nets can be extremely large and cause stack overflow errors
    FullyConnectedNeuralNet *myNet = new FullyConnectedNeuralNet; 
    (*myNet).createFullyConnectedNeuralNet(netSpec);
    (*myNet).createMemoryBuffersAndKernels(context, fullyConnectedNeuralNetProgram);

    //Code to load a neural network and train that one
    //(*myNet).loadFullyConnectedNeuralNetFromFile("NN-43.net");
    //(*myNet).createMemoryBuffersAndKernels(context, fullyConnectedNeuralNetProgram);

    //Ok, now train the neural net
    int trainingIterations = 150;
    (*myNet).trainFullyConnectedNeuralNet(inputs, targets, &queue, trainingIterations);

    (*myNet).calculateError(inputs, targets, &queue);
    (*myNet).writeFullyConnectedNeuralNetToFile(queue);

    //Now test on the test data
    vector<float*> testInputs;
    vector<int*> testTargets;
    cout << "Reading MNIST data set for testing..." << endl;
    readMNISTTest(testInputs,testTargets,inputSize);
    cout << "Finished reading. Now testing" << endl;
    (*myNet).calculateError(testInputs,testTargets,&queue);
}

void testMNISTFullyConnectedNN()
{
    vector<float*> testInputs;
    vector<int*> testTargets;
    int inputSize;
    cout << "Reading MNIST data set for testing..." << endl;
    readMNISTTest(testInputs,testTargets,inputSize);
    cout << "Finished reading. Now test" << endl;

    //Get the platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //Create a context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platforms.begin())(), 0};
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    //Create and build the program
    cl::Program fullyConnectedNeuralNetProgram;
    fullyConnectedNeuralNetProgram = createProgram(context, "fullyconnectedneuralnet.cl");

    //Create the command queue from the first device in context
    cl::CommandQueue queue;
    queue = cl::CommandQueue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

    //Iteratively loads a NN from a folder and tests to see how good it is
    for (int i = 0 ; i != 51; i += 1)//Iterates through neural nets and tests it against the testing set
    {
        std::ofstream netFile;
        std::ostringstream fileNameStream;
        fileNameStream << "NN-" << i << ".net";
        cout << "TESTING NET " << fileNameStream.str() << endl;
        FullyConnectedNeuralNet *myNet = new FullyConnectedNeuralNet; 
        (*myNet).loadFullyConnectedNeuralNetFromFile(fileNameStream.str());
        (*myNet).createMemoryBuffersAndKernels(context, fullyConnectedNeuralNetProgram);
        (*myNet).calculateError(testInputs,testTargets,&queue);
        delete myNet;
    }
}

void trainMNISTConvolutionalNN()
{
    vector<float*> inputs;
    vector<int*> targets;
    int inputSize;
    cout << "Reading MNIST data set" << endl;
    readMNIST(inputs,targets,inputSize);
    cout << "Finished reading. Now creating neural net" << endl;

    //Get the platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //Create a context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platforms.begin())(), 0};
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    //Create and build the program
    cl::Program fullyConnectedNeuralNetProgram;
    fullyConnectedNeuralNetProgram = createProgram(context, "fullyconnectedneuralnet.cl");
    cl::Program convolutionalNeuralNetProgram;
    convolutionalNeuralNetProgram = createProgram(context, "convolutionalneuralnet.cl");

    //Create the command queue from the first device in context
    cl::CommandQueue queue;
    queue = cl::CommandQueue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0]);

#ifndef FILENAME
    //Parameters of the CNN portion:
    int filterDim = 5;
    int filterNumberSize = 5;
    int inputDim = 28;
    int outputDim = (inputDim - filterDim + 1) / MAXPOOLDIM;

    //Parameters for the fully connected NN
    int numberOfInputsToFullyConnectedNN = outputDim * outputDim * filterNumberSize;
    cl_int netSpecArray[] = {numberOfInputsToFullyConnectedNN,70,10};
    cout << "NUMBER OF INPUTS TO FULLY CONNECTED PORTION " << numberOfInputsToFullyConnectedNN << endl;
    vector<cl_int> netSpec (netSpecArray, netSpecArray + sizeof(netSpecArray)/sizeof(int)); 

    //Need to allocate the net to the heap as neural nets can be extremely large and cause stack overflow errors
    NeuralNetwork *myNet = new NeuralNetwork; 
    (*myNet).createNeuralNetwork(
        context,
        fullyConnectedNeuralNetProgram,
        convolutionalNeuralNetProgram,
        netSpec, //fullyConnectedNetSpec
        filterDim, //newFilterDim
        filterNumberSize, //newFilterNumberSize
        inputDim); //newInputDim
    myNet->writeFileCounter = 0;
#else
    std::ofstream netFile;
    std::ostringstream fileNameStream;
    fileNameStream << FILENAME;
    NeuralNetwork* myNet = new NeuralNetwork; 
    (*myNet).loadNeuralNetworkFromFile(
        fileNameStream.str(),
        context,
        fullyConnectedNeuralNetProgram,
        convolutionalNeuralNetProgram);
    myNet->writeFileCounter = 30;
#endif
    cout << "Finished creating network. Now training it" << endl;

    //(*myNet).calculateError(inputs,targets,&queue);
    (*myNet).trainNeuralNet(inputs, targets, &queue, 100);
}

void testMNISTConvolutionalNN()
{
    vector<float*> testInputs;
    vector<int*> testTargets;
    int inputSize;
    cout << "Reading MNIST data set for testing..." << endl;
    readMNISTTest(testInputs,testTargets,inputSize);
    cout << "Finished reading. Now test" << endl;

    //Get the platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //Create a context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platforms.begin())(), 0};
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    //Create and build the program
    cl::Program fullyConnectedNeuralNetProgram;
    fullyConnectedNeuralNetProgram = createProgram(context, "fullyconnectedneuralnet.cl");
    cl::Program convolutionalNeuralNetProgram;
    convolutionalNeuralNetProgram = createProgram(context, "convolutionalneuralnet.cl");

    //Create the command queue from the first device in context
    cl::CommandQueue queue;
    queue = cl::CommandQueue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

#ifdef FILENAME
    NeuralNetwork *myNet = new NeuralNetwork; 
    (*myNet).loadNeuralNetworkFromFile(
        FILENAME,
        context,
        fullyConnectedNeuralNetProgram,
        convolutionalNeuralNetProgram);
    (*myNet).calculateError(testInputs,testTargets,&queue);
    delete myNet;
#else
    //Iteratively loads a NN from a folder and tests to see how good it is
    for (int i = 34 ; i < 50; ++i)
    {
        std::ofstream netFile;
        std::ostringstream fileNameStream;
        fileNameStream << "CNN-" << i << ".net";
        cout << "TESTING NET " << fileNameStream.str() << endl;
        NeuralNetwork *myNet = new NeuralNetwork; 
        (*myNet).loadNeuralNetworkFromFile(
            fileNameStream.str(),
            context,
            fullyConnectedNeuralNetProgram,
            convolutionalNeuralNetProgram);
        (*myNet).calculateError(testInputs,testTargets,&queue);
        delete myNet;
    }
#endif
}