#include "neuralnet.h"

using std::vector;
using std::cout;
using std::cin;
using std::endl;


cl::Program createProgram(cl::Context &context, std::string fname, const std::string params = "") 
{
    cl::Program::Sources sources;
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES> ();
    
    std::string sourceCode = getFileContents(fname.c_str());
    sources.insert(sources.end(), std::make_pair(sourceCode.c_str(),
        sourceCode.length()));
    cl::Program* program = new cl::Program(context,sources);

    try
    {
        (*program).build(devices, params.c_str());
    }
    catch (cl::Error e)
    {
        cout << "Compilation build error log: " << endl <<
            (*program).getBuildInfo <CL_PROGRAM_BUILD_LOG> (devices [0]) << endl;
    }

    return (*program);
}

int main()
{
    //Get the platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //Create a context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platforms.begin())(), 0};
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    //Create and build the program
    cl::Program program;
    program = createProgram(context, "neuralnet.cl");

    //Create the neural network as a vector of layers
    cl_int netSpecArray[] = {2, 50, 30, 1};//We include the input layer in the netSpec, which means that we will have to perform some offsets
    vector<cl_int> netSpec (netSpecArray, netSpecArray + sizeof(netSpecArray)/sizeof(int)); 
    //vector<cl_int> netSpec;
    vector<Layer> layers;
    createNeuralNet(netSpec,layers);
    //loadNeuralNetFromFile("neuralNet.net",netSpec,layers);

    size_t sizeOfNet = getSizeOfNet(layers);

    //Create memory buffers
    cl::Buffer netSpecBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_int)*netSpec.size(), &netSpec[0]);

    cl::Buffer layersBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeOfNet, &layers[0]);

    //Create the command queue from the first device in context
    cl::CommandQueue queue;
    queue = cl::CommandQueue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

    //Test it
    vector<std::tuple<float*, int*> > testData = getTestData();
    calculateError(&context, &testData, &netSpec, &layers, &program, &netSpecBuffer, &layersBuffer, &queue);

    //OKAY NOW TRAIN IT!
    int trainingIterations = 100;
    trainNeuralNet(&context, &testData, &netSpec, &layers, &program, &netSpecBuffer, &layersBuffer, &queue, trainingIterations);

    cout << "Finished!" << endl;

    int wait;
    cin >> wait;
    return EXIT_SUCCESS;
}