#include "TensorEngine.hpp"
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
#include <iostream>
#include <fstream>
#include <logger.h>
#include </usr/src/tensorrt/samples/common/logger.h>
bool TensorEngine::fileExists(string FILENAME)
{
    ifstream f(FILENAME.c_str());
    return f.good();
}
string TensorEngine::serializeEngineName(const Configurations& config) 
{
    string name = "trt.engine";

    if(config.FP16)
    {
        name += ".fp16";
    }
    else
    {
        name += ".fp32";
    }
    name += "." + to_string(config.maxBatchSize); + ".";
    for (int i = 0; i < m_config.optBatchSize.size(); ++i)
    {
        name += to_string(m_config.optBatchSize[i]);
        if(i != m_config.optBatchSize.size() - 1)
        {
            name += "_";
        } 
    }
     
    return name;

}
TensorEngine::TensorEngine(const Configurations &config) : m_config(config) {}
TensorEngine::~TensorEngine()
{

}

bool TensorEngine::build(string ONNXFILENAME)
{
    
    //Check if engine file already exists
    m_engineName = serializeEngineName(m_config);
    if(fileExists(m_engineName))
    {
        cout << "Engine already exists" << endl;
        return true;
    }
    //no engine found create new
    cout << "Creating engine..." << endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if(!builder) 
    {
        cout << "Builder creation failed" << endl;
        return false;
    }
    //Set maximum batchsize
    builder->setMaxBatchSize(m_config.maxBatchSize);

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network= unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network)
    {
        cout << "Network creation failed" << endl;
        return false;
    }

    //Create the parser
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if(!parser)
    {
        cout << "Parser creation failed" << endl;
        return false;
    }

    //First read from memory then passe the parser to a buffer. 
    ifstream file(ONNXFILENAME, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<char> buffer(size);
    if(!file.read(buffer.data(), size))
    {
        throw runtime_error("Was not able to read the engine file");
    }
    auto parsed = parser->parse(buffer.data(), buffer.size());
    //auto parsed = parser->parseFromFile(ONNXFILE_2, (int)ILogger::Severity::kWARNING);
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        cout << parser->getError(i)->desc() << endl;
    }
    if(!parsed)
    {
        cout << "Parsing failed" << endl;
        return false;
    }
    //Getting input, output, height, weight and channels
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    m_inputName = input->getName();
    m_outputName = output->getName();
    printf("%s : %s\n", m_inputName, m_outputName);
    m_inputDims = input->getDimensions();
    int32_t inputChannel = m_inputDims.d[1];
    int32_t inputHeight = m_inputDims.d[2];
    int32_t inputWidth = m_inputDims.d[3];

    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config)
    {
        cout << "Was not able to build the config" << endl;
        return false;
    }

    //Specifying the optimization profile
    IOptimizationProfile *defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(m_inputName, OptProfileSelector::kMIN, Dims4(1, inputChannel, inputHeight, inputWidth));
    defaultProfile->setDimensions(m_inputName, OptProfileSelector::kOPT, Dims4(1, inputChannel, inputHeight, inputWidth));
    defaultProfile->setDimensions(m_inputName, OptProfileSelector::kMAX, Dims4(1, inputChannel, inputHeight, inputWidth));
    config->addOptimizationProfile(defaultProfile);

    config->setMaxWorkspaceSize(m_config.maxWorkspaceSize);
    //TODO: Make different optimization profiles prossible
    
    //Set GPU fallback

    //Making a cuda stream for the profile
    auto cudaStream = samplesCommon::makeCudaStream();
    if(!cudaStream)
    {
        cout << "Could not create cudaStream." << endl;
        return false;
    }
    //Setting the profile stream
    // config->setProfileStream(*cudaStream);
    // config->setFlag(BuilderFlag::kGPU_FALLBACK);
    // config->setFlag(BuilderFlag::kFP16);
    // config->setDefaultDeviceType(DeviceType::kDLA);
    // config->setDLACore(1);
    //Creating the serialized model of the engine
    unique_ptr<IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    if(!serializedModel)
    {
        cout << "Could not build serialized model" << endl;
        return false;
    }

    /*ADD DLA 
    */
   
    
    //write the engine to disk
    ofstream outfile(m_engineName, ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    cout << "The engine has been built and saved to disk successfully" << endl;

    return true;

}
bool TensorEngine::loadNetwork()
{
    ifstream file(m_engineName, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<char> buffer(size);
    if(!file.read(buffer.data(), size))
    {
        cout << "Could not read the network from disk" << endl;
        return false;
    }
    //Creates a runtime object for running inference
    unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};

    //TODO: Set device index

    //Let's create the engine
    m_engine = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if(!m_engine)
    {
        cout << "Creating the cuda engine failed" << endl;
        return false;
    }

    printf("\n\nBinding name NAME: %s\n\n", m_engine->getBindingName(1));
    printf("Binding index output: %d\n\n", m_engine->getBindingIndex("Plus214_Output_0"));
    printf("m_inputname == %s\n", m_inputName);
    m_inputName = m_engine->getBindingName(0);
    m_outputName = m_engine->getBindingName(1);
    
    m_inputDims = m_engine->getBindingDimensions(0);
    m_oututDims = m_engine->getBindingDimensions(1);

    printf("m_inputname == %s\n", m_inputName);
    printf("m_outputname == %s\n", m_outputName);
    
    m_context = shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if(!m_context)
    {
        cout << "Creating the execution context failed" << endl;
        return false;
    }
    
    return true;
}
//Location of the pgm files /usr/src/tensorrt/data/mnist
bool TensorEngine::inference()
{

    samplesCommon::BufferManager buffers(m_engine);
    if(!processInput(buffers))
    {
        cout << "Could not process the input";
        return false;
    }

    buffers.copyInputToDevice();

    bool succ = m_context->executeV2(buffers.getDeviceBindings().data());
    if(!succ)
    {
        cout << "Running inference failed";
        return false;
    }
    buffers.copyOutputToHost();

    if(!verifyOutput(buffers))
    {
        cout << "Sap doc" << endl;
        return false;
    }


    return true;
}
//Taken from the sample
bool TensorEngine::processInput(const samplesCommon::BufferManager& buffer)
{

    const int inputH = m_inputDims.d[2];
    const int inputW = m_inputDims.d[3];

    srand(unsigned(time(nullptr)));
    vector<uint8_t> filedata(inputH * inputW);
    m_location.push_back("/usr/src/tensorrt/data/mnist");
    m_pgmNumber = rand() % 10;
    
    readPGMFile(locateFile(to_string(m_pgmNumber) + ".pgm", m_location), filedata.data(), inputH, inputW);
    sample::gLogInfo << "Input:" << std::endl;
    for (size_t i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[filedata[i] / 26]) << (((i+1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;
    float *hostDataBuffer = static_cast<float*>(buffer.getHostBuffer(m_inputName));
    for (size_t i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(filedata[i] / 255.0);
    }
    
    
    return true;    
}
//Taken from the sample
bool TensorEngine::verifyOutput(const samplesCommon::BufferManager& buffer)
{
    float* output = static_cast<float*>(buffer.getHostBuffer(m_outputName));
    float val{0.0f};
    int idx{0};

    //calculate Softmax
    float sum{0.0f};
    for (size_t i = 0; i < m_oututDims.d[1]; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }
    printf("Output size%d\n\n", m_oututDims.d[1]);
    sample::gLogInfo << "Output: " << endl;
    for (size_t i = 0; i < m_oututDims.d[1]; i++)
    {
        output[i] /= sum;
        val = max(val, output[i]);
        if(val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << "Prob" << " " << fixed << setw(5) << setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << string(int(floor(output[i] * 10 + 0.5f)), '*')
                         << endl;   
    }
    sample::gLogInfo << std::endl;

    return idx == m_pgmNumber && val > 0.9f;

}



