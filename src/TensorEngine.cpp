#include "TensorEngine.hpp"
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
//#include <iostream>
#include <fstream>
bool TensorEngine::fileExists(string FILENAME)
{
    ifstream f(FILENAME.c_str());
    return f.good();
}

TensorEngine::TensorEngine(const Configurations &config) : m_config(config) {}

bool TensorEngine::build(string ONNXFILENAME)
{
    
    //Check if engine file already exists
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
    if(!parsed)
    {
        cout << "Parsing failed" << endl;
        return false;
    }
    //Getting input, output, height, weight and channels
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputChannel = inputDims.d[1];
    int32_t inputHeight = inputDims.d[2];
    int32_t inputWidth = inputDims.d[3];

    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config)
    {
        cout << "Was not able to build the config" << endl;
        return false;
    }

    //Specifying the optimization profile
    IOptimizationProfile *defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputChannel, inputHeight, inputWidth));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, inputChannel, inputHeight, inputWidth));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_config.maxBatchSize, inputChannel, inputHeight, inputWidth));
    config->addOptimizationProfile(defaultProfile);

    config->setMaxWorkspaceSize(m_config.maxWorkspaceSize);
    //TODO: Make different optimization profiles prossible
    
    //Making a cuda stream for the profile
    auto cudaStream = samplesCommon::makeCudaStream();
    if(!cudaStream)
    {
        cout << "Could not create cudaStream." << endl;
        return false;
    }
    //Setting the profile stream
    config->setProfileStream(*cudaStream);

    //Creating the serialized model of the engine
    unique_ptr<IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    if(!serializedModel)
    {
        cout << "Could not build serialized model" << endl;
        return false;
    }

    //Todo write the engine to disk
    ofstream outfile(m_engineName, ofstream::binary);

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
    m_engine = unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if(!m_engine)
    {
        cout << "Creating the cuda engine failed" << endl;
        return false;
    }

    m_context = unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if(!m_context)
    {
        cout << "Creating the execution context failed" << endl;
        return false;
    }

    //We loaded the network successfully
    return true;
}
bool TensorEngine::inference()
{
    auto dims = m_engine->getBindingDimensions(0);
    auto outputL = m_engine->getBindingDimensions(1).d[1];

}
