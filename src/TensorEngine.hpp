#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <NvInfer.h>

#include "Logger.hpp"
#include <buffers.h>
#include <memory>

using namespace std;

struct Configurations {
    //Using 16 point floats for inference
    bool FP16 = false;
    //Batch size for optimization
    vector<int32_t> optBatchSize;
    // Maximum allowed batch size
    int32_t maxBatchSize = 16;
    //Max GPU memory allowed for the model.
    int maxWorkspaceSize = 4000000000;
    //GPU device index number, might be useful for more Tegras in the future
    int deviceIndex = 0;

};

class TensorEngine 
{
    public:
        TensorEngine(const Configurations& config);
        //Builds the network from the onnx file
        bool build(string ONNXFILENAME);
        //Loads and prepares the network for inference
        bool loadNetwork();
        //Runs inference on the network
        bool inference();

    private:

        bool fileExists(string FILENAME);

        unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

        Logger m_logger;

        samplesCommon::ManagedBuffer m_inputBuffer;
        samplesCommon::ManagedBuffer m_outputBuffer;

        int batchSize = 0;
        
        string m_engineName;

        const Configurations& m_config;

};