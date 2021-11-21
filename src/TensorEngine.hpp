#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <NvInfer.h>

#include "Logger.hpp"
#include <buffers.h>
#include <memory>

using namespace std;

#define ONNXFILE_1 "/home/joakimfj/Documents/TensorRt/mnist2/model.onnx"
#define ONNXFILE_2 "yolov3-10.onnx"
#define PGM_LOC "/usr/src/tensorrt/data/mnist"

struct Configurations {
    //Using 16 point floats for inference
    bool FP16 = false;
    //Batch size for optimization
    vector<int32_t> optBatchSize;
    // Maximum allowed batch size
    int32_t maxBatchSize = 16;
    //Max GPU memory allowed for the model.
    int maxWorkspaceSize = 16000000;
    //GPU device index number, might be useful for more Tegras in the future
    int deviceIndex = 0;

};

class TensorEngine 
{
    public:
        //constructor
        TensorEngine(const Configurations& config);
        //destructor
        ~TensorEngine();
        //Builds the network from the onnx file
        bool build(string ONNXFILENAME);
        //Loads and prepares the network for inference
        bool loadNetwork();
        //Runs inference on the network
        bool inference();
        //Make engine name

    private:

        bool fileExists(string FILENAME);
        string serializeEngineName(const Configurations& config);
        bool processInput(const samplesCommon::BufferManager& buffer);

        shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        shared_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
        const char* m_inputName;
        const char* m_outputName;

        Logger m_logger;

        samplesCommon::ManagedBuffer m_inputBuffer;
        samplesCommon::ManagedBuffer m_outputBuffer;

        Dims m_inputDims;
        Dims m_oututDims;

        int batchSize = 0;
        
        string m_engineName;

        //Location of the pgm files
        vector<string> m_location;

        cudaStream_t m_cudaStream = nullptr;

        const Configurations& m_config;

};