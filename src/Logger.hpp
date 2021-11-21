#include <NvInferRuntimeCommon.h>
#include <iostream>
#include <logger.h>
#include <logging.h>
//This class extends the tensorRt logger. (From the NVIDIA tensorrt developers guide)
class Logger : public nvinfer1::ILogger 
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << msg << std::endl;
        }
    }
};
