#include <stdio.h>
#include "TensorEngine.hpp"
int main()
{
    Configurations config;
    config.optBatchSize = {2, 4, 8};

    TensorEngine engine(config);

    bool succ = engine.build(ONNXFILE_1);
    if(!succ)
    {
        throw runtime_error("Could not built TRT engine");
    }
    succ = engine.loadNetwork();
    if(!succ)
    {
        throw runtime_error("Could not load network");
    }
    // succ = engine.inference();
    // if(!succ)
    // {
    //     throw runtime_error("Could not run inference");
    // }
    printf("End of code\n");
    return 0;
}