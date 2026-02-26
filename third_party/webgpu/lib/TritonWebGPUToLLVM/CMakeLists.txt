add_triton_library(TritonWebGPUToLLVM
    TargetInfo.cpp
    TritonGPUToLLVM.cpp

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUToLLVM
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRUBToLLVM
)
