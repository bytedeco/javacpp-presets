#include "c10/util/ArrayRef.h"

// Included by
// ATen/cudnn/Types.h
// ATen/cudnn/Descriptors.h
// torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h
// torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp

#include "c10/core/impl/GPUTrace.h"
//#include "c10/cuda/impl/cuda_cmake_macros.h"
#include "c10/cuda/CUDAMacros.h"
#include "c10/cuda/CUDADeviceAssertionHost.h"
#include "c10/cuda/CUDAMiscFunctions.h",
#include "c10/cuda/CUDAException.h",
#include "c10/cuda/CUDAFunctions.h",
#include "ATen/cuda/CUDAContextLight.h"
#include "c10/cuda/CUDAStream.h"
#include "ATen/cuda/Exceptions.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cudnn/cudnn-wrapper.h"
#include "ATen/cuda/ATenCUDAGeneral.h"
#include "ATen/cudnn/Handle.h"
#include "ATen/cudnn/Utils.h"
#include "torch/csrc/distributed/c10d/NCCLUtils.hpp"
#include "c10/cuda/CUDAGraphsC10Utils.h"
#include "c10/cuda/CUDACachingAllocator.h",
#include "c10/cuda/impl/CUDAGuardImpl.h"
#include "c10/cuda/CUDAGuard.h"
#include "ATen/cuda/CUDAEvent.h"
#include "torch/csrc/distributed/c10d/intra_node_comm.hpp"
//#include "ATen/DynamicLibrary.h" // Useless ?
#include "ATen/cudnn/Descriptors.h"
#include "torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h"
#include "torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp"