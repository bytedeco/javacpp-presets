/*
 * JavaCPP parse shim for torch::inductor::AOTIModelContainerRunnerXpu.
 *
 * The real header (torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h)
 * pulls c10/xpu/XPUStream.h → SYCL types, which are not available when parsing
 * the common torch preset (macOS / CPU). This shim re-declares the class surface
 * that we can bind without SYCL:
 *   - constructors / destructor
 * and deliberately omits:
 *   - run_impl (already skipped on the base class; needs AtenTensorHandle)
 *   - run_with_xpu_stream (needs at::xpu::XPUStream / SYCL)
 *
 * JNI still links against the real libtorch symbol when present; on platforms
 * without XPU the class is parse-only (dynamic_lookup).
 */
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

// HERE we use C10_EXPORT because libtorch_python needs this Symbol be exported.
// And `TORCH_API` and `TORCH_XPU_API` do not export the symbol in Windows build.
class C10_EXPORT AOTIModelContainerRunnerXpu : public AOTIModelContainerRunner {
 public:
  // @param device_str: xpu device string, e.g. "xpu", "xpu:0"
  AOTIModelContainerRunnerXpu(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "xpu",
      const std::string& kernel_bin_dir = "",
      const bool run_single_threaded = false);

  ~AOTIModelContainerRunnerXpu() override;
};

} // namespace torch::inductor
