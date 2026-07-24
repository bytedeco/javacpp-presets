/*
 * JavaCPP parse shim for torch::inductor::AOTIModelContainerRunnerMps.
 *
 * The real header is guarded with `#if defined(__APPLE__)` which the JavaCPP
 * Parser may not define. This shim re-declares the class so it is always parsed
 * into org.bytedeco.pytorch.inductor. JNI still includes / links the real
 * implementation from libtorch on Apple platforms (dynamic_lookup elsewhere).
 */
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {

class TORCH_API AOTIModelContainerRunnerMps : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerMps(
      const std::string& model_so_path,
      size_t num_models = 1,
      const bool run_single_threaded = false);

  ~AOTIModelContainerRunnerMps() override;
};

} // namespace torch::inductor
