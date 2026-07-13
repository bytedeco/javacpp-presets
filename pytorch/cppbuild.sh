#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" pytorch
    popd
    exit
fi

export BUILD_TEST=0
#export CUDAHOSTCC="clang"
#export CUDAHOSTCXX="clang++"
export CUDACXX="/usr/local/cuda/bin/nvcc"
export CUDA_HOME="/usr/local/cuda"
export CUDNN_HOME="/usr/local/cuda"
export NCCL_ROOT="/usr/local/cuda"
export NCCL_ROOT_DIR="/usr/local/cuda"
export NCCL_INCLUDE_DIR="/usr/local/cuda/include"
export NCCL_LIB_DIR="/usr/local/cuda/lib64"
export NCCL_VERSION="2"
export MAX_JOBS=$MAKEJ
export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0
export USE_NUMPY=0
export USE_OPENMP=1
export USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1

if [[ "$EXTENSION" == *gpu ]]; then
    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_NCCL=1
    export USE_FAST_NVCC=0
    export CUDA_SEPARABLE_COMPILATION=OFF
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0;10.0;12.0"
fi

export PYTHON_BIN_PATH=$(which python3)
if [[ $PLATFORM == windows* ]]; then
    export PYTHON_BIN_PATH=$(which python.exe)
fi

PYTORCH_VERSION=2.12.0

export PYTORCH_BUILD_VERSION="$PYTORCH_VERSION"
export PYTORCH_BUILD_NUMBER=1

patch_embedding_from_pretrained() {
    local header="$1/torch/csrc/api/include/torch/nn/modules/embedding.h"
    if [[ ! -f "$header" ]] || grep -q 'JavaCPP from_pretrained adapters' "$header"; then
        return
    fi

    perl -i -ne '
        if (/^class TORCH_API EmbeddingImpl/) { $impl = "embedding"; }
        if (/^class TORCH_API EmbeddingBagImpl/) { $impl = "embedding_bag"; }
        if ($impl eq "embedding" && /  Tensor weight;/) {
            print;
            print "\n  // JavaCPP from_pretrained adapters\n";
            print "  static std::shared_ptr<EmbeddingImpl> from_pretrained(\n";
            print "      const Tensor& embeddings, const EmbeddingFromPretrainedOptions& options = {});\n";
            next;
        }
        if ($impl eq "embedding_bag" && /^  Tensor forward\(/) {
            print "\n  // JavaCPP from_pretrained adapters\n";
            print "  static std::shared_ptr<EmbeddingBagImpl> from_pretrained(\n";
            print "      const Tensor& embeddings, const EmbeddingBagFromPretrainedOptions& options = {});\n";
            print "  Tensor forward_with_offsets(const Tensor& input, const Tensor& offsets);\n\n";
        }
        if (/^class Embedding :/) { $holder = "embedding"; }
        if (/^class EmbeddingBag :/) { $holder = "embedding_bag"; }
        print;
        if ($holder eq "embedding" && /^};$/) {
            print "\ninline std::shared_ptr<EmbeddingImpl> EmbeddingImpl::from_pretrained(\n";
            print "    const Tensor& embeddings, const EmbeddingFromPretrainedOptions& options) {\n";
            print "  return Embedding::from_pretrained(embeddings, options).ptr();\n";
            print "}\n";
            $holder = "";
        }
        if ($holder eq "embedding_bag" && /^};$/) {
            print "\ninline std::shared_ptr<EmbeddingBagImpl> EmbeddingBagImpl::from_pretrained(\n";
            print "    const Tensor& embeddings, const EmbeddingBagFromPretrainedOptions& options) {\n";
            print "  return EmbeddingBag::from_pretrained(embeddings, options).ptr();\n";
            print "}\n";
            print "\ninline Tensor EmbeddingBagImpl::forward_with_offsets(\n";
            print "    const Tensor& input, const Tensor& offsets) {\n";
            print "  return forward(input, offsets, Tensor{});\n";
            print "}\n";
            $holder = "";
        }
        if ($impl ne "" && /^};$/) { $impl = ""; }
    ' "$header"
}

mkdir -p "$PLATFORM$EXTENSION"
cd "$PLATFORM$EXTENSION"
INSTALL_PATH=`pwd`

# For local macOS builds, reuse Homebrew's libtorch instead of cloning and
# rebuilding the full PyTorch source tree.
if [[ "$PLATFORM" == macosx-* && "$EXTENSION" != *gpu && "${USE_SYSTEM_LIBTORCH:-1}" != "0" ]]; then
    if [[ -z "${LIBTORCH_HOME:-}" ]] && command -v brew >/dev/null 2>&1; then
        LIBTORCH_HOME="$(brew --prefix libtorch 2>/dev/null || brew --prefix pytorch 2>/dev/null || true)"
    fi
    LIBTORCH_HOME="${LIBTORCH_HOME:-/opt/homebrew/opt/pytorch}"

    if [[ -f "$LIBTORCH_HOME/lib/libtorch.dylib" && -f "$LIBTORCH_HOME/lib/libtorch_cpu.dylib" && -f "$LIBTORCH_HOME/lib/libc10.dylib" && -d "$LIBTORCH_HOME/include" ]]; then
        echo "Using Homebrew libtorch from $LIBTORCH_HOME"
        rm -Rf include lib bin
        mkdir -p include lib bin
        cp -R -L "$LIBTORCH_HOME"/include/. include/
        patch_embedding_from_pretrained include

        # JavaCPP fix: libtorch 2.13 (cf30153c4c13) renamed
        # c10/util/AlignOf.h to c10/core/alignment.h. Anything in the
        # rest of the headers (or downstream parsers) still #includes
        # the old path; create a shim that just includes the new one.
        if [[ ! -f include/c10/util/AlignOf.h && -f include/c10/core/alignment.h ]]; then
            mkdir -p include/c10/util
            cat > include/c10/util/AlignOf.h <<'SHIM'
// JavaCPP fix (libtorch 2.13 rename shim):
//   c10/util/AlignOf.h was renamed to c10/core/alignment.h.
#pragma once
#include <c10/core/alignment.h>
SHIM
        fi

        # Apply the same header fixes that the source-build path applies below,
        # but only to the local copied headers.
        if [[ -f include/torch/csrc/api/include/torch/nn/options/pooling.h ]]; then
            sedinplace 's/using ExpandingArrayDouble/public: using ExpandingArrayDouble/g' include/torch/csrc/api/include/torch/nn/options/pooling.h
        fi
        sedinplace 's/TensorIndex(c10::nullopt_t.*)/TensorIndex(c10::nullopt_t none = None)/g' include/ATen/TensorIndexing.h
        sedinplace 's/TensorIndex(std::nullopt_t.*)/TensorIndex(std::nullopt_t none = None)/g' include/ATen/TensorIndexing.h
        sedinplace '/OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) =/{N;d;}' include/torch/csrc/api/include/torch/optim/optimizer.h
        sedinplace '/OptimizerParamGroup& operator=(OptimizerParamGroup&& param_group)/i\
  OptimizerParamGroup() {}\
  OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) {\
      params_ = param_group.params();\
      if (param_group.has_options())\
          options_ = param_group.options().clone();\
      return *this;\
  }\
' include/torch/csrc/api/include/torch/optim/optimizer.h
        sedinplace '/using ExampleType = ExampleType_;/a\
  using BatchType = ChunkType;\
  using DataType = ExampleType;\
' include/torch/csrc/api/include/torch/data/datasets/chunk.h
        sedinplace '/^};/a\
TORCH_API std::ostream& operator<<(std::ostream& stream, const nn::Module& module);\
' include/torch/csrc/api/include/torch/nn/module.h
        if ! grep -q '^struct TORCH_API ASMoutput;' include/torch/csrc/api/include/torch/nn/module.h; then
            sedinplace '/^namespace torch::nn {/a\
struct TORCH_API ASMoutput;\
' include/torch/csrc/api/include/torch/nn/module.h
        fi
        if ! grep -q '#include <tuple>' include/torch/csrc/api/include/torch/nn/module.h; then
            sedinplace '/#include <type_traits>/a\
#include <tuple>\
' include/torch/csrc/api/include/torch/nn/module.h
        fi
        if ! grep -q 'Module::forward_tuple_tensor_tensor_attn(query, key, value' include/torch/csrc/api/include/torch/nn/module.h; then
            sedinplace '/void apply(const ModuleApplyFunction& function);/i\
  virtual Tensor forward_tensor(const Tensor& input) { TORCH_CHECK(false, "Module::forward_tensor(input) is not implemented for ", name()); }\
  virtual Tensor forward_tensor2(const Tensor& input1, const Tensor& input2) { TORCH_CHECK(false, "Module::forward_tensor2(input1, input2) is not implemented for ", name()); }\
  virtual Tensor forward_tensor3(const Tensor& input1, const Tensor& input2, const Tensor& input3) { TORCH_CHECK(false, "Module::forward_tensor3(input1, input2, input3) is not implemented for ", name()); }\
  virtual Tensor forward_tensor4(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4) { TORCH_CHECK(false, "Module::forward_tensor4(input1, input2, input3, input4) is not implemented for ", name()); }\
  virtual Tensor forward_tensor6(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6) { TORCH_CHECK(false, "Module::forward_tensor6(input1..input6) is not implemented for ", name()); }\
  virtual Tensor forward_tensor8(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6, const Tensor& input7, const Tensor& input8) { TORCH_CHECK(false, "Module::forward_tensor8(input1..input8) is not implemented for ", name()); }\
  virtual Tensor forward_tensor_output_size(const Tensor& input, std::optional<at::IntArrayRef> output_size) { TORCH_CHECK(false, "Module::forward_tensor_output_size(input, output_size) is not implemented for ", name()); }\
  virtual Tensor forward_tensor_indices_output_size(const Tensor& input, const Tensor& indices, std::optional<std::vector<int64_t>> output_size) { TORCH_CHECK(false, "Module::forward_tensor_indices_output_size(input, indices, output_size) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor(const Tensor& input) { TORCH_CHECK(false, "Module::forward_tuple_tensor_t_tensortensor(input) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor_opt(const Tensor& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt) { TORCH_CHECK(false, "Module::forward_tuple_tensor_t_tensortensor_opt(input, hx_opt) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor(const Tensor& input) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor(input) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor2(const Tensor& input1, const Tensor& input2) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor2(input1, input2) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor3(const Tensor& input1, const Tensor& input2, const Tensor& input3) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor3(input1, input2, input3) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_opt(const Tensor& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor_opt(input, hx_opt) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_attn(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& key_padding_mask, bool need_weights, const Tensor& attn_mask, bool average_attn_weights) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor_attn(query, key, value, ...) is not implemented for ", name()); }\
  Tensor forward(const Tensor& input) { return forward_tensor(input); }\
  Tensor forward(const Tensor& input1, const Tensor& input2) { return forward_tensor2(input1, input2); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3) { return forward_tensor3(input1, input2, input3); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4) { return forward_tensor4(input1, input2, input3, input4); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6) { return forward_tensor6(input1, input2, input3, input4, input5, input6); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6, const Tensor& input7, const Tensor& input8) { return forward_tensor8(input1, input2, input3, input4, input5, input6, input7, input8); }\
  Tensor forward(const Tensor& input, std::optional<at::IntArrayRef> output_size) { return forward_tensor_output_size(input, output_size); }\
  Tensor forward(const Tensor& input, const Tensor& indices, std::optional<std::vector<int64_t>> output_size) { return forward_tensor_indices_output_size(input, indices, output_size); }\
  size_t javacpp_module_object_id() const noexcept { return reinterpret_cast<size_t>(this); }\
' include/torch/csrc/api/include/torch/nn/module.h
        fi
        if ! grep -q 'forward_method<ModuleType>' include/torch/csrc/api/include/torch/nn/modules/container/any.h; then
            sedinplace '/^ private:$/a\
  template <typename ModuleType>\
  static auto forward_method() {\
    using M = std::remove_cv_t<std::remove_reference_t<ModuleType>>;\
    if constexpr (std::is_same_v<M, Module>) {\
      return static_cast<Tensor (M::*)(const Tensor&)>(&M::forward_tensor);\
    } else {\
      return &M::forward;\
    }\
  }\
' include/torch/csrc/api/include/torch/nn/modules/container/any.h
            sedinplace 's/&std::remove_reference_t<ModuleType>::forward/forward_method<ModuleType>()/g' include/torch/csrc/api/include/torch/nn/modules/container/any.h
            sedinplace 's/torch::detail::has_forward<ModuleType>::value,/torch::detail::has_forward<ModuleType>::value || std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, Module>,/g' include/torch/csrc/api/include/torch/nn/modules/container/any.h
            sedinplace 's/torch::detail::has_forward<M>::value,/torch::detail::has_forward<M>::value || std::is_same_v<M, Module>,/g' include/torch/csrc/api/include/torch/nn/modules/container/any.h
            sedinplace 's/return get_(&M::forward);/return get_(forward_method<ModuleType>());/g' include/torch/csrc/api/include/torch/nn/modules/container/any.h
        fi
        if ! grep -q 'module_->forward_tensor' include/torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h; then
            sedinplace 's/return AnyValue(module_->forward(std::forward<Ts>(ts)...));/if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, Module>) {\
        return AnyValue(module_->forward_tensor(std::forward<Ts>(ts)...));\
      } else {\
        return AnyValue(module_->forward(std::forward<Ts>(ts)...));\
      }/g' include/torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h
        fi
        # JavaCPP fix: capture forward_method<ModuleType>() as a member
        # pointer at AnyModuleHolder construction time. Dispatch then
        # goes through (module_->*forward_)(...) directly — bypassing
        # the C++ name-hiding trap on derived `forward(Tensor)` (which
        # only ever declares a single by-value overload) and the
        # throwing `Module::forward_tensorN(...)` virtuals that the
        # *Impl classes never override. This is what makes
        # nn::Sequential usable from JavaCPP for DropoutImpl,
        # ReLUImpl, ELUImpl, SELUImpl, HardshrinkImpl, TanhshrinkImpl,
        # SoftsignImpl, SoftplusImpl, PReLUImpl, RReLUImpl, CELUImpl,
        # GLUImpl, SoftminImpl, SoftmaxImpl, LogSoftmaxImpl,
        # LogSigmoidImpl, HardtanhImpl, the Functional Lambda shim, etc.
        if ! grep -q 'JavaCPP captured-forward-dispatch fix' include/torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h; then
            perl -i -0pe 's{struct InvokeForward \{[\s\S]*?std::shared_ptr<ModuleType>& module_;\s*\};}{struct InvokeForward {
    // JavaCPP captured-forward-dispatch fix.
    // For concrete *Impl classes we capture &ModuleType::forward once at
    // construction time so dispatch goes through (module_.get()->*forward_)(...)
    // - this bypasses both the C++ name-hiding trap on derived forward(Tensor)
    // and the throwing Module::forward_tensorN virtuals that *Impl classes
    // never override. Module::forward is an overloaded set so we explicitly
    // specialise forward_member_ptr_t for it instead of writing &Module::forward.
    // For the Module base class the dispatch path uses module_->forward_tensor
    // directly via virtual dispatch (so we do not need to take the
    // address of Module::forward_tensor, which would force any_module_holder.h
    // to see Module full definition).
    template <typename T> struct forward_member_ptr_t { using type = decltype(\&T::forward); };
    template <> struct forward_member_ptr_t<torch::nn::Module> { using type = Tensor (torch::nn::Module::*)(const Tensor\&); };
    template <typename T>
    using forward_member_ptr = typename forward_member_ptr_t<std::remove_cv_t<std::remove_reference_t<T>>>::type;
    InvokeForward(std::shared_ptr<ModuleType>\& m) : module_(m) {
      if constexpr (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, torch::nn::Module>) {
        forward_ = \&ModuleType::forward;
      }
    }
    template <typename... Ts>
    AnyValue operator()(Ts\&\&... ts) {
      if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, torch::nn::Module>) {
        return AnyValue(module_->forward_tensor(std::forward<Ts>(ts)...));
      } else {
        return AnyValue((module_.get()->*forward_)(std::forward<Ts>(ts)...));
      }
    }
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    std::shared_ptr<ModuleType>\& module_;
    forward_member_ptr<ModuleType> forward_;
  };}s' include/torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h
        fi
        if ! grep -q 'JavaCPP OrderedDict<shared_ptr<Module>> ctor' include/torch/csrc/api/include/torch/nn/modules/container/sequential.h; then
            sedinplace '/Constructs the `Sequential` from an `OrderedDict` of named `AnyModule`s\./i\
  // JavaCPP OrderedDict<shared_ptr<Module>> ctor\
  explicit SequentialImpl(torch::OrderedDict<std::string, std::shared_ptr<Module>>& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto& item : ordered_dict) {\
      push_back(item.key(), item.value());\
    }\
  }\
\
  explicit SequentialImpl(torch::OrderedDict<std::string, std::shared_ptr<Module>>&& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto& item : ordered_dict) {\
      push_back(item.key(), std::move(item.value()));\
    }\
  }\
' include/torch/csrc/api/include/torch/nn/modules/container/sequential.h
        fi
        if ! grep -q 'JavaCPP OrderedDict<AnyModule> lvalue ctor' include/torch/csrc/api/include/torch/nn/modules/container/sequential.h; then
            sedinplace '/Constructs the `Sequential` from an `OrderedDict` of named `AnyModule`s\./i\
  // JavaCPP OrderedDict<AnyModule> lvalue ctor\
  explicit SequentialImpl(\
      torch::OrderedDict<std::string, AnyModule>& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto& item : ordered_dict) {\
      push_back(item.key(), item.value());\
    }\
  }\
\
' include/torch/csrc/api/include/torch/nn/modules/container/sequential.h
        fi
        sedinplace 's/if (module->_forward_has_default_args()) {/if (false \&\& module->_forward_has_default_args()) {/g' include/torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h
        # No header patches needed for javacpp-presets — the push_back
        # for Module is done through the C++ helper `push_back_module`
        # defined in torch.java preset's cppText, which wraps the user
        # pointer in a ModuleHolder<Module> to avoid the
        # has_forward<Module> static_assert in AnyModule.
        sedinplace 's/char(\(.*\))/\1/g' include/torch/csrc/jit/serialization/pickler.h
        sedinplace 's/const std::string& interface)/const std::string\& interface_name)/g' include/torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
        sedinplace '/^ private:$/,/^  torch::OrderedDict<std::string, std::shared_ptr<Module>> modules_;$/s/^ private:$/ public:/' include/torch/csrc/api/include/torch/nn/modules/container/moduledict.h

        # JavaCPP fix: libtorch 2.13 renamed c10/util/AlignOf.h to
        # c10/core/alignment.h. Provide a shim so anything still #including
        # the old path (downstream parsers / generator caches) compiles.
        if [[ ! -f include/c10/util/AlignOf.h && -f include/c10/core/alignment.h ]]; then
            mkdir -p include/c10/util
            cat > include/c10/util/AlignOf.h <<'SHIM'
// JavaCPP fix (libtorch 2.13 rename shim):
//   c10/util/AlignOf.h was renamed to c10/core/alignment.h.
#pragma once
#include <c10/core/alignment.h>
SHIM
        fi

        # JavaCPP fix: c10/core/AutogradState.h uses C++20 bitfield default
        # initializers (e.g. `bool x : 1 = false;`). JavaCPP's parser is
        # C++17-only and rejects them with "Unexpected token '='". Strip
        # the initializers from the bitfield declarations (the C++ side
        # defaults to value-initialized bits which is 0/false — same effect).
        if [[ -f include/c10/core/AutogradState.h ]]; then
            sedinplace -E 's/(bool [a-zA-Z_]+ : [0-9]+) = (false|true);/\1;/g' include/c10/core/AutogradState.h
        fi

        # JavaCPP fix: libtorch 2.13 renamed c10/util/AlignOf.h to
        # c10/core/alignment.h. Provide a shim so anything still #including
        # the old path (downstream parsers / generator caches) compiles.
        if [[ ! -f include/c10/util/AlignOf.h && -f include/c10/core/alignment.h ]]; then
            mkdir -p include/c10/util
            cat > include/c10/util/AlignOf.h <<'SHIM'
// JavaCPP fix (libtorch 2.13 rename shim):
//   c10/util/AlignOf.h was renamed to c10/core/alignment.h.
#pragma once
#include <c10/core/alignment.h>
SHIM
        fi

        # JavaCPP fix: c10/core/AutogradState.h uses C++20 bitfield default
        # initializers (e.g. `bool x : 1 = false;`). JavaCPP's parser is
        # C++17-only and rejects them with "Unexpected token '='". Strip
        # the initializers from the bitfield declarations (the C++ side
        # defaults to value-initialized bits which is 0/false — same effect).
        if [[ -f include/c10/core/AutogradState.h ]]; then
            sedinplace -E 's/(bool [a-zA-Z_]+ : [0-9]+) = (false|true);/\1;/g' include/c10/core/AutogradState.h
        fi

        for P in "$LIBTORCH_HOME"/lib/*.dylib; do
            [[ -e "$P" ]] || continue
            ln -sf "$P" "lib/$(basename "$P")"
        done
        if [[ -d "$LIBTORCH_HOME/bin" ]]; then
            for P in "$LIBTORCH_HOME"/bin/*; do
                [[ -e "$P" ]] || continue
                ln -sf "$P" "bin/$(basename "$P")"
            done
        fi

        # Homebrew installs ProcessGroupGloo.hpp but not the gloo headers it
        # includes, so keep a matching copy of PyTorch's gloo submodule headers.
        GLOO_COMMIT=3135b0b41b67dde590eef0938a0bf3d6238df5f7
        if [[ ! -f gloo-src/gloo/algorithm.h ]]; then
            rm -Rf gloo-src gloo.tar.gz
            if declare -f download >/dev/null 2>&1; then
                download "https://codeload.github.com/pytorch/gloo/tar.gz/$GLOO_COMMIT" gloo.tar.gz
            else
                curl -L "https://codeload.github.com/pytorch/gloo/tar.gz/$GLOO_COMMIT" -o gloo.tar.gz --fail
            fi
            mkdir gloo-src
            tar xfz gloo.tar.gz -C gloo-src --strip-components=1
        fi
        rm -Rf include/gloo
        ln -sf ../gloo-src/gloo include/gloo

        if command -v brew >/dev/null 2>&1; then
            LIBOMP_DYLIB="$(brew ls libomp 2>/dev/null | grep '/libomp.dylib$' | head -n 1 || true)"
            if [[ -n "$LIBOMP_DYLIB" ]]; then
                ln -sf "$LIBOMP_DYLIB" lib/libomp.dylib
                ln -sf "$LIBOMP_DYLIB" lib/libiomp5.dylib
            fi
        fi

        return 0
    fi
fi

# Distributed needs libuv on Windows (on other platforms, it's included in tensorpipe)
if [[ $PLATFORM == windows* ]]; then
    if [[ ! -d libuv ]]; then
        mkdir libuv
        cd libuv
        download https://dist.libuv.org/dist/v1.39.0/libuv-v1.39.0.tar.gz libuv.tgz
        tar xfz libuv.tgz
        mkdir build
        cd build
        export CC="cl.exe"
        export CXX="cl.exe"
        cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ../libuv-v1.39.0 -DBUILD_TESTING=OFF
        cmake --build . --config Release
        cmake --install . --config Release --prefix ../dist
        cd ../..
    fi
    export libuv_ROOT=${INSTALL_PATH}/libuv/dist
fi

if [[ ! -d pytorch ]]; then
    git clone https://github.com/pytorch/pytorch
fi
cd pytorch
git reset --hard
git checkout v$PYTORCH_VERSION
git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'

# https://github.com/pytorch/pytorch/pull/158184
# https://github.com/pytorch/pytorch/pull/159869
#patch -Np1 < ../../../pytorch.patch

# https://github.com/pytorch/pytorch/pull/164570
#patch -Np1 < ../../../pytorch-cuda.patch

CPYTHON_HOST_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/host/"
CPYTHON_PATH="$INSTALL_PATH/../../../cpython/cppbuild/$PLATFORM/"
OPENBLAS_PATH="${OPENBLAS_PATH:-$INSTALL_PATH/../../../openblas/cppbuild/$PLATFORM/}"
NUMPY_PATH="$INSTALL_PATH/../../../numpy/cppbuild/$PLATFORM/"

find_local_maven_openblas() {
    local repo="$HOME/.m2/repository/org/bytedeco/openblas"
    local jar
    jar=$(find "$repo" -type f -name "openblas-*-${PLATFORM}.jar" 2>/dev/null | sort | tail -n 1)
    if [[ -z "$jar" ]]; then
        return 1
    fi

    local extracted="$INSTALL_PATH/.cache/openblas/$PLATFORM"
    if [[ ! -f "$extracted/org/bytedeco/openblas/$PLATFORM/include/openblas_config.h" ]]; then
        rm -Rf "$extracted"
        mkdir -p "$extracted"
        (cd "$extracted" && jar xf "$jar")
    fi

    OPENBLAS_PATH="$extracted/org/bytedeco/openblas/$PLATFORM"
    return 0
}

if [[ -n "${BUILD_PATH:-}" ]]; then
    PREVIFS="$IFS"
    IFS="$BUILD_PATH_SEPARATOR"
    for P in $BUILD_PATH; do
        if [[ $(find "$P" -name Python.h) ]]; then
            if [[ "$(basename $P)" == "$PLATFORM_HOST" ]]; then
                CPYTHON_HOST_PATH="$P"
            fi
            if [[ "$(basename $P)" == "$PLATFORM" ]]; then
                CPYTHON_PATH="$P"
            fi
        elif [[ -f "$P/include/openblas_config.h" ]]; then
            OPENBLAS_PATH="$P"
        elif [[ -f "$P/python/numpy/_core/include/numpy/numpyconfig.h" ]]; then
            NUMPY_PATH="$P"
        fi
    done
    IFS="$PREVIFS"
fi

find_local_maven_openblas || true

export OpenBLAS_HOME=$OPENBLAS_PATH

CPYTHON_HOST_PATH="${CPYTHON_HOST_PATH//\\//}"
CPYTHON_PATH="${CPYTHON_PATH//\\//}"
OPENBLAS_PATH="${OPENBLAS_PATH//\\//}"
NUMPY_PATH="${NUMPY_PATH//\\//}"

CPYTHON_PATH="$CPYTHON_HOST_PATH"
if [[ -f "$CPYTHON_PATH/include/python3.14/Python.h" ]]; then
    # setup.py won't pick up the right libgfortran.so without this
    export LD_LIBRARY_PATH="$OPENBLAS_PATH/lib/:$CPYTHON_PATH/lib/:$NUMPY_PATH/lib/"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python3.14"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/python3.14/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/python3.14/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/python3.14/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/python3.14/site-packages/pip/_vendor/certifi/cacert.pem"
    chmod +x "$PYTHON_BIN_PATH"
elif [[ -f "$CPYTHON_PATH/include/Python.h" ]]; then
    CPYTHON_PATH=$(cygpath $CPYTHON_PATH)
    OPENBLAS_PATH=$(cygpath $OPENBLAS_PATH)
    NUMPY_PATH=$(cygpath $NUMPY_PATH)
    export PATH="$OPENBLAS_PATH:$CPYTHON_PATH:$NUMPY_PATH:$PATH"
    export PYTHON_BIN_PATH="$CPYTHON_PATH/bin/python.exe"
    export PYTHON_INCLUDE_PATH="$CPYTHON_PATH/include/"
    export PYTHON_LIB_PATH="$CPYTHON_PATH/lib/"
    export PYTHON_INSTALL_PATH="$INSTALL_PATH/lib/site-packages/"
    export SSL_CERT_FILE="$CPYTHON_PATH/lib/pip/_vendor/certifi/cacert.pem"
fi
export PYTHONPATH="$PYTHON_INSTALL_PATH:$NUMPY_PATH/python/"
mkdir -p "$PYTHON_INSTALL_PATH"

export CFLAGS="-I$CPYTHON_PATH/include/ -I$PYTHON_LIB_PATH/include/python/"
export PYTHONNOUSERSITE=1
$PYTHON_BIN_PATH -m pip install --target=$PYTHON_LIB_PATH setuptools==67.6.1 pyyaml==6.0.2 typing_extensions==4.8.0 packaging==25.0

case $PLATFORM in
    linux-x86)
        export CC="gcc -m32"
        export CXX="g++ -m32"
        ;;
    linux-x86_64)
        export CC="gcc -m64"
        export CXX="g++ -m64"
        ;;
    macosx-arm64)
        export CC="clang"
        export CXX="clang++"
        # export PATH=$(brew --prefix llvm@18)/bin:$PATH # Use brew LLVM instead of Xcode LLVM 14
        export USE_MKLDNN=OFF
        export USE_QNNPACK=OFF # not compatible with arm64 as of PyTorch 2.1.2
        export CMAKE_OSX_DEPLOYMENT_TARGET=11.00 # minimum needed for arm64 support
        ;;
    macosx-x86_64)
        export CC="clang"
        export CXX="clang++"
        export USE_MKLDNN=OFF
        # export PATH=$(brew --prefix llvm@18)/bin:$PATH # Use brew LLVM instead of Xcode LLVM 14
        ;;
    windows-x86_64)
        if which ccache.exe; then
            export CC="ccache.exe cl.exe"
            export CXX="ccache.exe cl.exe"
#            export CUDAHOSTCC="cl.exe"
#            export CUDAHOSTCXX="cl.exe"
        else
            export CC="cl.exe"
            export CXX="cl.exe"
        fi
        if [[ -n "${CUDA_PATH:-}" ]]; then
            export CUDACXX="$CUDA_PATH/bin/nvcc"
            export CUDA_HOME="$CUDA_PATH"
            export CUDNN_HOME="$CUDA_PATH"
        fi
        export USE_NCCL=0 # Not supported on Windows
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        return 0
        ;;
esac

#sedinplace 's/,code=sm_.*)/,code=compute_60)/g' cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake

# work around issues with the build system
sedinplace '/Werror/d' CMakeLists.txt third_party/fbgemm/CMakeLists.txt third_party/fmt/CMakeLists.txt
sedinplace '/setuptools.command.bdist_wheel/d' setup.py
sedinplace 's/build_python=True/build_python=False/g' setup.py
sedinplace 's/    build_deps()/    build_deps(); sys.exit()/g' setup.py
sedinplace 's/AND NOT DEFINED ENV{CUDAHOSTCXX}//g' cmake/public/cuda.cmake
sedinplace 's/CMAKE_CUDA_FLAGS "/CMAKE_CUDA_FLAGS " --use-local-env /g' CMakeLists.txt
sedinplace 's/-Xcompiler  \/Zc:__cplusplus/-Xcompiler  \/Zc:__cplusplus -Xcompiler \/Zc:preprocessor/g' CMakeLists.txt

#sedinplace '/pycore_opcode.h/d' torch/csrc/dynamo/cpython_defs.c functorch/csrc/dim/dim*
sedinplace 's/using ExpandingArrayDouble/public: using ExpandingArrayDouble/g' ./torch/csrc/api/include/torch/nn/options/pooling.h

# allow setting the build directory and passing CUDA options
sedinplace "s/BUILD_DIR = .build./BUILD_DIR = os.environ['BUILD_DIR'] if 'BUILD_DIR' in os.environ else 'build'/g" tools/setup_helpers/env.py
sedinplace 's/var.startswith(("BUILD_", "USE_", "CMAKE_"))/var.startswith(("BUILD_", "USE_", "CMAKE_", "CUDA_"))/g' tools/setup_helpers/cmake.py

# allow resizing std::vector<at::indexing::TensorIndex> and std::vector<torch::optim::OptimizerParamGroup>
sedinplace 's/TensorIndex(c10::nullopt_t.*)/TensorIndex(c10::nullopt_t none = None)/g' aten/src/ATen/TensorIndexing.h
sedinplace 's/TensorIndex(std::nullopt_t.*)/TensorIndex(std::nullopt_t none = None)/g' aten/src/ATen/TensorIndexing.h
sedinplace '/OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) =/{N;d;}' torch/csrc/api/include/torch/optim/optimizer.h
sedinplace '/OptimizerParamGroup& operator=(OptimizerParamGroup&& param_group)/i\
  OptimizerParamGroup() {}\
  OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) {\
      params_ = param_group.params();\
      if (param_group.has_options())\
          options_ = param_group.options().clone();\
      return *this;\
  }\
' torch/csrc/api/include/torch/optim/optimizer.h

# add missing declarations
sedinplace '/using ExampleType = ExampleType_;/a\
  using BatchType = ChunkType;\
  using DataType = ExampleType;\
' torch/csrc/api/include/torch/data/datasets/chunk.h
sedinplace '/^};/a\
TORCH_API std::ostream& operator<<(std::ostream& stream, const nn::Module& module);\
' torch/csrc/api/include/torch/nn/module.h
if ! grep -q '^struct TORCH_API ASMoutput;' torch/csrc/api/include/torch/nn/module.h; then
    sedinplace '/^namespace torch::nn {/a\
struct TORCH_API ASMoutput;\
' torch/csrc/api/include/torch/nn/module.h
fi
if ! grep -q '#include <tuple>' torch/csrc/api/include/torch/nn/module.h; then
    sedinplace '/#include <type_traits>/a\
#include <tuple>\
' torch/csrc/api/include/torch/nn/module.h
fi
if ! grep -q 'Module::forward_tuple_tensor_tensor_attn(query, key, value' torch/csrc/api/include/torch/nn/module.h; then
    sedinplace '/void apply(const ModuleApplyFunction& function);/i\
  virtual Tensor forward_tensor(const Tensor& input) { TORCH_CHECK(false, "Module::forward_tensor(input) is not implemented for ", name()); }\
  virtual Tensor forward_tensor2(const Tensor& input1, const Tensor& input2) { TORCH_CHECK(false, "Module::forward_tensor2(input1, input2) is not implemented for ", name()); }\
  virtual Tensor forward_tensor3(const Tensor& input1, const Tensor& input2, const Tensor& input3) { TORCH_CHECK(false, "Module::forward_tensor3(input1, input2, input3) is not implemented for ", name()); }\
  virtual Tensor forward_tensor4(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4) { TORCH_CHECK(false, "Module::forward_tensor4(input1, input2, input3, input4) is not implemented for ", name()); }\
  virtual Tensor forward_tensor6(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6) { TORCH_CHECK(false, "Module::forward_tensor6(input1..input6) is not implemented for ", name()); }\
  virtual Tensor forward_tensor8(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6, const Tensor& input7, const Tensor& input8) { TORCH_CHECK(false, "Module::forward_tensor8(input1..input8) is not implemented for ", name()); }\
  virtual Tensor forward_tensor_output_size(const Tensor& input, std::optional<at::IntArrayRef> output_size) { TORCH_CHECK(false, "Module::forward_tensor_output_size(input, output_size) is not implemented for ", name()); }\
  virtual Tensor forward_tensor_indices_output_size(const Tensor& input, const Tensor& indices, std::optional<std::vector<int64_t>> output_size) { TORCH_CHECK(false, "Module::forward_tensor_indices_output_size(input, indices, output_size) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor(const Tensor& input) { TORCH_CHECK(false, "Module::forward_tuple_tensor_t_tensortensor(input) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor_opt(const Tensor& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt) { TORCH_CHECK(false, "Module::forward_tuple_tensor_t_tensortensor_opt(input, hx_opt) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor(const Tensor& input) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor(input) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor2(const Tensor& input1, const Tensor& input2) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor2(input1, input2) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor3(const Tensor& input1, const Tensor& input2, const Tensor& input3) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor3(input1, input2, input3) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_opt(const Tensor& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor_opt(input, hx_opt) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_attn(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& key_padding_mask, bool need_weights, const Tensor& attn_mask, bool average_attn_weights) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor_attn(query, key, value, ...) is not implemented for ", name()); }\
  Tensor forward(const Tensor& input) { return forward_tensor(input); }\
  Tensor forward(const Tensor& input1, const Tensor& input2) { return forward_tensor2(input1, input2); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3) { return forward_tensor3(input1, input2, input3); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4) { return forward_tensor4(input1, input2, input3, input4); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6) { return forward_tensor6(input1, input2, input3, input4, input5, input6); }\
  Tensor forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6, const Tensor& input7, const Tensor& input8) { return forward_tensor8(input1, input2, input3, input4, input5, input6, input7, input8); }\
  Tensor forward(const Tensor& input, std::optional<at::IntArrayRef> output_size) { return forward_tensor_output_size(input, output_size); }\
  Tensor forward(const Tensor& input, const Tensor& indices, std::optional<std::vector<int64_t>> output_size) { return forward_tensor_indices_output_size(input, indices, output_size); }\
  size_t javacpp_module_object_id() const noexcept { return reinterpret_cast<size_t>(this); }\
' torch/csrc/api/include/torch/nn/module.h
fi
if ! grep -q 'forward_method<ModuleType>' torch/csrc/api/include/torch/nn/modules/container/any.h; then
    sedinplace '/^ private:$/a\
  template <typename ModuleType>\
  static auto forward_method() {\
    using M = std::remove_cv_t<std::remove_reference_t<ModuleType>>;\
    if constexpr (std::is_same_v<M, Module>) {\
      return static_cast<Tensor (M::*)(const Tensor&)>(&M::forward_tensor);\
    } else {\
      return &M::forward;\
    }\
  }\
' torch/csrc/api/include/torch/nn/modules/container/any.h
    sedinplace 's/&std::remove_reference_t<ModuleType>::forward/forward_method<ModuleType>()/g' torch/csrc/api/include/torch/nn/modules/container/any.h
    sedinplace 's/torch::detail::has_forward<ModuleType>::value,/torch::detail::has_forward<ModuleType>::value || std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, Module>,/g' torch/csrc/api/include/torch/nn/modules/container/any.h
    sedinplace 's/torch::detail::has_forward<M>::value,/torch::detail::has_forward<M>::value || std::is_same_v<M, Module>,/g' torch/csrc/api/include/torch/nn/modules/container/any.h
    sedinplace 's/return get_(&M::forward);/return get_(forward_method<ModuleType>());/g' torch/csrc/api/include/torch/nn/modules/container/any.h
fi
if ! grep -q 'module_->forward_tensor' torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h; then
    sedinplace 's/return AnyValue(module_->forward(std::forward<Ts>(ts)...));/if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, Module>) {\
        return AnyValue(module_->forward_tensor(std::forward<Ts>(ts)...));\
      } else {\
        return AnyValue(module_->forward(std::forward<Ts>(ts)...));\
      }/g' torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h
fi
# JavaCPP fix: capture forward_method<ModuleType>() as a member pointer
# at AnyModuleHolder construction time. Dispatch then goes through
# (module_->*forward_)(...) directly — bypassing the C++ name-hiding
# trap on derived `forward(Tensor)` (which only ever declares a single
# by-value overload) and the throwing `Module::forward_tensorN(...)`
# virtuals that the *Impl classes never override. This is what makes
# nn::Sequential usable from JavaCPP for DropoutImpl, ReLUImpl,
# ELUImpl, SELUImpl, HardshrinkImpl, TanhshrinkImpl, SoftsignImpl,
# SoftplusImpl, PReLUImpl, RReLUImpl, CELUImpl, GLUImpl, SoftminImpl,
# SoftmaxImpl, LogSoftmaxImpl, LogSigmoidImpl, HardtanhImpl, the
# Functional Lambda shim, etc.
if ! grep -q 'JavaCPP captured-forward-dispatch fix' torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h; then
    perl -i -0pe 's{struct InvokeForward \{[\s\S]*?std::shared_ptr<ModuleType>& module_;\s*\};}{struct InvokeForward {
    // JavaCPP captured-forward-dispatch fix.
    // For concrete *Impl classes we capture &ModuleType::forward once at
    // construction time so dispatch goes through (module_.get()->*forward_)(...)
    // - this bypasses both the C++ name-hiding trap on derived forward(Tensor)
    // and the throwing Module::forward_tensorN virtuals that *Impl classes
    // never override. Module::forward is an overloaded set so we explicitly
    // specialise forward_member_ptr_t for it instead of writing &Module::forward.
    // For the Module base class the dispatch path uses module_->forward_tensor
    // directly via virtual dispatch (so we do not need to take the
    // address of Module::forward_tensor, which would force any_module_holder.h
    // to see Module full definition).
    template <typename T> struct forward_member_ptr_t { using type = decltype(\&T::forward); };
    template <> struct forward_member_ptr_t<torch::nn::Module> { using type = Tensor (torch::nn::Module::*)(const Tensor\&); };
    template <typename T>
    using forward_member_ptr = typename forward_member_ptr_t<std::remove_cv_t<std::remove_reference_t<T>>>::type;
    InvokeForward(std::shared_ptr<ModuleType>\& m) : module_(m) {
      if constexpr (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, torch::nn::Module>) {
        forward_ = \&ModuleType::forward;
      }
    }
    template <typename... Ts>
    AnyValue operator()(Ts\&\&... ts) {
      if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, torch::nn::Module>) {
        return AnyValue(module_->forward_tensor(std::forward<Ts>(ts)...));
      } else {
        return AnyValue((module_.get()->*forward_)(std::forward<Ts>(ts)...));
      }
    }
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    std::shared_ptr<ModuleType>\& module_;
    forward_member_ptr<ModuleType> forward_;
  };}s' torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h
fi
if ! grep -q 'JavaCPP OrderedDict<shared_ptr<Module>> ctor' torch/csrc/api/include/torch/nn/modules/container/sequential.h; then
    sedinplace '/Constructs the `Sequential` from an `OrderedDict` of named `AnyModule`s\./i\
  // JavaCPP OrderedDict<shared_ptr<Module>> ctor\
  explicit SequentialImpl(torch::OrderedDict<std::string, std::shared_ptr<Module>>& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto& item : ordered_dict) {\
      push_back(item.key(), item.value());\
    }\
  }\
\
  explicit SequentialImpl(torch::OrderedDict<std::string, std::shared_ptr<Module>>&& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto& item : ordered_dict) {\
      push_back(item.key(), std::move(item.value()));\
    }\
  }\
' torch/csrc/api/include/torch/nn/modules/container/sequential.h
fi
if ! grep -q 'JavaCPP OrderedDict<AnyModule> lvalue ctor' torch/csrc/api/include/torch/nn/modules/container/sequential.h; then
    sedinplace '/Constructs the `Sequential` from an `OrderedDict` of named `AnyModule`s\./i\
  // JavaCPP OrderedDict<AnyModule> lvalue ctor\
  explicit SequentialImpl(\
      torch::OrderedDict<std::string, AnyModule>& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto& item : ordered_dict) {\
      push_back(item.key(), item.value());\
    }\
  }\
\
' torch/csrc/api/include/torch/nn/modules/container/sequential.h
fi
sedinplace 's/if (module->_forward_has_default_args()) {/if (false \&\& module->_forward_has_default_args()) {/g' torch/csrc/api/include/torch/nn/modules/container/any_module_holder.h
sedinplace 's/char(\(.*\))/\1/g' torch/csrc/jit/serialization/pickler.h

# some windows header defines a macro named "interface"
sedinplace 's/const std::string& interface)/const std::string\& interface_name)/g' torch/csrc/distributed/c10d/ProcessGroupGloo.hpp

# fix missing #include (Pytorch 2.4.0)
sedinplace 's/#include <stdexcept>/#include <stdexcept>\
#include <vector>\
#include <unordered_map>/'  torch/csrc/distributed/c10d/control_plane/Handlers.cpp

# Remove pytorch adaptations of FindOpenMP.cmake that.
# On Windows without iomp and with new versions of VS 2019, including -openmp:experimental and libomp, causes
# final binary to be linked to both libomp and vcomp and produce incorrect results.
# Wait for eventual upstream fix, or for cmake 2.30 that allows to choose between -openmp and -openmp:experimental
# and see if choosing experimental works. See Issue #1503.
# On Linux, pytorch FindOpenMP.cmake picks llvm libomp over libgomp. See Issue #1504.
# On MacOS CMake standard version works tooL
rm -f cmake/Modules/FindOpenMP.cmake
sedinplace 's/include(${CMAKE_CURRENT_LIST_DIR}\/Modules\/FindOpenMP.cmake)/find_package(OpenMP)/g' cmake/Dependencies.cmake

# delete broken CUDA kernels at least on Windows with CUDA 12.9
#rm -f aten/src/ATen/native/cuda/SegmentReduce.cu

#USE_FBGEMM=0 USE_KINETO=0 USE_GLOO=0 USE_MKLDNN=0 \
BLAS=OpenBLAS "$PYTHON_BIN_PATH" setup.py build

rm -Rf ../lib
if [[ ! -e torch/include/gloo ]]; then
    ln -sf ../../third_party/gloo/gloo torch/include
fi
ln -sf pytorch/torch/include ../include
ln -sf pytorch/torch/lib ../lib
ln -sf pytorch/torch/bin ../bin

case $PLATFORM in
    macosx-arm64)
        cp "$(brew ls libomp|grep libomp.dylib)" ../lib/
        ;;
    macosx-x86_64)
        # Disguise libomp as libiomp5 (they share the same codebase and have the same symbols)
        # This helps if user wants to link with MKL.
        # On linux, user linking with mkl would need to set
        # MKL_THREADING_LAYER=GNU
        cp "$(brew ls libomp|grep libomp.dylib)" ../lib/libiomp5.dylib
        chmod +w ../lib/libiomp5.dylib
        install_name_tool -id @rpath/libiomp5.dylib ../lib/libiomp5.dylib
        codesign --force -s - ../lib/libiomp5.dylib
        old=$(otool -L ../lib/libtorch_cpu.dylib|grep libomp.dylib|awk '{print $1}')
        echo install_name_tool -change $old @rpath/libiomp5.dylib ../lib/libtorch_cpu.dylib
        install_name_tool -change $old @rpath/libiomp5.dylib ../lib/libtorch_cpu.dylib
        codesign --force -s - ../lib/libtorch_cpu.dylib
        ;;
    windows-*)
        cp ../libuv/dist/lib/Release/* ../lib
	;;
esac

cd ../..
