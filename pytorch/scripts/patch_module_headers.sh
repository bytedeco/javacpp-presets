#!/bin/bash
# Apply JavaCPP Module/AnyModule/Sequential/Embedding header patches to a
# libtorch include tree. Idempotent. Usage:
#   scripts/patch_module_headers.sh <include-root>
# where include-root contains torch/csrc/api/include/...
set -euo pipefail
ROOT="${1:?include root required}"
API="$ROOT/torch/csrc/api/include/torch"

sedinplace() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

# --- embedding from_pretrained ---
patch_embedding_from_pretrained() {
    local header="$API/nn/modules/embedding.h"
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

# --- Module forward_tensor overloads + object id ---
#
# CRITICAL ABI RULE: never insert NEW virtual methods before existing Module
# virtuals (train/is_training/to/zero_grad/save/load/pretty_print/
# is_serializable/_forward_*/clone_). Stock libtorch_cpu.dylib was compiled
# against the unpatched layout; peer objects built against a shifted vtable
# make stock *Impl::forward (e.g. DropoutImpl) call is_training() and land
# on the wrong slot (historically forward_tensor2) -> TORCH_CHECK crash.
#
# Therefore:
#   - virtual forward_tensor* / forward_tuple*  -> append AFTER clone_ (last
#     stock virtual), still private so layout of public/protected virtuals
#     is untouched and new slots only extend the end of the vtable.
#   - non-virtual Tensor forward(...) wrappers + javacpp_module_object_id()
#     -> public section before apply() (no vtable impact).
#
patch_module_h() {
    local h="$API/nn/module.h"
    [[ -f "$h" ]] || return 0

    if ! grep -q '^struct TORCH_API ASMoutput;' "$h"; then
        sedinplace '/^namespace torch::nn {/a\
struct TORCH_API ASMoutput;\
' "$h"
    fi
    if ! grep -q '#include <tuple>' "$h"; then
        sedinplace '/#include <type_traits>/a\
#include <tuple>\
' "$h"
    fi

    # Migration: remove a previous early (ABI-breaking) insertion of virtual
    # forward_tensor* that was placed before apply(). Idempotent for headers
    # that already only have the late (after clone_) placement.
    if grep -q 'virtual Tensor forward_tensor(const Tensor& input)' "$h"; then
        first_line=$(grep -n 'virtual Tensor forward_tensor(const Tensor& input)' "$h" | head -1 | cut -d: -f1)
        if [ -n "$first_line" ] && [ "$first_line" -lt 400 ]; then
            python3 - "$h" <<'PY2'
import re, sys
path = sys.argv[1]
text = open(path).read()
# Remove the early block of virtuals + non-virtual wrappers that used to sit before apply()
pat = re.compile(
    r'  virtual Tensor forward_tensor\(const Tensor& input\) \{ TORCH_CHECK\(false, "Module::forward_tensor\(input\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor2\(const Tensor& input1, const Tensor& input2\) \{ TORCH_CHECK\(false, "Module::forward_tensor2\(input1, input2\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor3\(const Tensor& input1, const Tensor& input2, const Tensor& input3\) \{ TORCH_CHECK\(false, "Module::forward_tensor3\(input1, input2, input3\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor4\(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4\) \{ TORCH_CHECK\(false, "Module::forward_tensor4\(input1, input2, input3, input4\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor6\(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6\) \{ TORCH_CHECK\(false, "Module::forward_tensor6\(input1\.\.input6\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor8\(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6, const Tensor& input7, const Tensor& input8\) \{ TORCH_CHECK\(false, "Module::forward_tensor8\(input1\.\.input8\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor_output_size\(const Tensor& input, std::optional<at::IntArrayRef> output_size\) \{ TORCH_CHECK\(false, "Module::forward_tensor_output_size\(input, output_size\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual Tensor forward_tensor_indices_output_size\(const Tensor& input, const Tensor& indices, std::optional<std::vector<int64_t>> output_size\) \{ TORCH_CHECK\(false, "Module::forward_tensor_indices_output_size\(input, indices, output_size\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor\(const Tensor& input\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_t_tensortensor\(input\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor_opt\(const Tensor& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_t_tensortensor_opt\(input, hx_opt\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor\(const Tensor& input\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_tensor\(input\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor2\(const Tensor& input1, const Tensor& input2\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_tensor2\(input1, input2\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor3\(const Tensor& input1, const Tensor& input2, const Tensor& input3\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_tensor3\(input1, input2, input3\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_opt\(const Tensor& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_tensor_opt\(input, hx_opt\) is not implemented for ", name\(\)\); \}\n'
    r'  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_attn\(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& key_padding_mask, bool need_weights, const Tensor& attn_mask, bool average_attn_weights\) \{ TORCH_CHECK\(false, "Module::forward_tuple_tensor_tensor_attn\(query, key, value, \.\.\.\) is not implemented for ", name\(\)\); \}\n'
    r'  Tensor forward\(const Tensor& input\) \{ return forward_tensor\(input\); \}\n'
    r'  Tensor forward\(const Tensor& input1, const Tensor& input2\) \{ return forward_tensor2\(input1, input2\); \}\n'
    r'  Tensor forward\(const Tensor& input1, const Tensor& input2, const Tensor& input3\) \{ return forward_tensor3\(input1, input2, input3\); \}\n'
    r'  Tensor forward\(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4\) \{ return forward_tensor4\(input1, input2, input3, input4\); \}\n'
    r'  Tensor forward\(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6\) \{ return forward_tensor6\(input1, input2, input3, input4, input5, input6\); \}\n'
    r'  Tensor forward\(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& input5, const Tensor& input6, const Tensor& input7, const Tensor& input8\) \{ return forward_tensor8\(input1, input2, input3, input4, input5, input6, input7, input8\); \}\n'
    r'  Tensor forward\(const Tensor& input, std::optional<at::IntArrayRef> output_size\) \{ return forward_tensor_output_size\(input, output_size\); \}\n'
    r'  Tensor forward\(const Tensor& input, const Tensor& indices, std::optional<std::vector<int64_t>> output_size\) \{ return forward_tensor_indices_output_size\(input, indices, output_size\); \}\n'
    r'  size_t javacpp_module_object_id\(\) const noexcept \{ return reinterpret_cast<size_t>\(this\); \}\n',
    re.M)
# Only strip the first (early) occurrence
m = pat.search(text)
if m and m.start() < text.find('void apply(const ModuleApplyFunction& function);'):
    text = text[:m.start()] + text[m.end():]
    open(path, 'w').write(text)
    print(f"Stripped early ABI-breaking forward_tensor block from {path}")
else:
    print(f"No early block to strip in {path} (or already late)")
PY2
        fi
    fi

    # Non-virtual public forward wrappers + object id (safe anywhere public).
    if ! grep -q 'javacpp_module_object_id()' "$h"; then
        sedinplace '/void apply(const ModuleApplyFunction& function);/i\
  // JavaCPP multi-arg forward shims (non-virtual - no vtable impact).\
  Tensor forward(const Tensor\& input) { return forward_tensor(input); }\
  Tensor forward(const Tensor\& input1, const Tensor\& input2) { return forward_tensor2(input1, input2); }\
  Tensor forward(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3) { return forward_tensor3(input1, input2, input3); }\
  Tensor forward(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3, const Tensor\& input4) { return forward_tensor4(input1, input2, input3, input4); }\
  Tensor forward(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3, const Tensor\& input4, const Tensor\& input5, const Tensor\& input6) { return forward_tensor6(input1, input2, input3, input4, input5, input6); }\
  Tensor forward(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3, const Tensor\& input4, const Tensor\& input5, const Tensor\& input6, const Tensor\& input7, const Tensor\& input8) { return forward_tensor8(input1, input2, input3, input4, input5, input6, input7, input8); }\
  Tensor forward(const Tensor\& input, std::optional<at::IntArrayRef> output_size) { return forward_tensor_output_size(input, output_size); }\
  Tensor forward(const Tensor\& input, const Tensor\& indices, std::optional<std::vector<int64_t>> output_size) { return forward_tensor_indices_output_size(input, indices, output_size); }\
  size_t javacpp_module_object_id() const noexcept { return reinterpret_cast<size_t>(this); }\
' "$h"
    fi

    # Virtual forward_tensor* AFTER clone_ (last stock virtual) - ABI-safe.
    if ! grep -q 'virtual Tensor forward_tensor(const Tensor& input)' "$h"; then
        sedinplace '/virtual void clone_(Module& other, const std::optional<Device>& device);/a\
\
 public:\
  // JavaCPP multi-arity forward virtuals (appended AFTER all stock virtuals so\
  // peer vtables keep stock is_training/to/train/... slots ABI-compatible with\
  // prebuilt libtorch_cpu). Must be public: JavaCPP peers override them.\
  virtual Tensor forward_tensor(const Tensor\& input) { TORCH_CHECK(false, "Module::forward_tensor(input) is not implemented for ", name()); }\
  virtual Tensor forward_tensor2(const Tensor\& input1, const Tensor\& input2) { TORCH_CHECK(false, "Module::forward_tensor2(input1, input2) is not implemented for ", name()); }\
  virtual Tensor forward_tensor3(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3) { TORCH_CHECK(false, "Module::forward_tensor3(input1, input2, input3) is not implemented for ", name()); }\
  virtual Tensor forward_tensor4(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3, const Tensor\& input4) { TORCH_CHECK(false, "Module::forward_tensor4(input1, input2, input3, input4) is not implemented for ", name()); }\
  virtual Tensor forward_tensor6(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3, const Tensor\& input4, const Tensor\& input5, const Tensor\& input6) { TORCH_CHECK(false, "Module::forward_tensor6(input1..input6) is not implemented for ", name()); }\
  virtual Tensor forward_tensor8(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3, const Tensor\& input4, const Tensor\& input5, const Tensor\& input6, const Tensor\& input7, const Tensor\& input8) { TORCH_CHECK(false, "Module::forward_tensor8(input1..input8) is not implemented for ", name()); }\
  virtual Tensor forward_tensor_output_size(const Tensor\& input, std::optional<at::IntArrayRef> output_size) { TORCH_CHECK(false, "Module::forward_tensor_output_size(input, output_size) is not implemented for ", name()); }\
  virtual Tensor forward_tensor_indices_output_size(const Tensor\& input, const Tensor\& indices, std::optional<std::vector<int64_t>> output_size) { TORCH_CHECK(false, "Module::forward_tensor_indices_output_size(input, indices, output_size) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor(const Tensor\& input) { TORCH_CHECK(false, "Module::forward_tuple_tensor_t_tensortensor(input) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward_tuple_tensor_t_tensortensor_opt(const Tensor\& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt) { TORCH_CHECK(false, "Module::forward_tuple_tensor_t_tensortensor_opt(input, hx_opt) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor(const Tensor\& input) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor(input) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor2(const Tensor\& input1, const Tensor\& input2) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor2(input1, input2) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor3(const Tensor\& input1, const Tensor\& input2, const Tensor\& input3) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor3(input1, input2, input3) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_opt(const Tensor\& input, std::optional<std::tuple<Tensor, Tensor>> hx_opt) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor_opt(input, hx_opt) is not implemented for ", name()); }\
  virtual std::tuple<Tensor, Tensor> forward_tuple_tensor_tensor_attn(const Tensor\& query, const Tensor\& key, const Tensor\& value, const Tensor\& key_padding_mask, bool need_weights, const Tensor\& attn_mask, bool average_attn_weights) { TORCH_CHECK(false, "Module::forward_tuple_tensor_tensor_attn(query, key, value, ...) is not implemented for ", name()); }\
' "$h"
    fi

    if ! grep -q 'operator<<(std::ostream& stream, const nn::Module& module)' "$h"; then
        sedinplace '/^};/a\
TORCH_API std::ostream\& operator<<(std::ostream\& stream, const nn::Module\& module);\
' "$h" || true
    fi
    # Dedup accidental double operator<< from prior patches
    perl -i -0pe 's/(TORCH_API std::ostream& operator<<\(std::ostream& stream, const nn::Module& module\);\n){2,}/$1/g' "$h"
}

# --- AnyModule / holder / Sequential ---
patch_any_sequential() {
    local anyh="$API/nn/modules/container/any.h"
    local holder="$API/nn/modules/container/any_module_holder.h"
    local seq="$API/nn/modules/container/sequential.h"
    if [[ -f "$anyh" ]] && ! grep -q 'forward_method<ModuleType>' "$anyh"; then
        sedinplace '/^ private:$/a\
  template <typename ModuleType>\
  static auto forward_method() {\
    using M = std::remove_cv_t<std::remove_reference_t<ModuleType>>;\
    if constexpr (std::is_same_v<M, Module>) {\
      return static_cast<Tensor (M::*)(const Tensor\&)>(\&M::forward_tensor);\
    } else {\
      return \&M::forward;\
    }\
  }\
' "$anyh"
        sedinplace 's/&std::remove_reference_t<ModuleType>::forward/forward_method<ModuleType>()/g' "$anyh"
        sedinplace 's/torch::detail::has_forward<ModuleType>::value,/torch::detail::has_forward<ModuleType>::value || std::is_same_v<std::remove_cv_t<std::remove_reference_t<ModuleType>>, Module>,/g' "$anyh"
        sedinplace 's/torch::detail::has_forward<M>::value,/torch::detail::has_forward<M>::value || std::is_same_v<M, Module>,/g' "$anyh"
        sedinplace 's/return get_(&M::forward);/return get_(forward_method<ModuleType>());/g' "$anyh"
    fi
    if [[ -f "$holder" ]]; then
        sedinplace 's/if (module->_forward_has_default_args()) {/if (false \&\& module->_forward_has_default_args()) {/g' "$holder"
        if ! grep -q 'JavaCPP captured-forward-dispatch fix' "$holder"; then
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
  };}s' "$holder"
        fi
    fi
    if [[ -f "$seq" ]] && ! grep -q 'JavaCPP OrderedDict<AnyModule> lvalue ctor' "$seq"; then
        sedinplace '/Constructs the `Sequential` from an `OrderedDict` of named `AnyModule`s\./i\
  // JavaCPP OrderedDict<AnyModule> lvalue ctor\
  explicit SequentialImpl(\
      torch::OrderedDict<std::string, AnyModule>\& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto\& item : ordered_dict) {\
      push_back(item.key(), item.value());\
    }\
  }\
\
  // JavaCPP OrderedDict<shared_ptr<Module>> ctor\
  explicit SequentialImpl(torch::OrderedDict<std::string, std::shared_ptr<Module>>\& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto\& item : ordered_dict) {\
      push_back(item.key(), item.value());\
    }\
  }\
\
  explicit SequentialImpl(torch::OrderedDict<std::string, std::shared_ptr<Module>>\&\& ordered_dict) {\
    modules_.reserve(ordered_dict.size());\
    for (auto\& item : ordered_dict) {\
      push_back(item.key(), std::move(item.value()));\
    }\
  }\
' "$seq"
    fi
    # ModuleDict modules_ public for Java
    local md="$API/nn/modules/container/moduledict.h"
    if [[ -f "$md" ]]; then
        sedinplace '/^ private:$/,/^  torch::OrderedDict<std::string, std::shared_ptr<Module>> modules_;$/s/^ private:$/ public:/' "$md" || true
    fi
}

patch_embedding_from_pretrained
patch_module_h
patch_any_sequential
echo "Patched headers under $ROOT"
