#!/bin/bash
# JavaCPP fix (matches the pom.xml sed that runs this script after the
# parser regenerates Module.java / Tensor.java etc.):
#
#   1) Strip @Virtual(method="forward" | method="forwardT_*") from
#      src/gen/java/.../Module.java so JavaCPP does NOT do Java-side
#      virtual dispatch on `forward` overloads. Without this, JavaCPP's
#      vtable picks the wrong arity (e.g. forward_tensor2 on a DropoutImpl
#      inside an nn::Sequential) and TORCH_CHECK(false) crashes.
#
#   Note: Module's 1-3 arg `public Tensor forward(Tensor...)` Java shims
#   (and the corresponding `_forward_tensorN` private natives) are kept
#   intact so `dict.get("name").forward(x)` works on Module-typed
#   references for Tensor-returning layers (LinearImpl, FlattenImpl,
#   ReLUImpl, ...). The 6 RNN-family / AdaptiveLogSoftmaxWithLoss layers
#      declared in src/main/java/.../presets/torch.java
#      (RNNImpl, GRUImpl, LSTMImpl, LSTMCellImpl, MultiheadAttentionImpl,
#      AdaptiveLogSoftmaxWithLossImpl) keep their `forwardT_...` /
#      `forwardASMoutput` Java method names to avoid the Java compile error
#      of same-name + same-arity + different-return-type across the
#      Module -> *Impl hierarchy.
#
#   2) Inject a debug-friendly @Override toString() at the end of
#      Module.java / Tensor.java etc. (delegating to ModulePrinter /
#      TensorPrinter in src/main/java/...). This mirrors Python
#      PyTorch's `print(model)` / `print(tensor)` behavior so the
#      console shows shape / dtype / device / values / child modules
#      instead of the raw pointer address. Injection is idempotent:
#      pre-existing toString() overrides (e.g. from a previous build
#      that didn't get reset) are detected and the second pass is a
#      no-op.
#
# Captured-member-pointer InvokeForward dispatch in
# any_module_holder.h is what makes nn::Sequential -> *Impl::forward work
# end-to-end at runtime.

set -e
GEN_DIR="${1:-src/gen/java/org/bytedeco/pytorch}"

# 1) Strip @Virtual(method="forward(T_[A-Za-z_]*)?") on Module.java.
if [ -f "$GEN_DIR/Module.java" ]; then
    sed -i '' -E 's/@Virtual\(method="forward(T_[A-Za-z_]*)?"\)//g' "$GEN_DIR/Module.java"
fi

# 2) Inject toString() override at the end of each target file.
# Idempotent: skip injection if either (a) our prior inject exists, OR
# (b) the file already declares an explicit toString() (e.g. JavaCPP
# generates one for std::vector-shaped Pointer subclasses that
# returns Arrays.toString(get())). Trying to define a second one
# collides on the same name+arity+different-body.
inject_tostring() {
    local F="$1"
    local PRINTER="$2"
    # (a) our prior inject exists
    if grep -q "return ${PRINTER}\.format(this);" "$F" 2>/dev/null; then
        return
    fi
    # (b) the class already has a public String toString() declared
    if grep -qE '(^|\s)public String toString\(' "$F" 2>/dev/null; then
        return
    fi
    # Anchor on the LAST `^}` (each of these classes has exactly one
    # top-level closing brace at column 1).
    local LAST_BRACE_LINE=$(grep -n '^}' "$F" | tail -1 | cut -d: -f1)
    [ -z "$LAST_BRACE_LINE" ] && return
    sed -i '' "${LAST_BRACE_LINE}i\\
\\
  /** Debug-friendly string representation, mirroring Python PyTorch's\\
   *  {@code print(...)} behavior. See {@link ${PRINTER}}. */\\
  @Override public String toString() { return ${PRINTER}.format(this); }
" "$F"
}

# Module / Tensor / container types.
if [ -f "$GEN_DIR/Module.java" ]; then
    inject_tostring "$GEN_DIR/Module.java" ModulePrinter
fi
for ENTRY in \
    "Tensor.java:TensorPrinter" \
    "TensorVector.java:TensorPrinter" \
    "ModuleListImpl.java:ModulePrinter" \
    "ModuleDictImpl.java:ModulePrinter" \
    "ParameterListImpl.java:ModulePrinter" \
    "ParameterDictImpl.java:ModulePrinter" \
    "TensorDataset.java:TensorDatasetPrinter" \
    "DataLoaderOptions.java:DataLoaderConfigPrinter" \
    "FullDataLoaderOptions.java:FullDataLoaderOptionsPrinter" \
    "LossReduction.java:LossReductionPrinter"
do
    F="${ENTRY%%:*}"
    P="${ENTRY##*:}"
    if [ -f "$GEN_DIR/$F" ]; then
        inject_tostring "$GEN_DIR/$F" "$P"
    fi
done

# Optimizer subclasses (Adam, AdamW, SGD, LBFGS, ...) — each
# has its own options struct. Inject toString into the common
# parent (Optimizer.java) and rely on OptimizerPrinter's
# reflection-based hyper-param extraction to handle every
# subclass uniformly.
if [ -f "$GEN_DIR/Optimizer.java" ] && ! grep -q 'OptimizerPrinter' "$GEN_DIR/Optimizer.java"; then
    inject_tostring "$GEN_DIR/Optimizer.java" OptimizerPrinter
fi

# Sampler subclasses (RandomSampler, BatchSizeSampler, ...).
# Inject toString into the common Sampler parent so the batch of
# known subclasses all inherit a sane print path.
if [ -f "$GEN_DIR/Sampler.java" ] && ! grep -q 'SamplerPrinter' "$GEN_DIR/Sampler.java"; then
    inject_tostring "$GEN_DIR/Sampler.java" SamplerPrinter
fi

# Loss subclasses — register toString via LossPrinter on the
# common LossImplBase (if present) so MSELossImpl, NLLLossImpl,
# etc. all inherit it. Otherwise fall back to per-class injection.
for LOSS in MSELossImpl NLLLossImpl BCELossImpl BCEWithLogitsLossImpl L1LossImpl SmoothL1LossImpl; do
    if [ -f "$GEN_DIR/$LOSS.java" ] && ! grep -q 'LossPrinter' "$GEN_DIR/$LOSS.java"; then
        inject_tostring "$GEN_DIR/$LOSS.java" LossPrinter
    fi
done
# If a common Loss base exists, inject there too.
for BASE in LossImplBase; do
    if [ -f "$GEN_DIR/$BASE.java" ] && ! grep -q 'LossPrinter' "$GEN_DIR/$BASE.java"; then
        inject_tostring "$GEN_DIR/$BASE.java" LossPrinter
    fi
done