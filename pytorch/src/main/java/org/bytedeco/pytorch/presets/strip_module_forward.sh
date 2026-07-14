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

# JavaCPP parser bug (libtorch 2.12+): when ArrayRef<at::Dimname> is
# encountered in a struct from the at::namedinference namespace
# (TensorName / TensorNames), the parser mis-categorizes the typed
# ArrayRef class DimnameArrayRef in two distinct ways:
#
#   1) The @Name annotation on the class itself gets set to
#      @Name("DimnameArrayRef") (just the Java class name) instead of
#      the proper C++ type @Name("c10::ArrayRef<at::Dimname>"). The
#      JNI generator then embeds the bare Java class name in the
#      bridge code (e.g. `DimnameArrayRef* rptr = new DimnameArrayRef[...]`)
#      which fails to compile because there's no such C++ type.
#   2) When the same ArrayRef<at::Dimname> appears as a parameter type
#      in the constructors, the parser emits
#      `c10::ArrayRef<at::Dimname>` directly instead of the typed
#      DimnameArrayRef class. After the c10:: strip below reduces
#      that to `ArrayRef<at.Dimname>`, the result is invalid Java.
#   3) The class's own native constructors take `@Const Dimname` (a
#      Java class) as the data element. The JNI generator doesn't
#      apply the @Name mapping in the bridge code when this is the
#      case, producing `DimnameArrayRef* rptr = ...` instead of the
#      @Name-mapped `c10::ArrayRef<at::Dimname>* rptr = ...`. Rewrite
#      to `@Cast("const at::Dimname*") IntPointer` (primitive pointer)
#      which makes the JNI generator apply @Name correctly.
# All three are needed for libtorch 2.12+ with the homebrew libtorch
# headers. Order matters: apply the @Name fix LAST so the c10::→
# DimnameArrayRef sed doesn't accidentally rewrite it back.
if [ -f "$GEN_DIR/DimnameArrayRef.java" ]; then
    sed -i '' -E '
        s/public DimnameArrayRef\(@Const Dimname data, @Cast\("size_t"\) long length\)/public DimnameArrayRef(IntPointer data, long length)/g;
        s/private native void allocate\(@Const Dimname data, @Cast\("size_t"\) long length\)/private native void allocate(@Cast("const at::Dimname*") IntPointer data, @Cast("size_t") long length)/g;
        s/public DimnameArrayRef\(@Const Dimname begin, @Const Dimname end\)/public DimnameArrayRef(IntPointer begin, IntPointer end)/g;
        s/private native void allocate\(@Const Dimname begin, @Const Dimname end\)/private native void allocate(@Cast("const at::Dimname*") IntPointer begin, @Cast("const at::Dimname*") IntPointer end)/g;
    ' "$GEN_DIR/DimnameArrayRef.java"
fi
for f in $(grep -rl 'c10::ArrayRef<at::Dimname>' "$GEN_DIR" 2>/dev/null); do
    sed -i '' 's/c10::ArrayRef<at::Dimname>/DimnameArrayRef/g; s/c10::HeaderOnlyArrayRef<at::Dimname>/DimnameHeaderOnlyArrayRef/g' "$f"
done
# Apply the @Name fix LAST so the c10::→DimnameArrayRef sed above doesn't
# accidentally rewrite the just-fixed @Name back to bare DimnameArrayRef.
if [ -f "$GEN_DIR/DimnameArrayRef.java" ]; then
    sed -i '' 's/@Name("DimnameArrayRef")/@Name("c10::ArrayRef<at::Dimname>")/g' "$GEN_DIR/DimnameArrayRef.java"
fi

# JavaCPP parser bug (libtorch 2.12+): when ArrayRef<at::Dimname> is
# encountered in a struct from the at::namedinference namespace
# (TensorName / TensorNames), the parser mis-categorizes the typed
# ArrayRef class DimnameArrayRef in two distinct ways:
#
#   1) The @Name annotation on the class itself gets set to
#      @Name("DimnameArrayRef") (just the Java class name) instead of
#      the proper C++ type @Name("c10::ArrayRef<at::Dimname>"). The
#      JNI generator then embeds the bare Java class name in the
#      bridge code (e.g. `DimnameArrayRef* rptr = new DimnameArrayRef[...]`)
#      which fails to compile because there's no such C++ type.
#   2) When the same ArrayRef<at::Dimname> appears as a parameter type
#      in the constructors, the parser emits
#      `c10::ArrayRef<at::Dimname>` directly instead of the typed
#      DimnameArrayRef class. After the c10:: strip below reduces
#      that to `ArrayRef<at.Dimname>`, the result is invalid Java.
#   3) The class's own native constructors take `@Const Dimname` (a
#      Java class) as the data element. The JNI generator doesn't
#      apply the @Name mapping in the bridge code when this is the
#      case, producing `DimnameArrayRef* rptr = ...` instead of the
#      @Name-mapped `c10::ArrayRef<at::Dimname>* rptr = ...`. Rewrite
#      to `@Cast("const at::Dimname*") IntPointer` (primitive pointer)
#      which makes the JNI generator apply @Name correctly.
# All three are needed for libtorch 2.12+ with the homebrew libtorch
# headers. Order matters: apply the constructor fix and @Name fix
# BEFORE the @Name-rewriting sed below would otherwise revert it.
if [ -f "$GEN_DIR/DimnameArrayRef.java" ]; then
    sed -i '' -E '
        s/public DimnameArrayRef\(@Const Dimname data, @Cast\("size_t"\) long length\)/public DimnameArrayRef(IntPointer data, long length)/g;
        s/private native void allocate\(@Const Dimname data, @Cast\("size_t"\) long length\)/private native void allocate(@Cast("const at::Dimname*") IntPointer data, @Cast("size_t") long length)/g;
        s/public DimnameArrayRef\(@Const Dimname begin, @Const Dimname end\)/public DimnameArrayRef(IntPointer begin, IntPointer end)/g;
        s/private native void allocate\(@Const Dimname begin, @Const Dimname end\)/private native void allocate(@Cast("const at::Dimname*") IntPointer begin, @Cast("const at::Dimname*") IntPointer end)/g;
    ' "$GEN_DIR/DimnameArrayRef.java"
fi
for f in $(grep -rl 'c10::ArrayRef<at::Dimname>' "$GEN_DIR" 2>/dev/null); do
    sed -i '' 's/c10::ArrayRef<at::Dimname>/DimnameArrayRef/g; s/c10::HeaderOnlyArrayRef<at::Dimname>/DimnameHeaderOnlyArrayRef/g' "$f"
done
# Apply the @Name fix LAST so the previous sed doesn't accidentally
# rewrite the just-fixed @Name back to bare DimnameArrayRef.
if [ -f "$GEN_DIR/DimnameArrayRef.java" ]; then
    sed -i '' 's/@Name("DimnameArrayRef")/@Name("c10::ArrayRef<at::Dimname>")/g' "$GEN_DIR/DimnameArrayRef.java"
fi

# 1) Strip @Virtual(method="forward(T_[A-Za-z_]*)?") on Module.java.
# Removing these (multi-arg forward_tensorN @Virtuals) avoids the
# arity-mismatch crash where JavaCPP's vtable dispatch picks the wrong
# forward_tensorN overload on a built-in *Impl inside a Sequential.
if [ -f "$GEN_DIR/Module.java" ]; then
    sed -i '' -E 's/@Virtual\(method="forward(T_[A-Za-z_]*)?"\)//g' "$GEN_DIR/Module.java"
fi
# 1b) Re-add @Virtual(method="forward") to the 1-arg _forward_tensor ONLY.
# This is the callback the C++ AnyModuleHolder<Module>::forward path needs:
# when a user-defined Java subclass of Module (e.g. samples/example/
# TestSequentialPushBack.java's InputStem) is pushed into a SequentialImpl,
# C++ calls torch::nn::Module::forward_tensor on the module's C++ peer.
# forward_tensor is a C++ virtual (see the module.h patch in cppbuild.sh),
# so JavaCPP's @Virtual trampoline (generated because subclasses defaults
# to true) overrides it and, via method="forward", calls back into the Java
# `forward(Tensor)` shim - which dispatches to the user's override via
# ModuleAsHelper.hasForwardOverride. Without this, the call hits the base
# Module::forward_tensor that throws "not implemented for <Module>".
# Only the 1-arg overload is re-enabled; multi-arg forward_tensorN stay
# stripped to keep the arity-mismatch protection for built-in *Impl layers.
if [ -f "$GEN_DIR/Module.java" ]; then
    sed -i '' 's/private native @ByVal @Name("forward_tensor")  Tensor _forward_tensor(/private native @ByVal @Name("forward_tensor") @Virtual(method="forward") Tensor _forward_tensor(/' "$GEN_DIR/Module.java"
fi

# 1c) libtorch 2.12+ removed the std::vector<Dimname>&& overloads of
# NamedTensorMeta::set_names / NamedTensorMeta ctor / internal_set_names_inplace
# in favor of DimnameList (= ArrayRef<Dimname>). The parser still emits
# @StdVector Dimname overloads whose JNI passes a VectorAdapter<Dimname>,
# which matches NEITHER the DimnameList nor the std::vector<Dimname>&&
# C++ candidate -> "no matching member function / constructor" compile
# errors. Delete those broken overloads; the DimnameList (ArrayRef) overloads
# (which compile) remain, and a DimnameVector still binds to them via the
# DimnameArrayRef(DimnameVector) constructor. Files affected:
# NamedTensorMeta.java, global/torch.java.
for f in "$GEN_DIR/NamedTensorMeta.java" "$GEN_DIR/global/torch.java"; do
    [ -f "$f" ] || continue
    sed -i '' \
        -e '/public NamedTensorMeta(HAS_NON_WILDCARD arg0, @StdVector Dimname names)/d' \
        -e '/public NamedTensorMeta(@Cast("at::NamedTensorMeta::HAS_NON_WILDCARD") int arg0, @StdVector Dimname names)/d' \
        -e '/private native void allocate(HAS_NON_WILDCARD arg0, @StdVector Dimname names)/d' \
        -e '/private native void allocate(@Cast("at::NamedTensorMeta::HAS_NON_WILDCARD") int arg0, @StdVector Dimname names)/d' \
        -e '/public native void set_names(HAS_NON_WILDCARD arg0, @StdVector Dimname new_names)/d' \
        -e '/public native void set_names(@Cast("at::NamedTensorMeta::HAS_NON_WILDCARD") int arg0, @StdVector Dimname new_names)/d' \
        -e '/internal_set_names_inplace.*@StdVector Dimname/d' \
        "$f"
done

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