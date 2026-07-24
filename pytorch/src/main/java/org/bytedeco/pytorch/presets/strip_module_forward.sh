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
GEN_DIR="${1:-src/gen/java}"

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
MODULE_JAVA=$(find "$GEN_DIR" \( -path "*/org/bytedeco/pytorch/*/Module.java" -o -path "*/org/bytedeco/pytorch/Module.java" \) 2>/dev/null | head -1)
if [ -n "$MODULE_JAVA" ] && [ -f "$MODULE_JAVA" ]; then
    sed -i '' -E 's/@Virtual\(method="forward(T_[A-Za-z_]*)?"\)//g' "$MODULE_JAVA"
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
MODULE_JAVA=$(find "$GEN_DIR" \( -path "*/org/bytedeco/pytorch/*/Module.java" -o -path "*/org/bytedeco/pytorch/Module.java" \) 2>/dev/null | head -1)
if [ -n "$MODULE_JAVA" ] && [ -f "$MODULE_JAVA" ]; then
    sed -i '' 's/private native @ByVal @Name("forward_tensor")  Tensor _forward_tensor(/private native @ByVal @Name("forward_tensor") @Virtual(method="forward") Tensor _forward_tensor(/' "$MODULE_JAVA"
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
    local PRINTER_FQCN
    case "$PRINTER" in
      ModulePrinter|ModulePrinterShapeHelper) PRINTER_FQCN="org.bytedeco.pytorch.nn.$PRINTER" ;;
      *) PRINTER_FQCN="org.bytedeco.pytorch.$PRINTER" ;;
    esac
    # (a) our prior inject exists
    if grep -q "return ${PRINTER_FQCN}\\.format(this);" "$F" 2>/dev/null \
       || grep -q "return ${PRINTER}\\.format(this);" "$F" 2>/dev/null; then
        return
    fi
    # (b) the class already has a public String toString() declared
    if grep -qE '(^|\s)public String toString\(' "$F" 2>/dev/null; then
        return
    fi
    # (c) broken prior inject with empty FQCN
    if grep -q 'return \.format(this);' "$F" 2>/dev/null; then
        sed -i '' "s|return \\.format(this);|return ${PRINTER_FQCN}.format(this);|g" "$F"
        sed -i '' "s|{@link ${PRINTER}}|{@link ${PRINTER_FQCN}}|g" "$F" 2>/dev/null || true
        return
    fi
    # Anchor on the LAST `^}` (each of these classes has exactly one
    # top-level closing brace at column 1).
    local LAST_BRACE_LINE=$(grep -n '^}' "$F" | tail -1 | cut -d: -f1)
    [ -z "$LAST_BRACE_LINE" ] && return
    sed -i '' "${LAST_BRACE_LINE}i\\
\\
  /** Debug-friendly string representation, mirroring Python PyTorch's\\
   *  {@code print(...)} behavior. See {@link ${PRINTER_FQCN}}. */\\
  @Override public String toString() { return ${PRINTER_FQCN}.format(this); }
" "$F"
}

# Resolve a generated peer by simple class name under GEN_DIR (package-aware).
find_gen() {
    local name="$1"
    find "$GEN_DIR" -name "$name" 2>/dev/null | head -1
}

# Module / Tensor / container types.
MODULE_JAVA=$(find "$GEN_DIR" \( -path "*/org/bytedeco/pytorch/*/Module.java" -o -path "*/org/bytedeco/pytorch/Module.java" \) 2>/dev/null | head -1)
if [ -n "$MODULE_JAVA" ] && [ -f "$MODULE_JAVA" ]; then
    inject_tostring "$MODULE_JAVA" ModulePrinter
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
    FOUND=$(find_gen "$F")
    if [ -n "$FOUND" ] && [ -f "$FOUND" ]; then
        inject_tostring "$FOUND" "$P"
    fi
done

# Optimizer subclasses (Adam, AdamW, SGD, LBFGS, ...) — each
# has its own options struct. Inject toString into the common
# parent (Optimizer.java) and rely on OptimizerPrinter's
# reflection-based hyper-param extraction to handle every
# subclass uniformly.
OPT_JAVA=$(find_gen "Optimizer.java")
if [ -n "$OPT_JAVA" ] && [ -f "$OPT_JAVA" ] && ! grep -q 'OptimizerPrinter' "$OPT_JAVA"; then
    inject_tostring "$OPT_JAVA" OptimizerPrinter
fi

# Sampler subclasses (RandomSampler, BatchSizeSampler, ...).
# Inject toString into the common Sampler parent so the batch of
# known subclasses all inherit a sane print path.
SAMP_JAVA=$(find_gen "Sampler.java")
if [ -n "$SAMP_JAVA" ] && [ -f "$SAMP_JAVA" ] && ! grep -q 'SamplerPrinter' "$SAMP_JAVA"; then
    inject_tostring "$SAMP_JAVA" SamplerPrinter
fi

# Loss subclasses — register toString via LossPrinter on the
# common LossImplBase (if present) so MSELossImpl, NLLLossImpl,
# etc. all inherit it. Otherwise fall back to per-class injection.
for LOSS in MSELossImpl NLLLossImpl BCELossImpl BCEWithLogitsLossImpl L1LossImpl SmoothL1LossImpl; do
    LJ=$(find_gen "$LOSS.java")
    if [ -n "$LJ" ] && [ -f "$LJ" ] && ! grep -q 'LossPrinter' "$LJ"; then
        inject_tostring "$LJ" LossPrinter
    fi
done
# If a common Loss base exists, inject there too.
for BASE in LossImplBase; do
    BJ=$(find_gen "$BASE.java")
    if [ -n "$BJ" ] && [ -f "$BJ" ] && ! grep -q 'LossPrinter' "$BJ"; then
        inject_tostring "$BJ" LossPrinter
    fi
done
# 4) Restore convenience forward() wrappers on modules whose C++ forward
#    was renamed to forwardT_* / forwardASMoutput to avoid Java return-type
#    clashes with Module.forward(Tensor...).
#
#    Java forbids a subclass method with the same name+arity but a different
#    return type than a parent method. Module already exposes
#        Tensor forward(Tensor)
#        Tensor forward(Tensor, Tensor)
#        Tensor forward(Tensor, Tensor, Tensor)
#    so RNN/GRU/LSTM/MHA(3-arg)/AdaptiveLogSoftmax cannot reclaim the plain
#    name "forward" for their tuple/ASMoutput overloads. What we CAN do:
#      - MultiheadAttentionImpl 7-arg (unique arity) -> public forward(...)
#      - LSTM/LSTMCell (Tensor, Optional) if no Module Tensor clash -> forward
#    The forwardT_* / forwardASMoutput natives stay as the primary API (and
#    match Module's own tuple-forward naming). Wrappers are idempotent.
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PYW' "$GEN_DIR"
import re, sys
from pathlib import Path
root = Path(sys.argv[1])

def find_cls(name):
    hits = list(root.rglob(name + ".java"))
    return hits[0] if hits else None

def inject_after_natives(path, needle, block, tag):
    t = path.read_text()
    if tag in t:
        return False
    idx = t.rfind(needle)
    if idx < 0:
        return False
    semi = t.find(';', idx)
    if semi < 0:
        return False
    insert_at = semi + 1
    if insert_at < len(t) and t[insert_at] == '\n':
        insert_at += 1
    path.write_text(t[:insert_at] + "\n" + block + t[insert_at:])
    return True

# MultiheadAttentionImpl: 7-arg has no Module.forward clash
mha = find_cls("MultiheadAttentionImpl")
if mha:
    block = (
        "  /** Convenience alias for {@link #forwardT_TensorTensor_T(Tensor, Tensor, Tensor, Tensor, boolean, Tensor, boolean)}.\n"
        "   *  Restores the original C++ {@code forward(...)} name for the full 7-arg form\n"
        "   *  (unique arity - does not clash with {@link org.bytedeco.pytorch.nn.Module#forward}). */\n"
        "  public @ByVal T_TensorTensor_T forward(\n"
        "        @Const @ByRef Tensor query,\n"
        "        @Const @ByRef Tensor key,\n"
        "        @Const @ByRef Tensor value,\n"
        "        @Const @ByRef Tensor key_padding_mask,\n"
        "        @Cast(\"bool\") boolean need_weights,\n"
        "        @Const @ByRef Tensor attn_mask,\n"
        "        @Cast(\"bool\") boolean average_attn_weights) {\n"
        "    return forwardT_TensorTensor_T(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights);\n"
        "  }\n"
    )
    ok = inject_after_natives(
        mha,
        'boolean average_attn_weights/*=true*/);',
        block,
        'Restores the original C++')
    print("MHA 7-arg wrapper: %s" % ok)

# LSTMImpl / LSTMCellImpl: (Tensor, Optional) is distinct from Module.forward(Tensor,Tensor)
for cls, ret, native in [
    ("LSTMImpl", "T_TensorT_TensorTensor_T_T", "forwardT_TensorT_TensorTensor_T_T"),
    ("LSTMCellImpl", "T_TensorTensor_T", "forwardT_TensorTensor_T"),
]:
    f = find_cls(cls)
    if not f:
        print("%s: not found" % cls); continue
    t = f.read_text()
    if "Optional-hx arity does not clash" in t:
        print("%s: already has wrappers" % cls); continue
    block = (
        "  /** Convenience alias for {@link #%s(Tensor, T_TensorTensor_TOptional)}.\n"
        "   *  Optional-hx arity does not clash with Module.forward(Tensor, Tensor). */\n"
        "  public @ByVal %s forward(\n"
        "        @Const @ByRef Tensor input,\n"
        "        @Optional T_TensorTensor_T hx_opt) {\n"
        "    return %s(input, hx_opt);\n"
        "  }\n"
    ) % (native, ret, native)
    needle = '@Optional T_TensorTensor_T hx_opt/*={}*/);'
    if needle not in t:
        m = re.search(r'@Optional T_TensorTensor_T hx_opt[^;]*\);', t)
        if m:
            needle = m.group(0)
        else:
            print("%s: no Optional hx anchor" % cls); continue
    ok = inject_after_natives(f, needle, block, "Optional-hx arity does not clash")
    print("%s Optional-hx wrapper: %s" % (cls, ok))

# Annotate renames with a one-line JavaDoc if missing
for cls in ["RNNImpl", "GRUImpl", "MultiheadAttentionImpl", "AdaptiveLogSoftmaxWithLossImpl", "LSTMImpl", "LSTMCellImpl"]:
    f = find_cls(cls)
    if not f:
        continue
    t = f.read_text()
    if "C++ {@code forward} is exposed under a distinct Java name" in t:
        print("%s: doc already present" % cls); continue
    t2, n = re.subn(
        r'(  public native @ByVal @Name\("forward"\))',
        r'  /** Note: C++ {@code forward} is exposed under a distinct Java name because\n'
        r'   *  {@link org.bytedeco.pytorch.nn.Module} already owns Tensor-returning\n'
        r'   *  {@code forward(...)} overloads of the same arity (Java forbids same\n'
        r'   *  name+arity with a different return type). Call the method below, or\n'
        r'   *  the matching forwardT_* / forwardASMoutput name on Module. */\n\1',
        t, count=1)
    if n:
        f.write_text(t2)
        print("%s: added clash JavaDoc" % cls)
    else:
        print("%s: no native forward to annotate" % cls)
print("done wrappers")
PYW
fi

# 3) Package-relocation FQCN fixup: Module/helpers live in split packages.
#    After inject_tostring and parser javaText, rewrite bare helper names and
#    ensure moduleObjectId is public for cross-package ModuleAsHelper access.
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PY' "$GEN_DIR"
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
reps = [
    (r'(?<![\w.])ModuleAsHelper\.', 'org.bytedeco.pytorch.nn.ModuleAsHelper.'),
    (r'(?<![\w.])ModulePrinter\.', 'org.bytedeco.pytorch.nn.ModulePrinter.'),
    (r'(?<![\w.])TensorPrinter\.', 'org.bytedeco.pytorch.TensorPrinter.'),
    (r'(?<![\w.])OptimizerPrinter\.', 'org.bytedeco.pytorch.OptimizerPrinter.'),
    (r'(?<![\w.])LossPrinter\.', 'org.bytedeco.pytorch.LossPrinter.'),
    (r'(?<![\w.])LossReductionPrinter\.', 'org.bytedeco.pytorch.LossReductionPrinter.'),
    (r'(?<![\w.])DataLoaderConfigPrinter\.', 'org.bytedeco.pytorch.DataLoaderConfigPrinter.'),
    (r'(?<![\w.])FullDataLoaderOptionsPrinter\.', 'org.bytedeco.pytorch.FullDataLoaderOptionsPrinter.'),
    (r'(?<![\w.])SamplerPrinter\.', 'org.bytedeco.pytorch.SamplerPrinter.'),
    (r'(?<![\w.])TensorDatasetPrinter\.', 'org.bytedeco.pytorch.TensorDatasetPrinter.'),
    (r'(?<!public )native @Name\("javacpp_module_object_id"\) @Cast\("size_t"\) long moduleObjectId\(\);',
     'public native @Name("javacpp_module_object_id") @Cast("size_t") long moduleObjectId();'),
]
n = 0
for path in root.rglob("*.java"):
    t = path.read_text()
    orig = t
    for pat, rep in reps:
        t = re.sub(pat, rep, t)
    t = t.replace('org.bytedeco.pytorch.nn.org.bytedeco.pytorch.nn.', 'org.bytedeco.pytorch.nn.')
    t = t.replace('org.bytedeco.pytorch.org.bytedeco.pytorch.', 'org.bytedeco.pytorch.')
    t = t.replace('public public native', 'public native')
    if t != orig:
        path.write_text(t)
        n += 1
print(f"FQCN fixup touched {n} files under {root}")
PY
fi
