#!/usr/bin/env python3
"""
Relocate JavaCPP-generated peer classes for torch::{data,nn,jit,optim,serialize,
inductor,profiler,enumtype}, c10d (ProcessGroup/Store/…), and uncommon c10 types
into subpackages, and strip the "Java" prefix from javacpp:: adapter types.

Why this exists
---------------
JavaCPP's Parser always writes top-level classes into a single ``target``
package (see Parser.java: targetHeader + targetDir). Fully-qualified
pointerTypes only affect *references*, not the package of the generated
file. To put peers under org.bytedeco.pytorch.{data,nn,jit,optim,serialize,
distributed,inductor,profiler,enumtype,c10} we must post-process after parse.

Rules
-----
1. data     – torch::data::… / javacpp Dataset adapters / DATA_NAMES
2. nn       – torch::nn::… / NN_NAMES / ModuleApply* helpers (not Jit*)
3. jit      – torch::jit::…  (c10 types stay root unless uncommon; JitModuleApplyFunction → jit)
4. optim    – torch::optim::… / OPTIM_NAMES
5. serialize– torch::serialize::… / SERIALIZE_NAMES
6. distributed – c10d::… / c10d::detail::… / ProcessGroup* / Store* / DIST_NAMES
                 (does NOT touch torch::distributed::rpc — that stays in .rpc)
7. inductor – torch::inductor::… / AOTI* / INDUCTOR_NAMES
8. profiler – torch::profiler::… / PROFILER_NAMES
9. enumtype – torch::enumtype::… / k* enum structs / *Mode/*PaddingMode helpers
10. c10     – uncommon c10::… types (NOT Tensor/Scalar/ScalarType/Device/Stream/…)
11. Classes whose @Name is "javacpp::Dataset<...>" etc. are renamed from
    JavaXxx -> Xxx in both the file and all references across the gen tree.
12. Manual helpers under src/main/java are also moved/rewritten.
13. Already-relocated files are left alone except for rename/import rewrites.
14. Preserved subpackages (cuda/nccl/gloo/rpc/onnx/global) are never collapsed.

Usage
-----
  python3 scripts/relocate_packages.py [--gen-dir DIR] [--main-dir DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT_PKG = "org.bytedeco.pytorch"
DATA_PKG = f"{ROOT_PKG}.data"
NN_PKG = f"{ROOT_PKG}.nn"
JIT_PKG = f"{ROOT_PKG}.jit"
OPTIM_PKG = f"{ROOT_PKG}.optim"
SERIALIZE_PKG = f"{ROOT_PKG}.serialize"
DISTRIBUTED_PKG = f"{ROOT_PKG}.distributed"
INDUCTOR_PKG = f"{ROOT_PKG}.inductor"
PROFILER_PKG = f"{ROOT_PKG}.profiler"
ENUMTYPE_PKG = f"{ROOT_PKG}.enumtype"
C10_PKG = f"{ROOT_PKG}.c10"
AUTOGRAD_PKG = f"{ROOT_PKG}.autograd"

ALL_SUBPKGS = (
    DATA_PKG, NN_PKG, JIT_PKG, OPTIM_PKG, SERIALIZE_PKG,
    DISTRIBUTED_PKG, INDUCTOR_PKG, PROFILER_PKG,
    ENUMTYPE_PKG, C10_PKG, AUTOGRAD_PKG,
)

# Adapter class renames: JavaXxx -> Xxx (only for javacpp:: Dataset/DataLoader adapters)
JAVA_PREFIX_RENAMES: Dict[str, str] = {
    "JavaDataset": "Dataset",
    "JavaDatasetBase": "DatasetBase",
    "JavaBatchDataset": "BatchDataset",
    "JavaTensorDataset": "JavaTensorDataset",  # keep: would clash with torch TensorDataset
    "JavaTensorDatasetBase": "JavaTensorDatasetBase",
    "JavaTensorBatchDataset": "JavaTensorBatchDataset",
    "JavaStreamDataset": "StreamDataset",
    "JavaStreamBatchDataset": "StreamBatchDataset",
    "JavaStreamTensorDataset": "StreamTensorDataset",
    "JavaStreamTensorBatchDataset": "StreamTensorBatchDataset",
    "JavaStatefulDataset": "StatefulDataset",
    "JavaStatefulDatasetBase": "StatefulDatasetBase",
    "JavaStatefulBatchDataset": "StatefulBatchDataset",
    "JavaStatefulTensorDataset": "StatefulTensorDataset",
    "JavaStatefulTensorDatasetBase": "StatefulTensorDatasetBase",
    "JavaStatefulTensorBatchDataset": "StatefulTensorBatchDataset",
    "JavaRandomDataLoader": "RandomDataLoader",
    "JavaRandomDataLoaderBase": "RandomDataLoaderBase",
    "JavaRandomTensorDataLoader": "RandomTensorDataLoader",
    "JavaRandomTensorDataLoaderBase": "RandomTensorDataLoaderBase",
    "JavaSequentialDataLoader": "SequentialDataLoader",
    "JavaSequentialDataLoaderBase": "SequentialDataLoaderBase",
    "JavaSequentialTensorDataLoader": "SequentialTensorDataLoader",
    "JavaSequentialTensorDataLoaderBase": "SequentialTensorDataLoaderBase",
    "JavaDistributedRandomDataLoader": "DistributedRandomDataLoader",
    "JavaDistributedRandomDataLoaderBase": "DistributedRandomDataLoaderBase",
    "JavaDistributedRandomTensorDataLoader": "DistributedRandomTensorDataLoader",
    "JavaDistributedRandomTensorDataLoaderBase": "DistributedRandomTensorDataLoaderBase",
    "JavaDistributedSequentialDataLoader": "DistributedSequentialDataLoader",
    "JavaDistributedSequentialDataLoaderBase": "DistributedSequentialDataLoaderBase",
    "JavaDistributedSequentialTensorDataLoader": "DistributedSequentialTensorDataLoader",
    "JavaDistributedSequentialTensorDataLoaderBase": "DistributedSequentialTensorDataLoaderBase",
    "JavaStreamDataLoader": "StreamDataLoader",
    "JavaStreamDataLoaderBase": "StreamDataLoaderBase",
    "JavaStreamTensorDataLoader": "StreamTensorDataLoader",
    "JavaStreamTensorDataLoaderBase": "StreamTensorDataLoaderBase",
    "JavaStatefulDataLoader": "StatefulDataLoader",
    "JavaStatefulDataLoaderBase": "StatefulDataLoaderBase",
    "JavaStatefulTensorDataLoader": "StatefulTensorDataLoader",
    "JavaStatefulTensorDataLoaderBase": "StatefulTensorDataLoaderBase",
}

# Drop identity renames (kept above only as documentation of intentional keeps)
JAVA_PREFIX_RENAMES = {k: v for k, v in JAVA_PREFIX_RENAMES.items() if k != v}

# Simple-name allowlists for peers that don't carry a torch::data / torch::nn @Name
# (or where @Name is mangled / missing). Keep these tight to avoid false positives.
DATA_NAMES: Set[str] = {
    "BatchSize", "BatchSizeOptional", "BatchSizeSampler",
    "CustomBatchRequest",
    "DataLoaderOptions", "FullDataLoaderOptions",
    "Example", "ExampleCollation", "ExampleIterator", "ExampleOptional",
    "ExampleVector", "ExampleVectorIterator", "ExampleVectorOptional",
    "ExampleStack",
    "MNIST", "MNISTBatchDataset", "MNISTDataset", "MNISTMapBatchDataset",
    "MNISTMapDataset", "MNISTRandomDataLoader", "MNISTRandomDataLoaderBase",
    "RandomSampler", "Sampler", "SequentialSampler", "StreamSampler",
    "DistributedRandomSampler", "DistributedSampler", "DistributedSequentialSampler",
    "TensorExample", "TensorExampleCollation", "TensorExampleIterator",
    "TensorExampleOptional", "TensorExampleVector", "TensorExampleVectorIterator",
    "TensorExampleVectorOptional", "TensorExampleStack",
    "TensorDataset", "TensorDatasetBase", "TensorBatchDataset",
    "ChunkDataReader", "ChunkTensorDataReader", "ChunkDatasetOptions",
    "ChunkRecordIterator",
    "WorkerException",
    # transforms
    "BatchTransform", "Transform", "BatchLambda", "Lambda",
    "TensorTransform", "TensorLambda", "Normalize",
    "Collation", "Stack",
    "ExampleBatchTransform", "ExampleTransform", "ExampleBatchLambda", "ExampleLambda",
    "TensorExampleBatchTransform", "TensorExampleTransform",
}

NN_NAME_SUFFIXES = (
    "Impl", "ImplBase", "ImplBaseBase", "ImplCloneable",
    "Options", "FuncOptions",
    "Padding",  # Conv*Padding
)

NN_NAMES: Set[str] = {
    "Module", "AnyModule", "AnyModuleVector", "AnyValue",
    "Cloneable", "ModuleHolder",
    "ModuleDictImpl", "ModuleDictImplCloneable",
    "ModuleListImpl", "ModuleListImplCloneable",
    "SequentialImpl", "SequentialImplCloneable",
    "ParameterDictImpl", "ParameterDictImplCloneable",
    "ParameterListImpl", "ParameterListImplCloneable",
    "NamedAnyModule", "Functional", "FunctionalImpl",
    # Nonlinearity / FanModeType / *Padding* / LossReduction live in .enumtype
    # (std::variant over torch::enumtype::k*).
    "module_iterator", "module_list",
    "named_module_iterator", "named_module_list",
    "SharedModuleVector",
    "StringAnyModuleDict", "StringAnyModuleDictItem", "StringAnyModuleDictItemVector",
    "StringAnyModulePair", "StringAnyModuleVector",
    "StringSharedModuleDict", "StringSharedModuleDictItem",
    "StringSharedModuleDictItemVector", "StringSharedModulePair", "StringSharedModuleVector",
    # hand-written FunctionPointer adapters for Module::apply
    "ModuleApplyFunction", "NamedModuleApplyFunction",
    "SharedModuleApplyFunction", "NamedSharedModuleApplyFunction",
}

OPTIM_NAMES: Set[str] = {
    "Optimizer", "OptimizerOptions", "OptimizerParamState", "OptimizerParamGroup",
    "OptimizerParamGroupVector",
    "Adagrad", "AdagradOptions", "AdagradParamState",
    "Adam", "AdamOptions", "AdamParamState",
    "AdamW", "AdamWOptions", "AdamWParamState",
    "LBFGS", "LBFGSOptions", "LBFGSParamState",
    "RMSprop", "RMSpropOptions", "RMSpropParamState",
    "SGD", "SGDOptions", "SGDParamState",
    "LRScheduler", "StepLR", "ReduceLROnPlateauScheduler",
    # template specializations
    "OptimizerCloneableAdagradOptions", "OptimizerCloneableAdagradParamState",
    "OptimizerCloneableAdamOptions", "OptimizerCloneableAdamParamState",
    "OptimizerCloneableAdamWOptions", "OptimizerCloneableAdamWParamState",
    "OptimizerCloneableLBFGSOptions", "OptimizerCloneableLBFGSParamState",
    "OptimizerCloneableRMSpropOptions", "OptimizerCloneableRMSpropParamState",
    "OptimizerCloneableSGDOptions", "OptimizerCloneableSGDParamState",
}

SERIALIZE_NAMES: Set[str] = {
    "InputArchive", "OutputArchive",
}

# c10d ProcessGroup / Store / collective options / DDP helpers.
# Does NOT include torch::distributed::rpc (owned by torch_rpc preset → .rpc).
DISTRIBUTED_NAMES: Set[str] = {
    # ProcessGroup / Backend
    "ProcessGroup", "ProcessGroupGloo", "ProcessGroupStatus",
    "ProcessGroupCppCommHookInterface",
    "Backend", "BackendOptional", "BackendOptionsOptional",
    "DistributedBackendOptions",
    # Store family
    "Store", "StoreTimeoutGuard",
    "FileStore", "HashStore", "PrefixStore",
    "TCPStore", "TCPStoreOptions",
    "TCPClient", "TCPServer", "SocketAddress",
    # Collectives / work
    "Work", "WorkInfo",
    "ReduceOp", "PreMulSumSupplement", "_SupplementBase",
    "AllToAllOptions", "AllgatherOptions", "AllreduceOptions",
    "AllreduceCoalescedOptions", "BarrierOptions", "BroadcastOptions",
    "GatherOptions", "ReduceOptions", "ReduceScatterOptions", "ScatterOptions",
    # DDP / logging
    "Reducer", "GradBucket", "BucketAccumulator",
    "CommHookInterface",
    "C10dLogger", "C10dLoggingData", "Logger", "LoggerOptional",
    "Timer",
}

INDUCTOR_NAMES: Set[str] = {
    "AOTIModelContainerRunner", "AOTIModelContainerRunnerCpu",
    "AOTIModelContainerRunnerCuda",
    "AOTIModelPackageLoader",
}

# torch::autograd peers (NOT torch::dynamo::autograd — those stay root).
AUTOGRAD_NAMES: Set[str] = {
    "AnomalyMetadata", "AnomalyMode", "AutogradContext",
    "DetectAnomalyGuard", "Edge", "EdgeVector",
    "ForwardADLevel", "ForwardGrad",
    "FunctionPostHook", "FunctionPostHookVector",
    "FunctionPreHook", "FunctionPreHookVector",
    "InputMetadata", "InputMetadataOptional", "InputMetadataOptionalVector",
    "Node", "NodeSet",
    "PostAccumulateGradHook", "SavedVariableHooks", "VariableInfo",
}

PROFILER_NAMES: Set[str] = {
    "ProfilerConfig", "ExperimentalConfig",
    "ActivityTypeSet", "FileLineFunc", "Result", "SaveNcclMetaConfig",
    # libkineto peer used by profiler bindings
    "ITraceActivity",
}

# torch::enumtype peers (kLinear, kReLU, GridSampleMode, …) plus std::variant
# wrappers over those enums (historically mapped as nn helpers).
ENUMTYPE_NAMES: Set[str] = {
    "kArea", "kBatchMean", "kBicubic", "kBilinear", "kBorder", "kCircular",
    "kConstant", "kConv1D", "kConv2D", "kConv3D",
    "kConvTranspose1D", "kConvTranspose2D", "kConvTranspose3D",
    "kFanIn", "kFanOut", "kGELU", "kGRU", "kLSTM", "kLeakyReLU", "kLinear",
    "kMax", "kMean", "kMish", "kNearest", "kNearestExact", "kNone",
    "kRNN_RELU", "kRNN_TANH", "kReLU", "kReflect", "kReflection", "kReplicate",
    "kSame", "kSiLU", "kSigmoid", "kSum", "kTanh", "kTrilinear", "kValid", "kZeros",
    # variant helper / pretty-print types generated alongside enums
    "GridSampleMode", "GridSamplePaddingMode", "InterpolateMode",
    "KLDivLossReduction", "RNNBaseMode", "RNNNonlinearity", "UpsampleMode",
    # std::variant wrappers over torch::enumtype (pointerTypes in torch.java)
    "Nonlinearity", "FanModeType", "LossReduction", "kLossReduction",
    "ConvPaddingMode", "Conv1dPadding", "Conv2dPadding", "Conv3dPadding",
    "EmbeddingBagMode", "PaddingMode", "TransformerActivation",
}

# Hot-path c10 types that must remain in org.bytedeco.pytorch (root).
# Everything else under c10:: (and uncommon c10 helpers) goes to .c10.
# Explicitly exclude Tensor / ScalarType family as requested.
C10_KEEP_ROOT: Set[str] = {
    # Tensor core
    "Tensor", "TensorOptional", "TensorVector", "TensorArrayRef",
    "TensorList", "TensorListIterator", "TensorElementReference",
    "TensorHeaderOnlyArrayRef", "TensorOptionalArrayRef",
    "TensorOptionalElementReference", "TensorOptionalHeaderOnlyArrayRef",
    "TensorOptionalList", "TensorOptionalListIterator",
    "TensorIndexArrayRef", "TensorIndexHeaderOnlyArrayRef",
    "TensorArgArrayRef", "TensorArgHeaderOnlyArrayRef",
    "TensorBaseMaybeOwned", "TensorMaybeOwned",
    "TensorTensorDict", "TensorTensorDictIterator",
    "TensorImpl", "TensorImplSet", "TensorImplVector",
    "UndefinedTensorImpl", "NestedTensorImpl", "QTensorImpl",
    "TensorOptions", "TensorType",
    "AbstractTensor",
    # Scalar / ScalarType
    "Scalar", "ScalarOptional", "ScalarArrayRef", "ScalarHeaderOnlyArrayRef",
    "ScalarType", "ScalarTypeOptional", "ScalarTypeVector",
    "ScalarTypeArrayRef", "ScalarTypeHeaderOnlyArrayRef",
    "ScalarTypeEnumerationType", "ScalarTypeType", "ScalarTypeTypePtr",
    # Device / Stream / Storage / Allocator / DataPtr
    "Device", "DeviceOptional", "DeviceType", "DeviceTypeOptional",
    "DeviceTypeSet", "DeviceVector", "DeviceIndex",
    "DeviceObjType", "DeviceObjTypePtr",
    "Stream", "StreamOptional", "StreamData3",
    "StreamObjType", "StreamObjTypePtr",
    "Storage", "StorageImpl", "StorageExtraMeta",
    "StorageType", "StorageTypePtr",
    "Allocator", "AllocatorOptional",
    "DataPtr", "DataPtrVector",
    # IValue / Type system (used heavily by jit + samples)
    "IValue", "IValueOptional", "IValueVector", "IValueOptionalVector",
    "IValueArrayRef", "IValueHeaderOnlyArrayRef",
    "Type", "TypePtr", "TypeArrayRef", "TypeHeaderOnlyArrayRef",
    "SharedType", "StrongTypePtr", "WeakTypePtr",
    "SingletonTypePtr", "ListType", "DictType", "TupleType",
    "OptionalType", "AnyType", "AnyTypePtr",
    "TensorType", "ClassType", "FunctionType", "InterfaceType",
    "EnumType", "NoneType", "NoneTypePtr",
    "IntType", "IntTypePtr", "FloatType", "FloatTypePtr",
    "BoolType", "BoolTypePtr", "StringType", "StringTypePtr",
    "NumberType", "NumberTypePtr", "ComplexType", "ComplexTypePtr",
    # Generators / RNG
    "Generator", "GeneratorOptional", "GeneratorImpl",
    "GeneratorType", "GeneratorTypePtr",
    # Grad / modes commonly used
    "NoGradMode", "NoGradGuard", "GradMode", "AutoGradMode",
    "InferenceMode", "AutoFwGradMode",
    # Common numeric / layout helpers used at API surface
    "Half", "BFloat16", "ComplexFloat", "ComplexDouble",
    "FloatComplex", "DoubleComplex", "HalfComplex",
    "IntArrayRef", "LongArrayRef", "SymInt", "SymIntArrayRef",
    "SymBool", "SymFloat", "SymNode",
    "Dimname", "DimnameList", "DimVector", "SymDimVector",
    "MemoryFormat", "Layout", "QScheme",
    "OptionalDeviceGuard", "OptionalStreamGuard",
    "DispatchKey", "DispatchKeySet", "DispatchKeyOptional",
    "ArrayRef", "OptionalArrayRef",
    # DDPLoggingData is c10:: not c10d:: — keep root (not distributed)
    "DDPLoggingData",
}

# Names that look like nn/data/… but must stay in root
ROOT_FORCE: Set[str] = {
    "JitModule", "NamedJitModule", "BuiltinModule",
    "ModuleInstanceInfo", "ModuleInstanceInfoOptional", "ModulePolicy",
    "DataPtr", "DataPtrVector",
    # AnomalyMetadata / InputMetadata* are torch::autograd → .autograd
    "InlinedCallStack", "InlinedCallStackOptional",
    "StackEntry", "StashTorchDispatchStackGuard",
    # DDPLoggingData is c10:: not c10d:: — keep root (not distributed)
    "DDPLoggingData",
    # ApproximateClock is c10, not profiler — but uncommon → c10 package
    # (handled by C10_KEEP_ROOT absence)
}

NAME_ATTR_RE = re.compile(
    r'@Name\s*\(\s*"([^"]+)"\s*\)'
)
CLASS_DECL_RE = re.compile(
    r'^\s*public\s+(?:static\s+)?(?:final\s+)?class\s+(\w+)',
    re.MULTILINE,
)
PACKAGE_RE = re.compile(
    r'^package\s+([\w.]+)\s*;',
    re.MULTILINE,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def classify_by_name_attr(text: str, simple: str) -> Optional[str]:
    """Return package kind or None based on @Name/@Namespace attributes.

    Explicit 'root' means: keep in org.bytedeco.pytorch even if the simple name
    looks like nn/data (e.g. hot-path c10:: types, at:: types).
    """
    if simple in ROOT_FORCE or simple in C10_KEEP_ROOT:
        return "root"
    names = NAME_ATTR_RE.findall(text)
    namespaces = re.findall(r'@Namespace\s*\(\s*"([^"]+)"\s*\)', text)
    tokens = names + namespaces
    for n in tokens:
        # data
        if "torch::data::" in n or n.startswith("javacpp::Dataset") \
                or n.startswith("javacpp::StreamDataset") \
                or n.startswith("javacpp::StatefulDataset") \
                or n == "torch::data":
            return "data"
        # nn
        if "torch::nn::" in n or n == "torch::nn":
            return "nn"
        # jit
        if "torch::jit::" in n or n == "torch::jit":
            return "jit"
        # optim
        if "torch::optim::" in n or n == "torch::optim":
            return "optim"
        # serialize (torch::serialize only; caffe2::serialize stays root)
        if "torch::serialize::" in n or n == "torch::serialize":
            return "serialize"
        # inductor
        if "torch::inductor::" in n or n == "torch::inductor":
            return "inductor"
        # autograd (torch::autograd only; torch::dynamo::autograd stays root)
        if n == "torch::autograd" or n.startswith("torch::autograd::"):
            # Guard: dynamo nested ns must not match the prefix above
            if "torch::dynamo::" in n:
                return "root"
            return "autograd"
        # profiler
        if "torch::profiler::" in n or n == "torch::profiler" \
                or n.startswith("torch::profiler"):
            return "profiler"
        # enumtype (kLinear, kReLU, …)
        if "torch::enumtype" in n or n == "torch::enumtype" \
                or n.startswith("torch::enumtype::"):
            return "enumtype"
        # distributed c10d (NOT torch::distributed::rpc — those live under .rpc)
        if n == "c10d" or n.startswith("c10d::") \
                or ("c10d::" in n and "torch::distributed::rpc" not in n):
            # Guard: torch::distributed::rpc must stay in .rpc (preserved)
            if "torch::distributed::rpc" in n:
                return "root"  # should already be under .rpc; don't reclassify
            return "distributed"
        # Uncommon c10 types → .c10; hot-path ones already returned root above.
        if n.startswith("c10::") or n.startswith("c10_") or n in ("c10",) \
                or n.startswith("c10_complex"):
            return "c10"
        # at:: core types stay in root (Generator, Tensor hooks, …).
        if n.startswith("at::") or n == "at":
            return "root"
        # caffe2::serialize is used both by torch::data chunk loaders
        # (ChunkRecordIterator → data) and by generic archive helpers (root).
        # Defer to simple-name classification instead of hard root.
        if n.startswith("caffe2::") or n == "caffe2":
            continue
        # libkineto peers used by the profiler bindings
        if n.startswith("libkineto"):
            return "profiler" if simple in PROFILER_NAMES else "root"
        # torch::distributed::rpc must never be reclassified into .distributed
        if "torch::distributed::rpc" in n:
            return "root"
    return None


def classify_by_simple_name(simple: str) -> Optional[str]:
    if simple in ROOT_FORCE or simple in C10_KEEP_ROOT:
        return "root"

    # Hand-written ModuleApply* → nn; JitModuleApplyFunction → jit
    if simple == "JitModuleApplyFunction":
        return "jit"
    if simple in (
        "ModuleApplyFunction", "NamedModuleApplyFunction",
        "SharedModuleApplyFunction", "NamedSharedModuleApplyFunction",
    ):
        return "nn"

    if simple in ENUMTYPE_NAMES or (
        simple.startswith("k") and len(simple) > 1 and simple[1].isupper()
    ):
        return "enumtype"

    if simple in DATA_NAMES or simple.startswith("Chunk") or simple.startswith("Java") \
            or simple.startswith("MNIST"):
        if simple.startswith("Java") or simple in DATA_NAMES \
                or simple.startswith("Chunk") or simple.startswith("MNIST"):
            return "data"

    if simple in OPTIM_NAMES or simple.startswith("OptimizerCloneable"):
        return "optim"

    if simple in SERIALIZE_NAMES:
        return "serialize"

    if simple in DISTRIBUTED_NAMES \
            or simple.startswith("ProcessGroup") \
            or (simple.endswith("Store") and simple not in ("IStore",)) \
            or simple in (
                "AllToAllOptions", "AllgatherOptions", "AllreduceOptions",
                "AllreduceCoalescedOptions", "BarrierOptions", "BroadcastOptions",
                "GatherOptions", "ReduceOptions", "ReduceScatterOptions",
                "ScatterOptions", "DistributedBackendOptions",
            ):
        return "distributed"

    if simple in INDUCTOR_NAMES or simple.startswith("AOTI"):
        return "inductor"

    if simple in AUTOGRAD_NAMES:
        return "autograd"

    if simple in PROFILER_NAMES:
        return "profiler"

    if simple in NN_NAMES:
        return "nn"
    if simple.endswith(NN_NAME_SUFFIXES):
        # Avoid data-side *Options that aren't nn
        if simple in ("ChunkDatasetOptions", "DataLoaderOptions", "FullDataLoaderOptions"):
            return "data"
        # Optimizer / c10d option class names
        if simple.startswith(("Adagrad", "Adam", "AdamW", "LBFGS", "RMSprop", "SGD",
                              "Optimizer")):
            return "optim"
        if simple in (
            "AllreduceOptions", "AllreduceCoalescedOptions", "AllToAllOptions",
            "AllgatherOptions", "BarrierOptions", "BroadcastOptions",
            "GatherOptions", "ReduceOptions", "ReduceScatterOptions",
            "ScatterOptions", "DistributedBackendOptions", "TCPStoreOptions",
        ):
            return "distributed"
        # ProfilerConfig / ExperimentalConfig end with Config not Options
        return "nn"
    return None


def classify_file(path: Path, text: str) -> Optional[str]:
    simple = path.stem
    by_attr = classify_by_name_attr(text, simple)
    if by_attr:
        return by_attr
    return classify_by_simple_name(simple)


def collect_java_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = []
    for p in root.rglob("*.java"):
        if not p.is_file():
            continue
        # Never rewrite presets / resources / module-info
        parts = set(p.parts)
        if "presets" in parts or "resources" in parts:
            continue
        if p.name == "module-info.java":
            continue
        files.append(p)
    return sorted(files)


def kind_to_pkg(kind: str) -> Optional[str]:
    return {
        "data": DATA_PKG,
        "nn": NN_PKG,
        "jit": JIT_PKG,
        "optim": OPTIM_PKG,
        "serialize": SERIALIZE_PKG,
        "distributed": DISTRIBUTED_PKG,
        "inductor": INDUCTOR_PKG,
        "profiler": PROFILER_PKG,
        "enumtype": ENUMTYPE_PKG,
        "c10": C10_PKG,
        "autograd": AUTOGRAD_PKG,
        "root": ROOT_PKG,
    }.get(kind)


def plan_moves(gen_root: Path, main_root: Optional[Path]) -> Tuple[Dict[Path, Tuple[str, Path]], Dict[str, str], List[Path]]:
    """
    Returns:
      moves: src_path -> (new_package, dest_path)  [dest may equal src if only rename]
      renames: old_simple -> new_simple
      deletes: leftover Java* adapter files superseded by already-correct non-prefixed peers
    """
    moves: Dict[Path, Tuple[str, Path]] = {}
    renames: Dict[str, str] = dict(JAVA_PREFIX_RENAMES)
    deletes: List[Path] = []

    roots = [gen_root]
    if main_root and main_root.exists():
        roots.append(main_root)

    # Pre-scan: simple names already present under each root (post-rename target)
    existing_by_root: Dict[Path, Set[str]] = {}
    for root in roots:
        names: Set[str] = set()
        for path in collect_java_files(root):
            names.add(path.stem)
        existing_by_root[root] = names

    # Existing module packages that this script must never collapse into root
    # (cuda/nccl/gloo/rpc/onnx/global are separate presets / targets).
    preserved_subpkgs = {
        f"{ROOT_PKG}.cuda",
        f"{ROOT_PKG}.nccl",
        f"{ROOT_PKG}.gloo",
        f"{ROOT_PKG}.rpc",
        f"{ROOT_PKG}.onnx",
        f"{ROOT_PKG}.global",
    }

    for root in roots:
        existing = existing_by_root.get(root, set())
        for path in collect_java_files(root):
            # Only consider files currently under ROOT_PKG path layout
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            # expect .../org/bytedeco/pytorch[/sub]/Name.java
            text = read_text(path)
            simple = path.stem
            new_simple = renames.get(simple, simple)
            pkg_kind = classify_file(path, text)

            # Leftover Java* adapter from a previous parse: non-prefixed peer already
            # exists (jp="" generation). Drop the stale Java* file instead of
            # overwriting the correct peer via rename.
            if simple != new_simple and new_simple in existing and (path.parent / f"{new_simple}.java").exists():
                deletes.append(path)
                continue

            # Package from directory layout (…/org/bytedeco/pytorch[/sub]/Name.java)
            parent_pkg_parts = parts[:-1]  # drop filename
            current_pkg = ".".join(parent_pkg_parts) if parent_pkg_parts else ROOT_PKG

            in_preserved = current_pkg in preserved_subpkgs or any(
                current_pkg.startswith(p + ".") for p in preserved_subpkgs
            )

            if in_preserved:
                # Exception: torch::inductor AOTI* peers may be emitted under
                # the cuda target (model_container_runner_cuda.h) but belong in
                # .inductor. Allow reclassification of those only.
                if current_pkg == f"{ROOT_PKG}.cuda" and (
                    simple.startswith("AOTI") or simple in INDUCTOR_NAMES
                ):
                    pkg_kind = "inductor"
                else:
                    # Leave cuda/nccl/gloo/rpc/onnx/global alone entirely.
                    # Especially: do not pull rpc types into .distributed.
                    continue

            if pkg_kind and pkg_kind != "root":
                new_pkg = kind_to_pkg(pkg_kind) or ROOT_PKG
            elif pkg_kind == "root":
                # Only force root when currently mis-placed under a managed subpkg.
                if current_pkg in ALL_SUBPKGS:
                    new_pkg = ROOT_PKG
                else:
                    new_pkg = current_pkg if current_pkg.startswith(ROOT_PKG) else ROOT_PKG
            else:
                new_pkg = current_pkg if current_pkg.startswith(ROOT_PKG) else ROOT_PKG

            # Already in the right package with the right simple name — nothing to do.
            if new_simple == simple and new_pkg == current_pkg:
                continue

            dest_dir = root.joinpath(*new_pkg.split("."))
            dest = dest_dir / f"{new_simple}.java"
            moves[path] = (new_pkg, dest)

    return moves, renames, deletes


def apply_renames_in_text(text: str, renames: Dict[str, str]) -> str:
    # Longer names first to avoid partial replacements
    for old, new in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        if old == new:
            continue
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)
    return text


def needed_star_imports(new_pkg: str, text: str) -> List[str]:
    """Star-imports other subpackages + root when short names are used."""
    extras: List[str] = []
    if new_pkg != ROOT_PKG:
        extras.append(f"import {ROOT_PKG}.*;")

    checks = [
        (DATA_PKG, r'\b(Example|MNIST|Sampler|DataLoaderOptions|ChunkDataset|TensorExample)\b'),
        (NN_PKG, r'\b(Module|LinearImpl|AnyModule|SequentialImpl|ModuleApplyFunction|SharedModuleApplyFunction)\b'),
        # Value is the main short name that breaks hand-written helpers (ValueMapper).
        (JIT_PKG, r'\b(JitModule|Graph|ScriptModule|TypePtr|JitModuleApplyFunction|JitObject|Value)\b'),
        (OPTIM_PKG, r'\b(Optimizer|SGD|Adam|AdamW|Adagrad|LBFGS|RMSprop|LRScheduler|StepLR)\b'),
        (SERIALIZE_PKG, r'\b(InputArchive|OutputArchive)\b'),
        (DISTRIBUTED_PKG, r'\b(ProcessGroup|Store|TCPStore|FileStore|HashStore|PrefixStore|Work|ReduceOp|Backend|Reducer)\b'),
        (INDUCTOR_PKG, r'\b(AOTIModelContainerRunner|AOTIModelContainerRunnerCuda|AOTIModelPackageLoader)\b'),
        (PROFILER_PKG, r'\b(ProfilerConfig|ExperimentalConfig|ActivityTypeSet|ITraceActivity)\b'),
        (ENUMTYPE_PKG, r'\b(kLinear|kReLU|kGELU|kNone|kMean|kSum|kBilinear|kNearest|GridSampleMode|InterpolateMode)\b'),
        # Obj is c10::ivalue::Object — critical for JitObject / ObjLoader.
        (C10_PKG, r'\b(AliasInfo|FunctionSchema|OperatorHandle|Dispatcher|SymNode|Future|GenericDict|GenericList|Obj)\b'),
        (AUTOGRAD_PKG, r'\b(Edge|Node|AnomalyMode|AnomalyMetadata|AutogradContext|FunctionPreHook|FunctionPostHook|InputMetadata|SavedVariableHooks|ForwardGrad|VariableInfo)\b'),
        # Preserved cuda target peers referenced from .inductor (AOTI CUDA runner).
        (f"{ROOT_PKG}.cuda", r'\b(CUDAStream|CUDAStreamGuard|CUDAGuard|CUDAAllocator|MemPool)\b'),
    ]
    for pkg, pat in checks:
        if new_pkg == pkg:
            continue
        if f"{pkg}." in text or re.search(pat, text):
            extras.append(f"import {pkg}.*;")
    return extras


def rewrite_file(
    src: Path,
    dest: Path,
    new_pkg: str,
    renames: Dict[str, str],
    package_imports: Dict[str, str],
    dry_run: bool,
) -> None:
    text = read_text(src)
    text = apply_renames_in_text(text, renames)

    # Fix package declaration
    if PACKAGE_RE.search(text):
        text = PACKAGE_RE.sub(f"package {new_pkg};", text, count=1)
    else:
        text = f"package {new_pkg};\n\n" + text

    # Ensure cross-package imports for siblings that moved
    for imp in needed_star_imports(new_pkg, text):
        if imp not in text:
            text = text.replace(
                f"package {new_pkg};\n",
                f"package {new_pkg};\n\n{imp}\n",
                1,
            )

    # Drop self-package single-type imports
    text = re.sub(rf'^import {re.escape(new_pkg)}\.\w+;\n', '', text, flags=re.MULTILINE)

    if dry_run:
        action = "MOVE" if src != dest else "REWRITE"
        print(f"  [{action}] {src} -> {dest}  (package {new_pkg})")
        return

    if src != dest and dest.exists() and dest.resolve() != src.resolve():
        # Overwrite destination
        pass
    write_text(dest, text, dry_run=False)
    if src != dest and src.exists():
        src.unlink()
        # clean empty dirs
        try:
            src.parent.rmdir()
        except OSError:
            pass


def build_package_index(moves: Dict[Path, Tuple[str, Path]]) -> Dict[str, str]:
    """simpleName -> package for every moved/rewritten type."""
    idx: Dict[str, str] = {}
    for src, (pkg, dest) in moves.items():
        idx[dest.stem] = pkg
    return idx


def scan_existing_package_index(roots: List[Path]) -> Dict[str, str]:
    """Scan on-disk layout for simpleName -> package (post-move state preferred).

    When the same simple name exists in multiple packages (gloo.Store vs
    distributed.Store), prefer the managed subpackage that is *not* a
    preserved preset package; for preserved collisions the on-disk package
    of each file is what matters and we do not put that name in left_root
    blindly. We only index managed subpackages + root.
    """
    preserved = {
        f"{ROOT_PKG}.cuda", f"{ROOT_PKG}.nccl", f"{ROOT_PKG}.gloo",
        f"{ROOT_PKG}.rpc", f"{ROOT_PKG}.onnx", f"{ROOT_PKG}.global",
    }
    idx: Dict[str, str] = {}
    collisions: Dict[str, Set[str]] = {}
    for root in roots:
        for path in collect_java_files(root):
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) < 2:
                continue
            pkg = ".".join(parts[:-1])
            if not pkg.startswith(ROOT_PKG):
                continue
            # Skip preserved preset packages for the global index — their
            # simple names must not drive star-imports into foreign files.
            if pkg in preserved or any(pkg.startswith(p + ".") for p in preserved):
                continue
            simple = path.stem
            if simple in idx and idx[simple] != pkg:
                collisions.setdefault(simple, {idx[simple]}).add(pkg)
            # Prefer more specific managed subpackages over root
            if simple not in idx or idx[simple] == ROOT_PKG:
                idx[simple] = pkg
    return idx


def rewrite_references_everywhere(
    roots: List[Path],
    renames: Dict[str, str],
    package_index: Dict[str, str],
    dry_run: bool,
) -> None:
    """Second pass: fix imports / FQCNs in files that did not move."""
    # Merge planned moves with on-disk layout so re-runs still fix imports.
    disk_index = scan_existing_package_index(roots)
    disk_index.update(package_index)  # moves win
    left_root = {name: pkg for name, pkg in disk_index.items() if pkg != ROOT_PKG}

    # Names that collide with types inside preserved packages (gloo.Store).
    # When adding star-imports, skip names that equal the current file's stem.
    preserved_prefixes = (
        f"{ROOT_PKG}.cuda", f"{ROOT_PKG}.nccl", f"{ROOT_PKG}.gloo",
        f"{ROOT_PKG}.rpc", f"{ROOT_PKG}.onnx", f"{ROOT_PKG}.global",
    )

    for root in roots:
        for path in collect_java_files(root):
            text0 = read_text(path)
            text = apply_renames_in_text(text0, renames)

            # Update `import org.bytedeco.pytorch.Foo;` when Foo moved
            def repl_import(m: re.Match) -> str:
                name = m.group(1)
                pkg = left_root.get(name)
                if pkg:
                    return f"import {pkg}.{name};"
                return m.group(0)

            text = re.sub(
                rf'^import {re.escape(ROOT_PKG)}\.(\w+);',
                repl_import,
                text,
                flags=re.MULTILINE,
            )

            # Ensure star-imports for subpackages when short names from those pkgs are used
            pkg_m = PACKAGE_RE.search(text)
            current_pkg = pkg_m.group(1) if pkg_m else ROOT_PKG
            simple = path.stem

            in_preserved = current_pkg in preserved_prefixes or any(
                current_pkg.startswith(p + ".") for p in preserved_prefixes
            )

            # Short / common identifiers that collide with English/Java names and
            # must not alone trigger a star-import (jit has Use/Method/Global/…).
            # NOTE: real peer types that happen to be short/common are listed in
            # IMPORTANT_SHORT so they still drive star-imports (Obj, Value, …).
            AMBIGUOUS_SHORT = {
                "Call", "Code", "Decl", "Def", "Dots", "Expr", "For", "Global",
                "If", "Method", "Pass", "Self", "Stmt", "Tree", "Use",
                "Var", "With", "Result", "Error", "Context", "Message",
                "Type", "Stack", "Timer", "Logger", "Work", "Backend",
                "Reader", "Writer", "Module", "Graph",  # Graph is real but common
            }
            # Short but unambiguous peer type names that MUST trigger imports.
            IMPORTANT_SHORT = {
                "Obj",       # c10::ivalue::Object
                "Value",     # torch::jit::Value  (ValueMapper etc.)
                "Node",      # torch::autograd::Node
                "Edge",      # torch::autograd::Edge
                "Store",     # c10d::Store (rpc.TensorPipeAgent)
            }

            def uses_pkg(target: str) -> bool:
                for n, p in left_root.items():
                    if p != target:
                        continue
                    # Don't treat the file's own simple name as a foreign use
                    # (gloo/Store.java must not pull in distributed.* for "Store").
                    if n == simple:
                        continue
                    # Skip ultra-short / ambiguous identifiers unless allow-listed —
                    # they produce false star-imports (SimpleMNIST ← jit.Use).
                    if n not in IMPORTANT_SHORT and (len(n) <= 3 or n in AMBIGUOUS_SHORT):
                        continue
                    if re.search(rf'\b{re.escape(n)}\b', text):
                        return True
                return False

            extras: List[str] = []
            for sub in ALL_SUBPKGS:
                if current_pkg == sub:
                    continue
                if f"import {sub}.*;" in text:
                    continue
                if not uses_pkg(sub):
                    continue
                # gloo must never star-import distributed (Store name clash).
                # Other preserved packages (nccl/rpc) legitimately need it.
                if in_preserved and current_pkg.startswith(f"{ROOT_PKG}.gloo") \
                        and sub == DISTRIBUTED_PKG:
                    continue
                extras.append(f"import {sub}.*;")

            if extras:
                block = "\n".join(extras) + "\n"
                if pkg_m:
                    insertion = pkg_m.group(0) + "\n" + block
                    text = text.replace(pkg_m.group(0) + "\n", insertion, 1)
                else:
                    # Default-package sources (samples/*.java): insert after the
                    # last existing import, or at the top if none.
                    last_imp = None
                    for m in re.finditer(r'^import\s+[\w.*]+;\s*$', text, re.MULTILINE):
                        last_imp = m
                    if last_imp:
                        pos = last_imp.end()
                        text = text[:pos] + "\n" + block + text[pos:]
                    else:
                        text = block + "\n" + text

            if text != text0:
                if dry_run:
                    print(f"  [REFS] {path}")
                else:
                    write_text(path, text, dry_run=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("src/gen/java"),
        help="JavaCPP parse output root (default: src/gen/java)",
    )
    ap.add_argument(
        "--main-dir",
        type=Path,
        default=Path("src/main/java"),
        help="Hand-written sources root to also relocate (default: src/main/java)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gen_root = args.gen_dir
    main_root = args.main_dir

    print(f"Relocating packages under {gen_root} (and {main_root})")
    moves, renames, deletes = plan_moves(gen_root, main_root if main_root.exists() else None)
    print(f"  planned file ops: {len(moves)}, renames: {len(renames)}, deletes: {len(deletes)}")

    package_index = build_package_index(moves)

    # Drop stale Java* adapters superseded by non-prefixed peers
    for path in sorted(deletes, key=str):
        if args.dry_run:
            print(f"  [DELETE] {path}  (superseded by non-prefixed peer)")
        else:
            path.unlink(missing_ok=True)

    # First pass: move/rewrite classified files
    for src, (new_pkg, dest) in sorted(moves.items(), key=lambda kv: str(kv[0])):
        rewrite_file(src, dest, new_pkg, renames, package_index, args.dry_run)

    # Second pass: fix references in the whole tree
    roots = [gen_root]
    if main_root.exists():
        roots.append(main_root)
    # Also fix samples if present next to project root
    samples = Path("samples")
    if samples.exists():
        roots.append(samples)
    rewrite_references_everywhere(roots, renames, package_index, args.dry_run)

    # Summary
    counts: Dict[str, int] = {}
    for p in package_index.values():
        counts[p] = counts.get(p, 0) + 1
    print("Done.")
    for pkg in ALL_SUBPKGS:
        print(f"  {pkg}: {counts.get(pkg, 0)}")
    print(f"  renames applied: {len(renames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
