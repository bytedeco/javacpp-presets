#!/usr/bin/env python3
"""
Relocate JavaCPP-generated peer classes for torch::{data,nn,jit,optim,serialize,
inductor,profiler,enumtype}, c10d (ProcessGroup/Store/…), ATen quantizers, and
uncommon c10 types into subpackages, and strip the "Java" prefix from javacpp::
adapter types.

Why this exists
---------------
JavaCPP's Parser always writes top-level classes into a single ``target``
package (see Parser.java: targetHeader + targetDir). Fully-qualified
pointerTypes only affect *references*, not the package of the generated
file. To put peers under org.bytedeco.pytorch.{data,nn,jit,optim,serialize,
distributed,inductor,profiler,enumtype,quantizer,c10} we must post-process
after parse.

Rules
-----
1. data     – torch::data::… / javacpp Dataset adapters / DATA_NAMES, then
              refined into data / data.datasets / data.dataloader / data.sampler /
              data.transforms / data.options by classify_data_subpackage
              (Dataset itself stays in data root)
2. nn       – torch::nn::… / NN_NAMES / ModuleApply* helpers (not Jit*), then
              refined into nn / nn.options / nn.modules / nn.modules.container /
              nn.functions by classify_nn_subpackage (suffix + allowlists)
3. jit      – torch::jit::… / JitModule / NamedJitModule / BuiltinModule /
              JitModuleApplyFunction (c10 types stay root unless uncommon)
4. optim    – torch::optim::… / OPTIM_NAMES, then refined into
              optim / optim.options / optim.state / optim.schedulers by
              classify_optim_subpackage (*Options → options, *State → state,
              LRScheduler / *LR / *Scheduler → schedulers; core Optimizer* stay)
5. serialize– torch::serialize::… / SERIALIZE_NAMES
6. distributed – c10d::… / c10d::detail::… / ProcessGroup* / Store* / DIST_NAMES
                 (does NOT touch torch::distributed::rpc — that stays in .rpc)
7. inductor – torch::inductor::… / AOTI* / INDUCTOR_NAMES
8. profiler – torch::profiler::… / PROFILER_NAMES
9. enumtype – torch::enumtype::… / k* enum structs / *Mode/*PaddingMode helpers
10. quantizer – at::Quantizer hierarchy (Quantizer, PerTensorAffineQuantizer, …)
11. c10     – uncommon c10::… types (NOT Tensor/Scalar/ScalarType/Device/Stream/…)
12. Classes whose @Name is "javacpp::Dataset<...>" etc. are renamed from
    JavaXxx -> Xxx in both the file and all references across the gen tree.
13. Manual helpers under src/main/java are also moved/rewritten.
14. Already-relocated files are left alone except for rename/import rewrites.
15. Preserved subpackages (cuda/gloo/rpc/onnx/global) are never collapsed.

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

# High-level pure-Java package moves (hand-written modules, samples, resources).
# Applied by rewrite_highlevel_imports() after peer relocation so re-runs stay idempotent.
# Longest prefixes first.
MANUAL_HIGHLEVEL_REWRITES: List[Tuple[str, str]] = [
    # plot left data.dataframe before dataframe itself moved
    ("org.bytedeco.pytorch.data.dataframe.plot", "org.bytedeco.pytorch.utils.plot"),
    ("org.bytedeco.pytorch.data.dataframe", "org.bytedeco.pytorch.dataframe"),
    ("org.bytedeco.pytorch.kvcache", "org.bytedeco.pytorch.llm.kvcache"),
    ("org.bytedeco.pytorch.utils.transformers", "org.bytedeco.pytorch.llm.transformers"),
    ("org.bytedeco.pytorch.utils.sentence", "org.bytedeco.pytorch.llm.sentence"),
    ("org.bytedeco.pytorch.utils.nltk", "org.bytedeco.pytorch.llm.nltk"),
    ("org.bytedeco.pytorch.utils.ragas", "org.bytedeco.pytorch.llm.ragas"),
    ("org.bytedeco.pytorch.utils.accelerate", "org.bytedeco.pytorch.llm.accelerate"),
    ("org.bytedeco.pytorch.utils.deepspeed", "org.bytedeco.pytorch.llm.deepspeed"),
    ("org.bytedeco.pytorch.utils.trl", "org.bytedeco.pytorch.llm.trl"),
    ("org.bytedeco.pytorch.utils.peft", "org.bytedeco.pytorch.llm.peft"),
    ("org.bytedeco.pytorch.utils.bitsandbytes", "org.bytedeco.pytorch.llm.bitsandbytes"),
    ("org.bytedeco.pytorch.utils.spacy", "org.bytedeco.pytorch.llm.spacy"),
    ("org.bytedeco.pytorch.utils.vllm", "org.bytedeco.pytorch.llm.vllm"),
    ("org.bytedeco.pytorch.utils.text", "org.bytedeco.pytorch.llm.text"),
    ("org.bytedeco.pytorch.utils.tokenizers", "org.bytedeco.pytorch.llm.tokenizers"),
    ("org.bytedeco.pytorch.utils.hub", "org.bytedeco.pytorch.llm.hub"),
    ("org.bytedeco.pytorch.utils.unsloth", "org.bytedeco.pytorch.llm.unsloth"),
    ("org.bytedeco.pytorch.utils.quantization", "org.bytedeco.pytorch.llm.quantization"),
]
DATA_PKG = f"{ROOT_PKG}.data"
DATA_DATASETS_PKG = f"{DATA_PKG}.datasets"
DATA_DATALOADER_PKG = f"{DATA_PKG}.dataloader"
DATA_SAMPLER_PKG = f"{DATA_PKG}.sampler"
DATA_TRANSFORMS_PKG = f"{DATA_PKG}.transforms"
# Data *Options → org.bytedeco.pytorch.data.options
DATA_OPTIONS_PKG = f"{DATA_PKG}.options"
NN_PKG = f"{ROOT_PKG}.nn"
NN_OPTIONS_PKG = f"{NN_PKG}.options"
NN_MODULES_PKG = f"{NN_PKG}.modules"
NN_CONTAINER_PKG = f"{NN_PKG}.modules.container"
NN_FUNCTIONS_PKG = f"{NN_PKG}.functions"
JIT_PKG = f"{ROOT_PKG}.jit"
OPTIM_PKG = f"{ROOT_PKG}.optim"
# optim subpackages (refined after first-stage "optim")
OPTIM_OPTIONS_PKG = f"{OPTIM_PKG}.options"      # *Options / OptimizerCloneable*Options
OPTIM_STATE_PKG = f"{OPTIM_PKG}.state"          # *ParamState / *State / OptimizerCloneable*ParamState
OPTIM_SCHEDULERS_PKG = f"{OPTIM_PKG}.schedulers"  # LRScheduler, *LR, *Scheduler
SERIALIZE_PKG = f"{ROOT_PKG}.serialize"
DISTRIBUTED_PKG = f"{ROOT_PKG}.distributed"
INDUCTOR_PKG = f"{ROOT_PKG}.inductor"
PROFILER_PKG = f"{ROOT_PKG}.profiler"
ENUMTYPE_PKG = f"{ROOT_PKG}.enumtype"
QUANTIZER_PKG = f"{ROOT_PKG}.quantizer"
C10_PKG = f"{ROOT_PKG}.c10"
AUTOGRAD_PKG = f"{ROOT_PKG}.autograd"

ALL_SUBPKGS = (
    DATA_PKG, DATA_DATASETS_PKG, DATA_DATALOADER_PKG, DATA_SAMPLER_PKG,
    DATA_TRANSFORMS_PKG, DATA_OPTIONS_PKG,
    NN_PKG, NN_OPTIONS_PKG, NN_MODULES_PKG, NN_CONTAINER_PKG,
    NN_FUNCTIONS_PKG, JIT_PKG,
    OPTIM_PKG, OPTIM_OPTIONS_PKG, OPTIM_STATE_PKG, OPTIM_SCHEDULERS_PKG,
    SERIALIZE_PKG,
    DISTRIBUTED_PKG, INDUCTOR_PKG, PROFILER_PKG,
    ENUMTYPE_PKG, QUANTIZER_PKG, C10_PKG, AUTOGRAD_PKG,
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

# data subpackage refinement (after first-stage "data").
# Order in classify_data_subpackage: options → sampler → dataloader → transforms → datasets → core data.
DATA_OPTIONS_NAMES: Set[str] = {
    "DataLoaderOptions", "FullDataLoaderOptions", "ChunkDatasetOptions",
}
DATA_SAMPLER_NAMES: Set[str] = {
    "Sampler", "RandomSampler", "SequentialSampler", "StreamSampler",
    "DistributedSampler", "DistributedRandomSampler", "DistributedSequentialSampler",
    "BatchSizeSampler",
}
DATA_TRANSFORMS_NAMES: Set[str] = {
    "BatchTransform", "Transform", "BatchLambda", "Lambda",
    "TensorTransform", "TensorLambda", "Normalize",
    "Collation", "Stack",
    "ExampleBatchTransform", "ExampleTransform", "ExampleBatchLambda", "ExampleLambda",
    "ExampleCollation", "ExampleStack",
    "TensorExampleBatchTransform", "TensorExampleTransform",
    "TensorExampleCollation", "TensorExampleStack",
}
# Explicit core data types that stay in org.bytedeco.pytorch.data
# Dataset (javacpp::Dataset peer) stays at data root — not data.datasets.
DATA_KEEP_NAMES: Set[str] = {
    "Dataset",
    "BatchSize", "BatchSizeOptional", "CustomBatchRequest",
    "Example", "ExampleIterator", "ExampleOptional",
    "ExampleVector", "ExampleVectorIterator", "ExampleVectorOptional",
    "TensorExample", "TensorExampleIterator", "TensorExampleOptional",
    "TensorExampleVector", "TensorExampleVectorIterator", "TensorExampleVectorOptional",
    "WorkerException", "NoTarget",
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

# nn subpackage allowlists (refined after first-stage "nn" classification).
# Order in classify_nn_subpackage matters: keep → container → options → modules → functions.
NN_CONTAINER_NAMES: Set[str] = {
    "SequentialImpl", "SequentialImplCloneable",
    "ModuleListImpl", "ModuleListImplCloneable",
    "ModuleDictImpl", "ModuleDictImplCloneable",
    "ParameterDictImpl", "ParameterDictImplCloneable",
    "ParameterListImpl", "ParameterListImplCloneable",
    "AnyModule", "AnyModuleVector", "AnyValue",
    "SharedModuleVector",
    "StringAnyModuleDict", "StringAnyModuleDictItem",
    "StringAnyModuleDictItemVector", "StringAnyModulePair",
    "StringAnyModuleVector",
    "StringSharedModuleDict", "StringSharedModuleDictItem",
    "StringSharedModuleDictItemVector", "StringSharedModulePair",
    "StringSharedModuleVector",
}

NN_FUNCTIONS_NAMES: Set[str] = {
    "FunctionCrossMapLRN2d",
}

# Explicit core nn types that must stay in org.bytedeco.pytorch.nn
# (not modules/options/container). Includes hand-written Parameter.
NN_KEEP_NAMES: Set[str] = {
    "Module", "ModuleHolder", "Cloneable", "Functional",
    "ModuleApplyFunction", "NamedModuleApplyFunction",
    "SharedModuleApplyFunction", "NamedSharedModuleApplyFunction",
    "module_iterator", "module_list",
    "named_module_iterator", "named_module_list",
    "NamedAnyModule",
    "ASMoutput", "PackedSequence",
    "T_PackedSequenceT_TensorTensor_T_T", "T_PackedSequenceTensor_T",
    "Parameter",
}

NN_OPTIONS_SUFFIXES = ("Options", "FuncOptions", "OptionsBase")
NN_IMPL_SUFFIXES = ("Impl", "ImplBase", "ImplBaseBase", "ImplCloneable")

OPTIM_NAMES: Set[str] = {
    # core (stay in optim)
    "Optimizer", "OptimizerParamGroup", "OptimizerParamGroupVector",
    # base options/state (also end with Options/State — classified into subpkgs)
    "OptimizerOptions", "OptimizerParamState",
    # concrete optimizers + their Options / ParamState
    "Adagrad", "AdagradOptions", "AdagradParamState",
    "Adam", "AdamOptions", "AdamParamState",
    "AdamW", "AdamWOptions", "AdamWParamState",
    "Adadelta", "AdadeltaOptions", "AdadeltaParamState",
    "Adamax", "AdamaxOptions", "AdamaxParamState",
    "ASGD", "ASGDOptions", "ASGDParamState",
    "LBFGS", "LBFGSOptions", "LBFGSParamState",
    "NAdam", "NAdamOptions", "NAdamParamState",
    "RAdam", "RAdamOptions", "RAdamParamState",
    "RMSprop", "RMSpropOptions", "RMSpropParamState",
    "Rprop", "RpropOptions", "RpropParamState",
    "SGD", "SGDOptions", "SGDParamState",
    # LR schedulers (→ optim.schedulers)
    "LRScheduler", "StepLR", "ReduceLROnPlateauScheduler",
    "MultiplicativeLR", "MultiStepLR", "ConstantLR", "LinearLR",
    "ExponentialLR", "PolynomialLR", "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts", "CyclicLR",
    "LambdaLR", "LRLambda",
    # template specializations (→ options / state by suffix)
    "OptimizerCloneableAdagradOptions", "OptimizerCloneableAdagradParamState",
    "OptimizerCloneableAdamOptions", "OptimizerCloneableAdamParamState",
    "OptimizerCloneableAdamWOptions", "OptimizerCloneableAdamWParamState",
    "OptimizerCloneableAdadeltaOptions", "OptimizerCloneableAdadeltaParamState",
    "OptimizerCloneableAdamaxOptions", "OptimizerCloneableAdamaxParamState",
    "OptimizerCloneableASGDOptions", "OptimizerCloneableASGDParamState",
    "OptimizerCloneableLBFGSOptions", "OptimizerCloneableLBFGSParamState",
    "OptimizerCloneableNAdamOptions", "OptimizerCloneableNAdamParamState",
    "OptimizerCloneableRAdamOptions", "OptimizerCloneableRAdamParamState",
    "OptimizerCloneableRMSpropOptions", "OptimizerCloneableRMSpropParamState",
    "OptimizerCloneableRpropOptions", "OptimizerCloneableRpropParamState",
    "OptimizerCloneableSGDOptions", "OptimizerCloneableSGDParamState",
}

# Core optim types that stay in org.bytedeco.pytorch.optim (not options/state/schedulers).
OPTIM_KEEP_NAMES: Set[str] = {
    "Optimizer",
    "OptimizerParamGroup",
    "OptimizerParamGroupVector",
    # concrete optimizer algorithm classes (no Options/State/LR/Scheduler suffix)
    "Adagrad", "Adam", "AdamW", "Adadelta", "Adamax", "ASGD",
    "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD",
}

# Scheduler simple names / suffixes (refined into optim.schedulers).
OPTIM_SCHEDULER_NAMES: Set[str] = {
    "LRScheduler", "LRLambda",
    "StepLR", "ReduceLROnPlateauScheduler",
    "MultiplicativeLR", "MultiStepLR", "ConstantLR", "LinearLR",
    "ExponentialLR", "PolynomialLR", "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts", "CyclicLR", "LambdaLR",
}
OPTIM_SCHEDULER_SUFFIXES = ("LR", "Scheduler")
# Options / ParamState suffixes for optim subpackages.
OPTIM_OPTIONS_SUFFIXES = ("Options",)
OPTIM_STATE_SUFFIXES = ("ParamState", "State")

SERIALIZE_NAMES: Set[str] = {
    "InputArchive", "OutputArchive",
}

# c10d ProcessGroup / Store / collective options / DDP helpers.
# Does NOT include torch::distributed::rpc (owned by torch_rpc preset → .rpc).
DISTRIBUTED_NAMES: Set[str] = {
    # ProcessGroup / Backend
    "ProcessGroup", "ProcessGroupGloo", "ProcessGroupStatus",
    "ProcessGroupCppCommHookInterface",
    # C++ c10d::ProcessGroupWrapper → Java ProcessGroupNativeWrapper
    "ProcessGroupNativeWrapper",
    # Optional backends (torch_mpi / torch_ucc presets)
    "ProcessGroupMPI", "ProcessGroupUCC", "ProcessGroupUCCLogger",
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
    "AOTIModelContainerRunnerMps",
    "AOTIModelContainerRunnerXpu",
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
    # Kineto chrome-trace path (enableProfiler / disableProfiler / save)
    "ProfilerResult", "KinetoEvent", "RecordScopeSet",
    "ProfilerResultEventVector",
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

# at:: quantizer hierarchy (ATen/core/QuantizerBase.h + ATen/quantized/Quantizer.h).
# User-facing short names (PerTensorQuantizer / StaticQuantizer / …) do not exist
# in ATen; the real peers are the Affine* leaves listed here.
QUANTIZER_NAMES: Set[str] = {
    "Quantizer",
    "UnknownQuantizer",
    "UniformQuantizer",
    "NonUniformQuantizer",
    "AffineQuantizer",
    "PerTensorAffineQuantizer",
    "PerChannelAffineQuantizer",
    "PerChannelAffineFloatQParamsQuantizer",
    # Hand-written Python-style AMP context (src/main/.../quantizer/AutocastContext.java)
    "AutocastContext",
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

# Names that look like nn/data/… but must stay in root.
# Note: JitModule / NamedJitModule / BuiltinModule intentionally go to .jit
# (torch::jit::Module peers), not root — see classify_by_simple_name.
ROOT_FORCE: Set[str] = {
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
    # Quantizer hierarchy is at:: but may carry c10::make_intrusive @Name
    # annotations — force .quantizer before the c10:: token walk below.
    if simple in QUANTIZER_NAMES:
        return "quantizer"
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
        # torch::autograd::profiler peers belong under .profiler (KinetoEvent, ProfilerResult)
        if n == "torch::autograd" or n.startswith("torch::autograd::"):
            # Guard: dynamo nested ns must not match the prefix above
            if "torch::dynamo::" in n:
                return "root"
            if "torch::autograd::profiler" in n or n.startswith("torch::autograd::profiler"):
                return "profiler"
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
        # (at::Quantizer hierarchy is handled via QUANTIZER_NAMES early return above.)
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
    # JitModule / NamedJitModule / BuiltinModule are torch::jit peers → jit
    # (also usually caught by @Name/@Namespace torch::jit::…, but keep explicit).
    if simple in ("JitModule", "NamedJitModule", "BuiltinModule", "JitModuleApplyFunction"):
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

    if simple in QUANTIZER_NAMES:
        return "quantizer"

    if simple in PROFILER_NAMES:
        return "profiler"

    if simple in NN_NAMES:
        return "nn"
    if simple.endswith(NN_NAME_SUFFIXES):
        # Avoid data-side *Options that aren't nn (hand-written dataframe/json I/O etc.)
        if simple in (
            "ChunkDatasetOptions", "DataLoaderOptions", "FullDataLoaderOptions",
            "CsvOptions", "JsonOptions", "JsonReadOptions", "JsonWriteOptions",
        ):
            return "data"
        # Optimizer / c10d option class names
        if simple.startswith((
            "Adagrad", "Adam", "AdamW", "Adadelta", "Adamax", "ASGD",
            "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD",
            "Optimizer",
        )):
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


def classify_nn_subpackage(simple: str) -> str:
    """Refine a first-stage 'nn' classification into nn / options / modules / container / functions.

    Order matters: keep → container → options → modules → functions → default nn.
    """
    if simple in NN_KEEP_NAMES:
        return "nn"
    if simple in NN_CONTAINER_NAMES:
        return "nn_container"
    # Options family (includes *FuncOptions / *OptionsBase / Detail*Options)
    # but exclude distributed/c10d *Options that happen to end with Options
    # (DistributedBackendOptions, TCPStoreOptions, AllToAllOptions, etc.)
    if simple.endswith(NN_OPTIONS_SUFFIXES) or (
        simple.startswith("Detail") and simple.endswith("Options")
    ):
        if simple in (
            # distributed options (c10d) that should NOT go to nn.options
            "DistributedBackendOptions", "TCPStoreOptions",
            "AllToAllOptions", "AllgatherOptions", "AllreduceOptions",
            "AllreduceCoalescedOptions", "BarrierOptions", "BroadcastOptions",
            "GatherOptions", "ReduceOptions", "ReduceScatterOptions", "ScatterOptions",
        ):
            pass  # fall through to default nn
        else:
            return "nn_options"
    if simple.endswith(NN_IMPL_SUFFIXES):
        return "nn_modules"
    if simple in NN_FUNCTIONS_NAMES or (
        simple.startswith("Function") and simple not in NN_KEEP_NAMES
    ):
        return "nn_functions"
    return "nn"


def classify_optim_subpackage(simple: str) -> str:
    """Refine a first-stage 'optim' classification into optim subpackages.

    Order matters: keep → schedulers → options → state → default optim.

    Target packages:
      optim_schedulers → org.bytedeco.pytorch.optim.schedulers
                         (LRScheduler, *LR, *Scheduler, LRLambda)
      optim_options    → org.bytedeco.pytorch.optim.options
                         (*Options, OptimizerCloneable*Options, OptimizerOptions)
      optim_state      → org.bytedeco.pytorch.optim.state
                         (*ParamState, *State, OptimizerCloneable*ParamState,
                          OptimizerParamState)
      optim            → org.bytedeco.pytorch.optim
                         (Optimizer, OptimizerParamGroup*, concrete optimizers)
    """
    if simple in OPTIM_KEEP_NAMES:
        return "optim"
    # Schedulers before Options/State so ReduceLROnPlateauScheduler is not
    # treated as a *State / *Options class.
    if simple in OPTIM_SCHEDULER_NAMES or (
        simple.endswith(OPTIM_SCHEDULER_SUFFIXES)
        and simple not in OPTIM_KEEP_NAMES
        and not simple.endswith(OPTIM_OPTIONS_SUFFIXES)
        and not simple.endswith(OPTIM_STATE_SUFFIXES)
    ):
        return "optim_schedulers"
    # Options: *Options including OptimizerOptions / OptimizerCloneable*Options
    if simple.endswith(OPTIM_OPTIONS_SUFFIXES) or (
        simple.startswith("OptimizerCloneable") and simple.endswith("Options")
    ):
        return "optim_options"
    # State: *ParamState / *State including OptimizerParamState /
    # OptimizerCloneable*ParamState
    if simple.endswith(OPTIM_STATE_SUFFIXES) or (
        simple.startswith("OptimizerCloneable") and simple.endswith("ParamState")
    ):
        return "optim_state"
    return "optim"


def classify_data_subpackage(simple: str, text: str = "") -> str:
    """Refine a first-stage 'data' classification into data subpackages.

    Order matters: options → sampler → dataloader → transforms → datasets → core data.

    Target packages:
      data_options    → org.bytedeco.pytorch.data.options  (DataLoader/Chunk *Options)
      data_sampler    → org.bytedeco.pytorch.data.sampler
      data_dataloader → org.bytedeco.pytorch.data.dataloader
      data_transforms → org.bytedeco.pytorch.data.transforms
      data_datasets   → org.bytedeco.pytorch.data.datasets
      data            → org.bytedeco.pytorch.data (Dataset, Example*, BatchSize, …)
    """
    if simple in DATA_KEEP_NAMES:
        return "data"
    if simple in DATA_OPTIONS_NAMES or (
        simple.endswith("Options") and simple in DATA_NAMES
    ) or simple.endswith(("DataLoaderOptions", "DatasetOptions")):
        return "data_options"
    if simple in DATA_SAMPLER_NAMES or simple.endswith("Sampler"):
        return "data_sampler"
    # DataLoader before Dataset: MNISTRandomDataLoader / ChunkRandomDataLoader
    if "DataLoader" in simple:
        return "data_dataloader"
    if simple in DATA_TRANSFORMS_NAMES or simple.endswith((
        "Transform", "Lambda", "Collation", "Stack",
    )) or simple == "Normalize":
        return "data_transforms"
    # datasets: *Dataset*, readers, MNIST, NoTarget, javacpp Dataset adapters
    if (
        simple in ("MNIST", "NoTarget", "ChunkDataReader", "ChunkTensorDataReader",
                   "ChunkRecordIterator")
        or "Dataset" in simple
        or simple.endswith("DataReader")
        or "torch::data::datasets" in text
        or "javacpp::Dataset" in text
        or "javacpp::StreamDataset" in text
        or "javacpp::StatefulDataset" in text
    ):
        return "data_datasets"
    return "data"


def classify_file(path: Path, text: str) -> Optional[str]:
    simple = path.stem
    # Hand-written data I/O option classes: pin exact packages so a previous
    # mis-classification into nn.options can be corrected on re-run.
    handwritten_option_pkg = {
        "CsvOptions": f"{ROOT_PKG}.dataframe.csv",
        "JsonOptions": f"{ROOT_PKG}.dataframe.json",
        "JsonReadOptions": f"{DATA_PKG}.json",
        "JsonWriteOptions": f"{DATA_PKG}.json",
        "AvroOptions": f"{DATA_PKG}.avro",
        "OrcOptions": f"{DATA_PKG}.orc",
        # spaCy pure-Java impls (must never land in nn.modules via *Impl heuristic)
        "DocImpl": f"{ROOT_PKG}.llm.spacy.impl",
        "TokenImpl": f"{ROOT_PKG}.llm.spacy.impl",
        "SpanImpl": f"{ROOT_PKG}.llm.spacy.impl",
        "LanguageImpl": f"{ROOT_PKG}.llm.spacy.impl",
        # Domain-specific *Options that must stay in their correct package,
        # not misrouted to nn.options by the broad *Options→nn_options heuristic.
        "DockerOptions": f"{ROOT_PKG}.deploy.docker",
        "K8sOptions": f"{ROOT_PKG}.deploy.k8s",
        "VistaOptions": f"{ROOT_PKG}.plot.vista",
        # Serve options
        "TorchServeOptions": f"{ROOT_PKG}.serve",
        "ModelLoadOptions": f"{ROOT_PKG}.serve",
        "InferenceOptions": f"{ROOT_PKG}.serve",
        # Deploy options
        "AbtestOptions": f"{ROOT_PKG}.deploy.abtest",
        "OfflineOptions": f"{ROOT_PKG}.deploy.offline",
        # Recommend options
        "NCFOptions": f"{ROOT_PKG}.recommend.modelops",
        "DeepFMOptions": f"{ROOT_PKG}.recommend.modelops",
        "DSSMOptions": f"{ROOT_PKG}.recommend.modelops",
        # Feature options
        "FeatureOptions": f"{ROOT_PKG}.feature",
        "ChartOptions": f"{ROOT_PKG}.plot.chart",
        "MatplotOptions": f"{ROOT_PKG}.plot.matplot",
        # Feature store classes misclassified as "distributed" due to *Store suffix
        "OfflineStore": f"{ROOT_PKG}.feature.offline",
        "FileOfflineStore": f"{ROOT_PKG}.feature.offline",
        "SqliteOfflineStore": f"{ROOT_PKG}.feature.offline",
        "OnlineStore": f"{ROOT_PKG}.feature.online",
        "OnlineFeatureRow": f"{ROOT_PKG}.feature.online",
        "OnlineWriteBatch": f"{ROOT_PKG}.feature.online",
        "InMemoryOnlineStore": f"{ROOT_PKG}.feature.online",
        "SqliteOnlineStore": f"{ROOT_PKG}.feature.online",
        "RedisOnlineStore": f"{ROOT_PKG}.feature.online",
        "FileOnlineStore": f"{ROOT_PKG}.feature.online",
        "RegistryStore": f"{ROOT_PKG}.feature.registry",
        "FileRegistryStore": f"{ROOT_PKG}.feature.registry",
        "InMemoryRegistryStore": f"{ROOT_PKG}.feature.registry",
        "FeatureVersion": f"{ROOT_PKG}.feature.registry",
        "EmbeddingStore": f"{ROOT_PKG}.feature.store",
        "MemoryEmbeddingStore": f"{ROOT_PKG}.feature.store",
        "StoreConfig": f"{ROOT_PKG}.feature.store",
        "MilvusEmbeddingStore": f"{ROOT_PKG}.feature.store",
        "RedisVectorEmbeddingStore": f"{ROOT_PKG}.feature.store",
        "DuckDbOfflineStore": f"{ROOT_PKG}.feature.offline",
        "LanceOfflineStore": f"{ROOT_PKG}.feature.offline",
        # Audio-specific classes
        "AudioDataset": f"{ROOT_PKG}.audio.datasets",
        "AudioFolder": f"{ROOT_PKG}.audio.datasets",
        "FakeAudio": f"{ROOT_PKG}.audio.datasets",
    }
    if simple in handwritten_option_pkg:
        return f"handwritten:{handwritten_option_pkg[simple]}"

    # TensorDataset in recommend.data must not be confused with data.datasets.TensorDataset
    # Pin by path to preserve the recommend module's own TensorDataset
    if simple == "TensorDataset":
        posix = path.as_posix()
        if "/recommend/" in posix:
            return f"handwritten:{ROOT_PKG}.recommend.data"

    # Multiple pure-Java Transform types share the simple name "Transform".
    # Pin by path / content — do NOT force all of them into
    # distribution.transforms (that previously ate audio/data Transform).
    if simple == "Transform":
        posix = path.as_posix()
        if "/distribution/transforms/" in posix:
            return f"handwritten:{ROOT_PKG}.distribution.transforms"
        if "/utils/audio/transforms/" in posix:
            return f"handwritten:{ROOT_PKG}.utils.audio.transforms"
        if "/utils/vision/transforms/" in posix:
            return f"handwritten:{ROOT_PKG}.utils.vision.transforms"
        if "/audio/transforms/" in posix:
            return f"handwritten:{ROOT_PKG}.audio.transforms"
        if "/vision/transforms/" in posix:
            return f"handwritten:{ROOT_PKG}.vision.transforms"
        if "/data/transforms/" in posix:
            return f"handwritten:{DATA_PKG}.transforms"
        # abstract bijector API (torch.distributions.transforms)
        if "eventDim" in text and "logAbsDetJacobian" in text:
            return f"handwritten:{ROOT_PKG}.distribution.transforms"
        # generic functional interface used by vision/audio/dataset pipelines
        if "@FunctionalInterface" in text:
            if "audio" in text.lower():
                return f"handwritten:{ROOT_PKG}.utils.audio.transforms"
            return f"handwritten:{DATA_PKG}.transforms"

    by_attr = classify_by_name_attr(text, simple)
    kind = by_attr if by_attr else classify_by_simple_name(simple)
    if kind == "nn":
        return classify_nn_subpackage(simple)
    if kind == "data":
        return classify_data_subpackage(simple, text)
    if kind == "optim":
        return classify_optim_subpackage(simple)
    return kind


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
    if kind and kind.startswith("handwritten:"):
        return kind.split(":", 1)[1]
    return {
        "data": DATA_PKG,
        "data_datasets": DATA_DATASETS_PKG,
        "data_dataloader": DATA_DATALOADER_PKG,
        "data_sampler": DATA_SAMPLER_PKG,
        "data_transforms": DATA_TRANSFORMS_PKG,
        "data_options": DATA_OPTIONS_PKG,
        "nn": NN_PKG,
        "nn_options": NN_OPTIONS_PKG,
        "nn_modules": NN_MODULES_PKG,
        "nn_container": NN_CONTAINER_PKG,
        "nn_functions": NN_FUNCTIONS_PKG,
        "jit": JIT_PKG,
        "optim": OPTIM_PKG,
        "optim_options": OPTIM_OPTIONS_PKG,
        "optim_state": OPTIM_STATE_PKG,
        "optim_schedulers": OPTIM_SCHEDULERS_PKG,
        "serialize": SERIALIZE_PKG,
        "distributed": DISTRIBUTED_PKG,
        "inductor": INDUCTOR_PKG,
        "profiler": PROFILER_PKG,
        "enumtype": ENUMTYPE_PKG,
        "quantizer": QUANTIZER_PKG,
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
    # (cuda/gloo/rpc/onnx/global are separate presets / targets).
    preserved_subpkgs = {
        f"{ROOT_PKG}.cuda",
        # f"{ROOT_PKG}.nccl",  # torch_nccl target is now .distributed
        f"{ROOT_PKG}.gloo",
        f"{ROOT_PKG}.rpc",
        f"{ROOT_PKG}.onnx",
        f"{ROOT_PKG}.global",
    }

    # Hand-written pure-Java packages under src/main (dataframe, utils, json I/O, …).
    # These are never produced by JavaCPP parse and must not be reclassified by
    # simple-name heuristics (e.g. *Options → nn.options, *Impl → nn.modules,
    # *Sampler → data.sampler). Losing these packages silently drops whole
    # modules (especially dataframe + utils) from the install jar.
    handwritten_data_prefixes = (
        f"{ROOT_PKG}.dataframe",
        f"{DATA_PKG}.json",
        f"{DATA_PKG}.parquet",
        f"{DATA_PKG}.arrow",
        f"{DATA_PKG}.avro",
        f"{DATA_PKG}.orc",
        f"{DATA_PKG}.numpy",
        f"{DATA_PKG}.pickle",
        f"{DATA_PKG}.safetensors",
        f"{DATA_PKG}.gguf",
        # NOTE: do NOT blanket-skip data.transforms — it mixes JavaCPP peers
        # (Normalize, *Lambda, …) with one pure-Java Transform<T,R>. The latter
        # is pinned in classify_file(); gen peers still need import rewrites.
        f"{ROOT_PKG}.distribution",
        f"{ROOT_PKG}.utils",
        f"{ROOT_PKG}.rl",
        f"{ROOT_PKG}.llm",
        f"{ROOT_PKG}.geometric",
        f"{ROOT_PKG}.amp",
        f"{ROOT_PKG}.info",
        f"{ROOT_PKG}.quantization",
        # Hand-written domain modules that must never be misclassified to nn/options or nn/modules
        f"{ROOT_PKG}.recommend",
        f"{ROOT_PKG}.feature",
        f"{ROOT_PKG}.deploy",
        f"{ROOT_PKG}.vision",
        f"{ROOT_PKG}.audio",
        f"{ROOT_PKG}.plot",
        f"{ROOT_PKG}.serve",
    )

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

            # Hand-written data helpers (dataframe/json/parquet/…) — leave alone.
            if any(
                current_pkg == p or current_pkg.startswith(p + ".")
                for p in handwritten_data_prefixes
            ):
                continue

            if in_preserved:
                # Exception: torch::inductor AOTI* peers may be emitted under
                # the cuda target (model_container_runner_cuda.h) but belong in
                # .inductor. Allow reclassification of those only.
                if current_pkg == f"{ROOT_PKG}.cuda" and (
                    simple.startswith("AOTI") or simple in INDUCTOR_NAMES
                ):
                    pkg_kind = "inductor"
                else:
                    # Leave cuda/gloo/rpc/onnx/global alone entirely.
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
        # data core (Dataset, Example*, BatchSize, …)
        (DATA_PKG, r'\b(Dataset|Example|ExampleOptional|ExampleVector|ExampleIterator|ExampleVectorIterator|ExampleVectorOptional|TensorExample|TensorExampleOptional|TensorExampleVector|TensorExampleIterator|TensorExampleVectorIterator|TensorExampleVectorOptional|BatchSize|BatchSizeOptional|CustomBatchRequest|WorkerException|NoTarget)\b'),
        # Exclude bare "Dataset" (lives in data root); match *Dataset* compound names.
        (DATA_DATASETS_PKG, r'\b(?:\w+Dataset\w*|Dataset\w+|\w+Dataset|MNIST|ChunkDataReader|ChunkTensorDataReader|ChunkRecordIterator)\b'),
        (DATA_DATALOADER_PKG, r'\b\w*DataLoader(?:Base)?\b'),
        (DATA_SAMPLER_PKG, r'\b\w*Sampler\b'),
        (DATA_TRANSFORMS_PKG, r'\b(?:\w*(?:Transform|Lambda|Collation|Stack)|Normalize)\b'),
        (DATA_OPTIONS_PKG, r'\b(?:DataLoaderOptions|FullDataLoaderOptions|ChunkDatasetOptions)\b'),
        # nn core
        (NN_PKG, r'\b(Module|ModuleHolder|ModuleApplyFunction|NamedModuleApplyFunction|SharedModuleApplyFunction|NamedSharedModuleApplyFunction|Parameter|PackedSequence|ASMoutput|Cloneable|Functional)\b'),
        # nn.options — *Options / *FuncOptions / *OptionsBase
        # Exclude hand-written data I/O option classes (Csv/Json*) that live under
        # dataframe / data.json — they also end with Options.
        (NN_OPTIONS_PKG, r'\b(?!CsvOptions|JsonOptions|JsonReadOptions|JsonWriteOptions|AvroOptions|OrcOptions)\w+(?:Func)?Options(?:Base)?\b'),
        # nn.modules — *Impl family excluding containers + spaCy pure-Java *Impl
        (NN_MODULES_PKG, r'\b(?!Sequential|ModuleList|ModuleDict|ParameterDict|ParameterList|DocImpl|TokenImpl|SpanImpl|LanguageImpl)\w+Impl(?:Base|BaseBase|Cloneable)?\b'),
        # nn.modules.container
        (NN_CONTAINER_PKG, r'\b(SequentialImpl|SequentialImplCloneable|ModuleListImpl|ModuleListImplCloneable|ModuleDictImpl|ModuleDictImplCloneable|ParameterDictImpl|ParameterDictImplCloneable|ParameterListImpl|ParameterListImplCloneable|AnyModule|AnyModuleVector|AnyValue|SharedModuleVector|StringAnyModule\w*|StringSharedModule\w*)\b'),
        # nn.functions
        (NN_FUNCTIONS_PKG, r'\b(FunctionCrossMapLRN2d)\b'),
        # Value is the main short name that breaks hand-written helpers (ValueMapper).
        (JIT_PKG, r'\b(JitModule|Graph|ScriptModule|TypePtr|JitModuleApplyFunction|JitObject|Value)\b'),
        # optim core — concrete optimizers + ParamGroup (not Options/State/schedulers)
        (OPTIM_PKG, r'\b(Optimizer|OptimizerParamGroup|OptimizerParamGroupVector|SGD|Adam|AdamW|Adagrad|Adadelta|Adamax|ASGD|LBFGS|NAdam|RAdam|RMSprop|Rprop)\b'),
        # optim.options — only torch.optim *Options (not nn / data / c10d *Options)
        (OPTIM_OPTIONS_PKG, r'\b(?:OptimizerOptions|OptimizerCloneable\w+Options|(?:Adagrad|AdamW?|Adadelta|Adamax|ASGD|LBFGS|NAdam|RAdam|RMSprop|Rprop|SGD)Options)\b'),
        # optim.state — only torch.optim *ParamState / OptimizerParamState
        (OPTIM_STATE_PKG, r'\b(?:OptimizerParamState|OptimizerCloneable\w+ParamState|(?:Adagrad|AdamW?|Adadelta|Adamax|ASGD|LBFGS|NAdam|RAdam|RMSprop|Rprop|SGD)ParamState)\b'),
        # optim.schedulers — LRScheduler / *LR / *Scheduler / LRLambda
        (OPTIM_SCHEDULERS_PKG, r'\b(?:LRScheduler|LRLambda|StepLR|ReduceLROnPlateauScheduler|MultiplicativeLR|MultiStepLR|ConstantLR|LinearLR|ExponentialLR|PolynomialLR|CosineAnnealingLR|CosineAnnealingWarmRestarts|CyclicLR|LambdaLR)\b'),
        (SERIALIZE_PKG, r'\b(InputArchive|OutputArchive)\b'),
        (DISTRIBUTED_PKG, r'\b(ProcessGroup|Store|TCPStore|FileStore|HashStore|PrefixStore|Work|ReduceOp|Backend|Reducer)\b'),
        (INDUCTOR_PKG, r'\b(AOTIModelContainerRunner(?:Cpu|Cuda|Mps|Xpu)?|AOTIModelPackageLoader)\b'),
        (PROFILER_PKG, r'\b(ProfilerConfig|ExperimentalConfig|ActivityTypeSet|ITraceActivity)\b'),
        (ENUMTYPE_PKG, r'\b(kLinear|kReLU|kGELU|kNone|kMean|kSum|kBilinear|kNearest|GridSampleMode|InterpolateMode)\b'),
        # Concrete quantizer peers + hand-written AutocastContext.
        # Negative lookahead keeps QuantizerType* in .c10.
        (QUANTIZER_PKG, r'\b(?:UnknownQuantizer|UniformQuantizer|NonUniformQuantizer|AffineQuantizer|PerTensorAffineQuantizer|PerChannelAffineQuantizer|PerChannelAffineFloatQParamsQuantizer|AutocastContext|Quantizer(?!Type))\b'),
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
        # Recursively prune empty parents up to (but not including) the package root.
        prune_empty_parents(src.parent, stop_at_pkg_root=True)


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
        f"{ROOT_PKG}.cuda", f"{ROOT_PKG}.gloo",
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


# Nested peer types that JavaCPP sometimes emits unqualified when the outer
# class lives in another package (e.g. rpc files referencing Type.TypePtr).
# Map bare simple name → qualified form. Only applied outside the outer class
# file itself (Type.java may use bare TypePtr inside the nested class body).
NESTED_TYPE_FIXES: Dict[str, str] = {
    "TypePtr": "Type.TypePtr",
}


def fix_nested_type_refs(path: Path, text: str) -> str:
    """Qualify bare nested-type names (TypePtr → Type.TypePtr) outside their outer class."""
    simple = path.stem
    for bare, qualified in NESTED_TYPE_FIXES.items():
        # Outer class file (Type.java) legitimately uses bare TypePtr inside
        # the nested class; leave it alone.
        if simple == qualified.split(".", 1)[0]:
            continue
        if bare not in text:
            continue
        # Word-boundary, not already qualified (no preceding '.' / C++ '::'),
        # not part of a longer identifier (AnyTypePtr / TypePtrOptional / …).
        # Exclude ':' so @Name("c10::TypePtr") stays a C++ token.
        text = re.sub(
            rf'(?<![\w.:]){re.escape(bare)}(?![\w])',
            qualified,
            text,
        )
    return text


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
        f"{ROOT_PKG}.cuda", f"{ROOT_PKG}.gloo",
        f"{ROOT_PKG}.rpc", f"{ROOT_PKG}.onnx", f"{ROOT_PKG}.global",
    )
    handwritten_data_prefixes = (
        f"{ROOT_PKG}.dataframe",
        f"{DATA_PKG}.json",
        f"{DATA_PKG}.parquet",
        f"{DATA_PKG}.arrow",
        f"{DATA_PKG}.avro",
        f"{DATA_PKG}.orc",
        f"{DATA_PKG}.numpy",
        f"{DATA_PKG}.pickle",
        f"{DATA_PKG}.safetensors",
        f"{DATA_PKG}.gguf",
        # data.transforms mixed gen+handwritten — do not blanket-skip (see plan_moves).
        f"{ROOT_PKG}.distribution",
        f"{ROOT_PKG}.utils",
        f"{ROOT_PKG}.rl",
        f"{ROOT_PKG}.llm",
        f"{ROOT_PKG}.geometric",
        f"{ROOT_PKG}.amp",
        f"{ROOT_PKG}.info",
        f"{ROOT_PKG}.quantization",
        # Hand-written domain modules that must never be misclassified to nn/options or nn/modules
        f"{ROOT_PKG}.recommend",
        f"{ROOT_PKG}.feature",
        f"{ROOT_PKG}.deploy",
        f"{ROOT_PKG}.vision",
        f"{ROOT_PKG}.audio",
        f"{ROOT_PKG}.plot",
        f"{ROOT_PKG}.serve",
    )

    for root in roots:
        for path in collect_java_files(root):
            text0 = read_text(path)
            try:
                rel = path.relative_to(root)
                cur_pkg = ".".join(rel.parts[:-1]) if len(rel.parts) > 1 else ROOT_PKG
            except ValueError:
                cur_pkg = ROOT_PKG
            # Hand-written pure-Java modules still need import/FQCN rewrites when
            # managed peers move (e.g. optim.AdamOptions → optim.options.AdamOptions).
            # Only skip *star-import injection* below — that was the original reason
            # for skipping these packages (nn/jit false positives).
            is_handwritten = any(
                cur_pkg == p or cur_pkg.startswith(p + ".")
                for p in handwritten_data_prefixes
            )

            text = apply_renames_in_text(text0, renames)
            # Qualify bare nested types (TypePtr → Type.TypePtr) outside Type.java
            text = fix_nested_type_refs(path, text)

            # Update single-type imports when Foo moved, covering both root and
            # managed subpackages (e.g. org.bytedeco.pytorch.nn.LinearImpl →
            # org.bytedeco.pytorch.nn.modules.LinearImpl after nn split).
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
            # Rewrite imports under any managed subpackage (including nested
            # nn.options / nn.modules / …) when the simple name has moved.
            for sub in ALL_SUBPKGS:
                text = re.sub(
                    rf'^import {re.escape(sub)}\.(\w+);',
                    repl_import,
                    text,
                    flags=re.MULTILINE,
                )

            # Rewrite fully-qualified type names in source bodies when a type left root
            # or moved deeper under nn (e.g. after JitModule → jit, or LinearImpl
            # → nn.modules). Only two old homes are rewritten to keep this O(names):
            #   org.bytedeco.pytorch.Name
            #   org.bytedeco.pytorch.nn.Name   (if final pkg is a nn subpackage)
            for name, pkg in left_root.items():
                if pkg == ROOT_PKG:
                    continue
                correct = f"{pkg}.{name}"
                wrong_root = f"{ROOT_PKG}.{name}"
                if wrong_root != correct and wrong_root in text:
                    text = re.sub(
                        rf'(?<![\w.]){re.escape(wrong_root)}\b',
                        correct,
                        text,
                    )
                if pkg.startswith(NN_PKG + "."):
                    wrong_nn = f"{NN_PKG}.{name}"
                    if wrong_nn != correct and wrong_nn in text:
                        text = re.sub(
                            rf'(?<![\w.]){re.escape(wrong_nn)}\b',
                            correct,
                            text,
                        )
                if pkg.startswith(DATA_PKG + ".") or pkg == DATA_OPTIONS_PKG:
                    wrong_data = f"{DATA_PKG}.{name}"
                    if wrong_data != correct and wrong_data in text:
                        text = re.sub(
                            rf'(?<![\w.]){re.escape(wrong_data)}\b',
                            correct,
                            text,
                        )
                # optim → optim.options / optim.state / optim.schedulers
                if pkg.startswith(OPTIM_PKG + "."):
                    wrong_optim = f"{OPTIM_PKG}.{name}"
                    if wrong_optim != correct and wrong_optim in text:
                        text = re.sub(
                            rf'(?<![\w.]){re.escape(wrong_optim)}\b',
                            correct,
                            text,
                        )

            # Ensure star-imports for subpackages when short names from those pkgs are used
            # (skip for hand-written modules — they manage their own imports).
            pkg_m = PACKAGE_RE.search(text)
            current_pkg = pkg_m.group(1) if pkg_m else ROOT_PKG
            simple = path.stem

            if is_handwritten:
                # If bare optim Options / ParamState / Scheduler short names are used
                # with only `import org.bytedeco.pytorch.optim.*`, inject the needed
                # subpackage star-imports (optim.* no longer covers them).
                hw_extras: List[str] = []
                optim_sub_checks = [
                    (OPTIM_OPTIONS_PKG,
                     r'\b(?:OptimizerOptions|OptimizerCloneable\w+Options|'
                     r'(?:Adagrad|AdamW?|Adadelta|Adamax|ASGD|LBFGS|NAdam|RAdam|RMSprop|Rprop|SGD)Options)\b'),
                    (OPTIM_STATE_PKG,
                     r'\b(?:OptimizerParamState|OptimizerCloneable\w+ParamState|'
                     r'(?:Adagrad|AdamW?|Adadelta|Adamax|ASGD|LBFGS|NAdam|RAdam|RMSprop|Rprop|SGD)ParamState)\b'),
                    (OPTIM_SCHEDULERS_PKG,
                     r'\b(?:LRScheduler|LRLambda|StepLR|ReduceLROnPlateauScheduler|'
                     r'MultiplicativeLR|MultiStepLR|ConstantLR|LinearLR|ExponentialLR|'
                     r'PolynomialLR|CosineAnnealingLR|CosineAnnealingWarmRestarts|'
                     r'CyclicLR|LambdaLR)\b'),
                ]
                for sub_pkg, pat in optim_sub_checks:
                    imp = f"import {sub_pkg}.*;"
                    if imp in text:
                        continue
                    if f"import {sub_pkg}." in text:
                        continue  # already has single-type imports from subpkg
                    if re.search(pat, text):
                        hw_extras.append(imp)
                if hw_extras:
                    pkg_m_hw = PACKAGE_RE.search(text)
                    block = "\n".join(hw_extras) + "\n"
                    if pkg_m_hw:
                        text = text.replace(
                            pkg_m_hw.group(0) + "\n",
                            pkg_m_hw.group(0) + "\n" + block,
                            1,
                        )
                    else:
                        text = block + text
                if text != text0 and not dry_run:
                    write_text(path, text, dry_run=False)
                elif text != text0 and dry_run:
                    print(f"  [REFS] {path}")
                continue

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
                # Other preserved packages (rpc) legitimately need it.
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





def pkg_to_rel_path(pkg: str) -> Path:
    """org.bytedeco.pytorch.foo.bar -> org/bytedeco/pytorch/foo/bar"""
    return Path(*pkg.split("."))


# Vacated pure-Java package trees (OLD locations only). After high-level moves the
# .java files live under the new package, but empty directory skeletons often
# remain in src/ and — more importantly — in target/classes from incremental
# compiles. maven-jar-plugin packages every directory entry, so those empty
# trees show up in the install jar as phantom packages (data/dataframe,
# utils/transformers, kvcache, info, quantization, …).
#
# ONLY list vacated (old) paths here. Never list current homes such as
# org.bytedeco.pytorch.dataframe / .llm / .utils.plot / .utils.tokenizers
# (tiktoken resources still live under utils/tokenizers).
STALE_PACKAGE_TREES: Tuple[str, ...] = tuple(
    dict.fromkeys(  # preserve order, dedupe
        [old for old, _new in MANUAL_HIGHLEVEL_REWRITES]
        + [
            f"{ROOT_PKG}.kvcache",       # folded into llm.kvcache
            f"{ROOT_PKG}.info",          # never populated / removed
            f"{ROOT_PKG}.quantization",  # real code is llm.quantization
            f"{ROOT_PKG}.transformers",  # legacy resources path
        ]
    )
)

# Absolute minimum depth under a classes/java root: org/bytedeco/pytorch/<seg>
_MIN_STALE_PARTS = len(ROOT_PKG.split(".")) + 1  # e.g. org/bytedeco/pytorch/kvcache


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _dir_file_count(tree: Path) -> int:
    try:
        return sum(1 for f in tree.rglob("*") if f.is_file())
    except OSError:
        return -1  # treat as non-empty / unsafe


def prune_empty_parents(start: Path, stop_at_pkg_root: bool = False) -> int:
    """Remove empty directories walking upward from start.

    Never deletes the org/bytedeco/pytorch package root itself (or anything above).
    """
    n = 0
    cur = start
    root_marker = tuple(ROOT_PKG.split("."))
    while cur is not None and cur.exists() and cur.is_dir():
        parts = cur.parts
        if len(parts) >= 3 and parts[-3:] == root_marker:
            break
        try:
            next(cur.iterdir())
            break  # non-empty
        except StopIteration:
            pass
        except OSError:
            break
        parent = cur.parent
        try:
            cur.rmdir()
            n += 1
        except OSError:
            break
        cur = parent
    return n


def prune_empty_dirs_under(root: Path, dry_run: bool = False) -> int:
    """Bottom-up remove every *empty* directory under root (never removes root, never deletes files)."""
    if root is None or not root.exists():
        return 0
    n = 0
    dirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in dirs:
        try:
            if any(d.iterdir()):
                continue
        except OSError:
            continue
        if dry_run:
            print(f"  [PRUNE-EMPTY] {d}")
            n += 1
            continue
        try:
            d.rmdir()
            n += 1
        except OSError:
            pass
    return n


def remove_stale_package_trees(roots: List[Path], dry_run: bool = False) -> int:
    """Delete vacated package directory trees that contain *zero* files.

    Safety rails:
      * only paths that match STALE_PACKAGE_TREES under each root
      * refuse to delete if any file exists anywhere under the tree
        (protects utils/tokenizers/tiktoken resources, etc.)
      * refuse paths shallower than org/bytedeco/pytorch/<segment>
      * refuse to delete anything outside the given root
    """
    import shutil
    n = 0
    for root in roots:
        if root is None or not root.exists():
            continue
        root = root.resolve()
        for pkg in STALE_PACKAGE_TREES:
            rel = pkg_to_rel_path(pkg)
            tree = (root / rel).resolve()
            if not tree.exists() or not tree.is_dir():
                continue
            if not _is_under_root(tree, root):
                print(f"  [PRUNE-STALE] REFUSE outside root: {tree}")
                continue
            # must be at least org/bytedeco/pytorch/<one-seg>
            try:
                rel_to_root = tree.relative_to(root)
            except ValueError:
                continue
            if len(rel_to_root.parts) < _MIN_STALE_PARTS:
                print(f"  [PRUNE-STALE] REFUSE too shallow: {tree}")
                continue
            nfiles = _dir_file_count(tree)
            if nfiles != 0:
                # nfiles > 0: keep (has resources/classes)
                # nfiles < 0: I/O error — keep
                continue
            if dry_run:
                print(f"  [PRUNE-STALE] {tree}")
            else:
                print(f"  [PRUNE-STALE] removing empty {tree}")
                shutil.rmtree(tree, ignore_errors=False)
            n += 1
    return n


def prune_build_artifacts(
    source_roots: List[Path],
    classes_dirs: List[Path],
    dry_run: bool = False,
) -> None:
    """Drop vacated empty package trees, then sweep remaining empty dirs.

    Never deletes files. Safe to run on src/ and target/classes alike.
    """
    all_roots = [r for r in list(source_roots) + list(classes_dirs) if r is not None]
    n_stale = remove_stale_package_trees(all_roots, dry_run=dry_run)
    n_empty = 0
    for r in all_roots:
        if not r.exists():
            continue
        # Sweep empties under org/bytedeco/pytorch when that subdir exists;
        # otherwise under the root itself. Never pass a missing path.
        pkg_root = r / "org" / "bytedeco" / "pytorch"
        target = pkg_root if pkg_root.is_dir() else r
        n_empty += prune_empty_dirs_under(target, dry_run=dry_run)
    print(f"  pruned stale package trees: {n_stale}, empty dirs: {n_empty}")


def rewrite_highlevel_imports(roots: List[Path], dry_run: bool) -> int:
    """Rewrite FQCNs/imports for hand-written package moves (dataframe, llm, plot, …).

    Safe to re-run: each old prefix is replaced with its new location. Does not
    touch binary resources; only *.java under the given roots (gen + main +
    optional samples/).
    """
    n_files = 0
    for root in roots:
        if root is None or not root.exists():
            continue
        for path in root.rglob("*.java"):
            if not path.is_file():
                continue
            text0 = read_text(path)
            text = text0
            for old, new in MANUAL_HIGHLEVEL_REWRITES:
                text = text.replace(old, new)
            if text != text0:
                n_files += 1
                if dry_run:
                    print(f"  [HIGHLEVEL] {path}")
                else:
                    write_text(path, text, dry_run=False)
    return n_files


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
    ap.add_argument(
        "--classes-dir",
        type=Path,
        action="append",
        default=None,
        help="target/classes (or similar) to prune of vacated empty package trees; repeatable",
    )
    ap.add_argument(
        "--prune-only",
        action="store_true",
        help="Only prune stale/empty package dirs (no peer relocate). Used at prepare-package.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gen_root = args.gen_dir
    main_root = args.main_dir
    classes_dirs: List[Path] = list(args.classes_dir or [])
    # Conventional default when not given: target/classes next to CWD
    if not classes_dirs:
        default_classes = Path("target/classes")
        if default_classes.exists():
            classes_dirs = [default_classes]

    source_roots = [gen_root]
    if main_root.exists():
        source_roots.append(main_root)

    if args.prune_only:
        print(f"Pruning stale/empty package dirs under sources={source_roots} classes={classes_dirs}")
        prune_build_artifacts(source_roots, classes_dirs, dry_run=args.dry_run)
        return 0

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

    # Third pass: hand-written high-level package moves (dataframe, llm.*, plot, …)
    # so samples + main sources stay consistent after re-parse/relocate.
    n_hl = rewrite_highlevel_imports(roots, args.dry_run)
    print(f"  high-level import rewrites: {n_hl} file(s)")

    # Drop vacated pure-Java package skeletons (src + target/classes) so the
    # install jar does not ship empty data/dataframe, utils/transformers, kvcache, …
    prune_build_artifacts(source_roots, classes_dirs, dry_run=args.dry_run)

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
