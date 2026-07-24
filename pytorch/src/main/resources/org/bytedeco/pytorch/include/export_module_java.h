/*
 * Parse-time / JNI shim for bindable torch::jit::ExportModule surface and
 * torch::jit::ToONNX (Graph→ONNX Graph) without walking ModelProto / pybind.
 *
 * Real headers:
 *   torch/csrc/jit/serialization/export.h  — ExportModule, export_opnames
 *   torch/csrc/jit/passes/onnx.h           — ToONNX (plus pybind overloads)
 *
 * At JNI compile time, torch_onnx.init() may swap this shim for the real
 * export.h if desired; symbols live in libtorch_cpu (ExportModule) and
 * libtorch_python (ToONNX).
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>

#include <ostream>
#include <string>
#include <vector>
#include <memory>

namespace torch::jit {

TORCH_API void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false,
    bool use_flatbuffer = false);

TORCH_API void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false,
    bool use_flatbuffer = false);

TORCH_API std::vector<std::string> export_opnames(const Module& m);

// Graph → ONNX Graph conversion (no protobuf). Full signature from passes/onnx.h.
TORCH_API std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::torch::onnx::OperatorExportTypes operator_export_type);

TORCH_API void RemovePrintOps(std::shared_ptr<Graph>& graph);
TORCH_API void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
