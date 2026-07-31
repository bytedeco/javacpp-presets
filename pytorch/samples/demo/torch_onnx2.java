///*
// * Copyright (C) 2026 Hervé Guillemet, Samuel Audet
// *
// * Licensed either under the Apache License, Version 2.0, or (at your option)
// * under the terms of the GNU General Public License as published by
// * the Free Software Foundation (subject to the "Classpath" exception),
// * either version 2, or any later version (collectively, the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *     http://www.gnu.org/licenses/
// *     http://www.gnu.org/software/classpath/license.html
// *
// * or as provided in the LICENSE.txt file that accompanied this code.
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//package org.bytedeco.pytorch.presets;
//import org.bytedeco.pytorch.jit.*;
//
//import org.bytedeco.javacpp.ClassProperties;
//import org.bytedeco.javacpp.LoadEnabled;
//import org.bytedeco.javacpp.annotation.Platform;
//import org.bytedeco.javacpp.annotation.Properties;
//import org.bytedeco.javacpp.tools.Info;
//import org.bytedeco.javacpp.tools.InfoMap;
//import org.bytedeco.javacpp.tools.InfoMapper;
//
///**
// * Maps the pure-C++ surface of {@code torch::onnx} and the bindable
// * TorchScript load / export helpers used with ONNX workflows.
// *
// * <p>Headers intentionally <em>not</em> included:
// * <ul>
// *   <li>{@code torch/csrc/onnx/init.h} — pulls {@code torch/csrc/utils/pybind.h}</li>
// *   <li>{@code torch/csrc/onnx/back_compat.h} — needs {@code onnx/onnx_pb.h}</li>
// *   <li>{@code torch/csrc/jit/serialization/export.h} in full — pulls mobile /
// *       python_print / ModelProto; we parse a lightweight shim
// *       ({@code export_module_java.h}) that declares {@code ExportModule},
// *       {@code export_opnames}, and {@code ToONNX} only.</li>
// *   <li>{@code torch/csrc/jit/passes/onnx.h} pybind overloads
// *       ({@code BlockToONNX}, {@code NodeToONNX}) — skipped.</li>
// * </ul>
// *
// * <p>Native symbols for {@code ExportModule}/{@code export_opnames} live in
// * {@code torch_cpu}; {@code ToONNX} lives in {@code torch_python}. The JNI
// * trampolines are built into {@code jnitorch_onnx} and link both.
// *
// * <p>Convenience Java wrappers {@code loadModule} / {@code exportModule} are
// * injected into the global class so callers can load a TorchScript module
// * and export it without dropping down to {@code torch.global} / {@code JitModule}.
// * Loading a raw {@code .onnx} protobuf for inference is the domain of the
// * separate {@code org.bytedeco.onnxruntime} preset (libtorch has no C++ ONNX
// * Runtime session API).
// */
//@Properties(
//    inherit = torch.class,
//    value = @Platform(
//        value = {"linux", "macosx", "windows"},
//        compiler = "cpp20",
//        include = {
//            "torch/csrc/onnx/onnx.h",
//            // Lightweight ExportModule / ToONNX surface (no ModelProto / pybind).
//            "export_module_java.h",
//            // "torch/csrc/onnx/back_compat.h", // needs onnx/onnx_pb.h
//            // "torch/csrc/onnx/init.h",        // needs Python/pybind
//        },
//        link = { "c10", "torch", "torch_cpu", "torch_python" }
//    ),
//    target = "org.bytedeco.pytorch.onnx",
//    global = "org.bytedeco.pytorch.global.torch_onnx"
//)
//public class torch_onnx implements LoadEnabled, InfoMapper {
//
//    @Override
//    public void init(ClassProperties properties) {
//        // Do not call torch.initIncludes — keep @Platform.include as source of truth.
//        // export_module_java.h / onnx.h need no Python.h at parse or compile time;
//        // ToONNX / ExportModule symbols are resolved at link time via torch_cpu /
//        // torch_python.
//    }
//
//    @Override
//    public void map(InfoMap infoMap) {
//        torch.sharedMap(infoMap);
//
//        infoMap
//            .put(new Info("torch::onnx::OperatorExportTypes")
//                    .enumerate().valueTypes("OperatorExportTypes"))
//            .put(new Info("torch::onnx::TrainingMode")
//                    .enumerate().valueTypes("TrainingMode"))
//            .put(new Info("torch::onnx::kOnnxNodeNameAttribute").javaText(
//                "/** Attribute name used for ONNX node names (torch::onnx::kOnnxNodeNameAttribute). */\n"
//              + "public static final String kOnnxNodeNameAttribute = \"onnx_name\";\n"))
//
//            // Export / convert surface from export_module_java.h
//            .put(new Info("torch::jit::ExportModule").javaNames("ExportModule"))
//            .put(new Info("torch::jit::export_opnames").javaNames("export_opnames"))
//            .put(new Info("torch::jit::ToONNX").javaNames("ToONNX"))
//            .put(new Info("torch::jit::RemovePrintOps").javaNames("RemovePrintOps"))
//            .put(new Info("torch::jit::PreprocessCaffe2Ops").javaNames("PreprocessCaffe2Ops"))
//
//            // Convenience wrappers: load TorchScript + export via JitModule.save
//            // / free ExportModule. Placed in the global class via null-key emit.
//            .put(new Info((String) null).javaText(
//                "/**\n"
//              + " * Load a serialized TorchScript {@code JitModule} from {@code filename}.\n"
//              + " * Equivalent to {@code torch.jit.load} / {@code org.bytedeco.pytorch.global.torch.load}.\n"
//              + " * For raw {@code .onnx} protobuf inference, use {@code org.bytedeco.onnxruntime}.\n"
//              + " */\n"
//              + "public static org.bytedeco.pytorch.jit.JitModule loadModule(String filename) {\n"
//              + "    return org.bytedeco.pytorch.global.torch.load(filename);\n"
//              + "}\n"
//              + "\n"
//              + "/**\n"
//              + " * Load a TorchScript module onto {@code device}.\n"
//              + " */\n"
//              + "public static org.bytedeco.pytorch.jit.JitModule loadModule(String filename,\n"
//              + "        org.bytedeco.pytorch.DeviceOptional device) {\n"
//              + "    return org.bytedeco.pytorch.global.torch.load(filename, device, true);\n"
//              + "}\n"
//              + "\n"
//              + "/**\n"
//              + " * Export {@code module} to {@code filename} via {@code torch::jit::ExportModule}.\n"
//              + " * Prefer this free function when you need bytecode/flatbuffer flags;\n"
//              + " * otherwise {@code module.save(filename)} is equivalent for the default path.\n"
//              + " */\n"
//              + "public static void exportModule(org.bytedeco.pytorch.jit.JitModule module, String filename) {\n"
//              + "    ExportModule(module, filename);\n"
//              + "}\n"
//              + "\n"
//              + "/**\n"
//              + " * Export {@code module} with explicit format flags.\n"
//              + " */\n"
//              + "public static void exportModule(org.bytedeco.pytorch.jit.JitModule module, String filename,\n"
//              + "        boolean bytecode_format, boolean save_mobile_debug_info, boolean use_flatbuffer) {\n"
//              + "    ExportModule(module, filename,\n"
//              + "        new org.bytedeco.pytorch.ExtraFilesMap(),\n"
//              + "        bytecode_format, save_mobile_debug_info, use_flatbuffer);\n"
//              + "}\n"
//            ))
//        ;
//    }
//}
