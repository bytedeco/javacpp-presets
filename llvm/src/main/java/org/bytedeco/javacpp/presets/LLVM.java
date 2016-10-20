/*
 * Copyright (C) 2014-2016 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(target = "org.bytedeco.javacpp.LLVM", value = @Platform(value = {"linux-x86", "macosx"}, define = {"__STDC_LIMIT_MACROS", "__STDC_CONSTANT_MACROS"},
    include = {"<llvm-c/Types.h>", "<llvm-c/Support.h>", "<llvm-c/Core.h>", "<llvm-c/Analysis.h>", "<llvm-c/BitReader.h>", "<llvm-c/BitWriter.h>",
               "<llvm-c/Disassembler.h>", "<llvm-c/Initialization.h>", "<llvm-c/IRReader.h>", "<llvm-c/Linker.h>", "<llvm-c/LinkTimeOptimizer.h>",
               "<llvm-c/lto.h>", "<llvm-c/Object.h>", "<llvm-c/Target.h>", "<llvm-c/TargetMachine.h>", "<llvm-c/ExecutionEngine.h>",
               "<llvm-c/Transforms/IPO.h>", "<llvm-c/Transforms/PassManagerBuilder.h>", "<llvm-c/Transforms/Scalar.h>", "<llvm-c/Transforms/Vectorize.h>"},
    compiler = "cpp11", link = {"LLVM-3.9", "LTO"}))
public class LLVM implements InfoMapper {
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LLVMOpaqueContext").pointerTypes("LLVMContextRef"))
               .put(new Info("LLVMOpaqueModule").pointerTypes("LLVMModuleRef"))
               .put(new Info("LLVMOpaqueType").pointerTypes("LLVMTypeRef"))
               .put(new Info("LLVMOpaqueValue").pointerTypes("LLVMValueRef"))
               .put(new Info("LLVMOpaqueBasicBlock").pointerTypes("LLVMBasicBlockRef"))
               .put(new Info("LLVMOpaqueBuilder").pointerTypes("LLVMBuilderRef"))
               .put(new Info("LLVMOpaqueModuleProvider").pointerTypes("LLVMModuleProviderRef"))
               .put(new Info("LLVMOpaqueMemoryBuffer").pointerTypes("LLVMMemoryBufferRef"))
               .put(new Info("LLVMOpaquePassManager").pointerTypes("LLVMPassManagerRef"))
               .put(new Info("LLVMOpaquePassRegistry").pointerTypes("LLVMPassRegistryRef"))
               .put(new Info("LLVMOpaqueUse").pointerTypes("LLVMUseRef"))
               .put(new Info("LLVMOpaqueAttributeRef").pointerTypes("LLVMAttributeRef"))
               .put(new Info("LLVMOpaqueDiagnosticInfo").pointerTypes("LLVMDiagnosticInfoRef"))
               .put(new Info("LLVMOpaqueTargetData").pointerTypes("LLVMTargetDataRef"))
               .put(new Info("LLVMOpaqueTargetLibraryInfotData").pointerTypes("LLVMTargetLibraryInfoRef"))
               .put(new Info("LLVMOpaqueTargetMachine").pointerTypes("LLVMTargetMachineRef"))
               .put(new Info("LLVMTarget").pointerTypes("LLVMTargetRef"))
               .put(new Info("LLVMOpaqueGenericValue").pointerTypes("LLVMGenericValueRef"))
               .put(new Info("LLVMOpaqueExecutionEngine").pointerTypes("LLVMExecutionEngineRef"))
               .put(new Info("LLVMOpaqueMCJITMemoryManager").pointerTypes("LLVMMCJITMemoryManagerRef"))
               .put(new Info("LLVMOpaqueLTOModule").pointerTypes("lto_module_t"))
               .put(new Info("LLVMOpaqueLTOCodeGenerator").pointerTypes("lto_code_gen_t"))
               .put(new Info("LLVMOpaqueThinLTOCodeGenerator").pointerTypes("thinlto_code_gen_t"))
               .put(new Info("LLVMOpaqueObjectFile").pointerTypes("LLVMObjectFileRef"))
               .put(new Info("LLVMOpaqueSectionIterator").pointerTypes("LLVMSectionIteratorRef"))
               .put(new Info("LLVMOpaqueSymbolIterator").pointerTypes("LLVMSymbolIteratorRef"))
               .put(new Info("LLVMOpaqueRelocationIterator").pointerTypes("LLVMRelocationIteratorRef"))
               .put(new Info("LLVMOpaquePassManagerBuilder").pointerTypes("LLVMPassManagerBuilderRef"))

               .put(new Info("LLVMContextRef").valueTypes("LLVMContextRef").pointerTypes("@ByPtrPtr LLVMContextRef", "@Cast(\"LLVMContextRef*\") PointerPointer"))
               .put(new Info("LLVMModuleRef").valueTypes("LLVMModuleRef").pointerTypes("@ByPtrPtr LLVMModuleRef", "@Cast(\"LLVMModuleRef*\") PointerPointer"))
               .put(new Info("LLVMTypeRef").valueTypes("LLVMTypeRef").pointerTypes("@ByPtrPtr LLVMTypeRef", "@Cast(\"LLVMTypeRef*\") PointerPointer"))
               .put(new Info("LLVMValueRef").valueTypes("LLVMValueRef").pointerTypes("@ByPtrPtr LLVMValueRef", "@Cast(\"LLVMValueRef*\") PointerPointer"))
               .put(new Info("LLVMBasicBlockRef").valueTypes("LLVMBasicBlockRef").pointerTypes("@ByPtrPtr LLVMBasicBlockRef", "@Cast(\"LLVMBasicBlockRef*\") PointerPointer"))
               .put(new Info("LLVMBuilderRef").valueTypes("LLVMBuilderRef").pointerTypes("@ByPtrPtr LLVMBuilderRef", "@Cast(\"LLVMBuilderRef*\") PointerPointer"))
               .put(new Info("LLVMModuleProviderRef").valueTypes("LLVMModuleProviderRef").pointerTypes("@ByPtrPtr LLVMModuleProviderRef", "@Cast(\"LLVMModuleProviderRef*\") PointerPointer"))
               .put(new Info("LLVMMemoryBufferRef").valueTypes("LLVMMemoryBufferRef").pointerTypes("@ByPtrPtr LLVMMemoryBufferRef", "@Cast(\"LLVMMemoryBufferRef*\") PointerPointer"))
               .put(new Info("LLVMPassManagerRef").valueTypes("LLVMPassManagerRef").pointerTypes("@ByPtrPtr LLVMPassManagerRef", "@Cast(\"LLVMPassManagerRef*\") PointerPointer"))
               .put(new Info("LLVMPassRegistryRef").valueTypes("LLVMPassRegistryRef").pointerTypes("@ByPtrPtr LLVMPassRegistryRef", "@Cast(\"LLVMPassRegistryRef*\") PointerPointer"))
               .put(new Info("LLVMUseRef").valueTypes("LLVMUseRef").pointerTypes("@ByPtrPtr LLVMUseRef", "@Cast(\"LLVMUseRef*\") PointerPointer"))
               .put(new Info("LLVMAttributeRef").valueTypes("LLVMAttributeRef").pointerTypes("@ByPtrPtr LLVMAttributeRef", "@Cast(\"LLVMAttributeRef*\") PointerPointer"))
               .put(new Info("LLVMDiagnosticInfoRef").valueTypes("LLVMDiagnosticInfoRef").pointerTypes("@ByPtrPtr LLVMDiagnosticInfoRef", "@Cast(\"LLVMDiagnosticInfoRef*\") PointerPointer"))
               .put(new Info("LLVMTargetDataRef").valueTypes("LLVMTargetDataRef").pointerTypes("@ByPtrPtr LLVMTargetDataRef", "@Cast(\"LLVMTargetDataRef*\") PointerPointer"))
               .put(new Info("LLVMTargetLibraryInfoRef").valueTypes("LLVMTargetLibraryInfoRef").pointerTypes("@ByPtrPtr LLVMTargetLibraryInfoRef", "@Cast(\"LLVMTargetLibraryInfoRef*\") PointerPointer"))
               .put(new Info("LLVMTargetMachineRef").valueTypes("LLVMTargetMachineRef").pointerTypes("@ByPtrPtr LLVMTargetMachineRef", "@Cast(\"LLVMTargetMachineRef*\") PointerPointer"))
               .put(new Info("LLVMTargetRef").valueTypes("LLVMTargetRef").pointerTypes("@ByPtrPtr LLVMTargetRef", "@Cast(\"LLVMTargetRef*\") PointerPointer"))
               .put(new Info("LLVMGenericValueRef").valueTypes("LLVMGenericValueRef").pointerTypes("@ByPtrPtr LLVMGenericValueRef", "@Cast(\"LLVMGenericValueRef*\") PointerPointer"))
               .put(new Info("LLVMExecutionEngineRef").valueTypes("LLVMExecutionEngineRef").pointerTypes("@ByPtrPtr LLVMExecutionEngineRef", "@Cast(\"LLVMExecutionEngineRef*\") PointerPointer"))
               .put(new Info("LLVMMCJITMemoryManagerRef").valueTypes("LLVMMCJITMemoryManagerRef").pointerTypes("@ByPtrPtr LLVMMCJITMemoryManagerRef", "@Cast(\"LLVMMCJITMemoryManagerRef*\") PointerPointer"))
               .put(new Info("lto_module_t").valueTypes("lto_module_t").pointerTypes("@ByPtrPtr lto_module_t", "@Cast(\"lto_module_t*\") PointerPointer"))
               .put(new Info("lto_code_gen_t").valueTypes("lto_code_gen_t").pointerTypes("@ByPtrPtr lto_code_gen_t", "@Cast(\"lto_code_gen_t*\") PointerPointer"))
               .put(new Info("thinlto_code_gen_t").valueTypes("thinlto_code_gen_t").pointerTypes("@ByPtrPtr thinlto_code_gen_t", "@Cast(\"thinlto_code_gen_t*\") PointerPointer"))
               .put(new Info("LLVMObjectFileRef").valueTypes("LLVMObjectFileRef").pointerTypes("@ByPtrPtr LLVMObjectFileRef", "@Cast(\"LLVMObjectFileRef*\") PointerPointer"))
               .put(new Info("LLVMSectionIteratorRef").valueTypes("LLVMSectionIteratorRef").pointerTypes("@ByPtrPtr LLVMSectionIteratorRef", "@Cast(\"LLVMSectionIteratorRef*\") PointerPointer"))
               .put(new Info("LLVMSymbolIteratorRef").valueTypes("LLVMSymbolIteratorRef").pointerTypes("@ByPtrPtr LLVMSymbolIteratorRef", "@Cast(\"LLVMSymbolIteratorRef*\") PointerPointer"))
               .put(new Info("LLVMRelocationIteratorRef").valueTypes("LLVMRelocationIteratorRef").pointerTypes("@ByPtrPtr LLVMRelocationIteratorRef", "@Cast(\"LLVMRelocationIteratorRef*\") PointerPointer"))
               .put(new Info("LLVMPassManagerBuilderRef").valueTypes("LLVMPassManagerBuilderRef").pointerTypes("@ByPtrPtr LLVMPassManagerBuilderRef", "@Cast(\"LLVMPassManagerBuilderRef*\") PointerPointer"))

               .put(new Info("defined(_MSC_VER) && !defined(inline)").define(false))
               .put(new Info("llvm_optimize_modules", "llvm_destroy_optimizer", "llvm_read_object_file", "llvm_create_optimizer").skip());
    }
}
