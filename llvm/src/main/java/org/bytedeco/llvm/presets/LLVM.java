/*
 * Copyright (C) 2014-2021 Samuel Audet
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

package org.bytedeco.llvm.presets;

import java.io.File;
import java.io.IOException;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.*;

@Properties(inherit = javacpp.class, target = "org.bytedeco.llvm.LLVM", global = "org.bytedeco.llvm.global.LLVM", value = {@Platform(
    value = {"linux", "macosx", "windows"}, define = {"__STDC_LIMIT_MACROS", "__STDC_CONSTANT_MACROS"},
    include = {"<llvm-c/DataTypes.h>", "<llvm-c/Types.h>", "<llvm-c/Support.h>", "<llvm-c/Core.h>", "<llvm-c/Analysis.h>", "<llvm-c/BitReader.h>", "<llvm-c/BitWriter.h>",
               "<llvm-c/DisassemblerTypes.h>", "<llvm-c/Disassembler.h>", "<llvm-c/Initialization.h>", "<llvm-c/IRReader.h>", "<llvm-c/Linker.h>",
               "<llvm-c/lto.h>", "<llvm-c/Object.h>", "<llvm-c/Target.h>", "<llvm-c/TargetMachine.h>", "<llvm-c/ExecutionEngine.h>",
               "<llvm-c/Comdat.h>", "<llvm-c/DebugInfo.h>", "<llvm-c/Error.h>", "<llvm-c/ErrorHandling.h>", "<llvm-c/Orc.h>", "<llvm-c/Remarks.h>",
               "<llvm-c/OrcEE.h>", "<llvm-c/LLJIT.h>", "<llvm-c/Transforms/AggressiveInstCombine.h>", "<llvm-c/Transforms/Coroutines.h>", "<llvm-c/Transforms/InstCombine.h>",
               "<llvm-c/Transforms/IPO.h>", "<llvm-c/Transforms/PassManagerBuilder.h>", "<llvm-c/Transforms/Scalar.h>", "<llvm-c/Transforms/Utils.h>", "<llvm-c/Transforms/Vectorize.h>",
               "<llvm-c/Transforms/PassBuilder.h>", "<polly/LinkAllPasses.h>", "<FullOptimization.h>", "<NamedMetadataOperations.h>", "<TargetStubs.h>"},
    compiler = "cpp14", link = {"LLVM-13", "LTO@.13", "Remarks@.13"}, resource = {"include", "lib", "libexec", "share"}),
        @Platform(value = "macosx", link = {"LLVM", "LTO", "Remarks"}),
        @Platform(value = "windows", link = {"LLVM", "LTO", "Remarks"})})
@NoException
public class LLVM implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "llvm"); }

    private static File packageFile = null;

    /** Returns {@code Loader.cacheResource("/org/bytedeco/llvm/" + Loader.getPlatform())}. */
    public static synchronized File cachePackage() throws IOException {
        if (packageFile != null) {
            return packageFile;
        }
        packageFile = Loader.cacheResource("/org/bytedeco/llvm/" + Loader.getPlatform());
        return packageFile;
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("LLVMOpaqueContext").pointerTypes("LLVMContextRef"))
               .put(new Info("LLVMOpaqueModule").pointerTypes("LLVMModuleRef"))
               .put(new Info("LLVMOpaqueType").pointerTypes("LLVMTypeRef"))
               .put(new Info("LLVMOpaqueValue").pointerTypes("LLVMValueRef"))
               .put(new Info("LLVMOpaqueBasicBlock").pointerTypes("LLVMBasicBlockRef"))
               .put(new Info("LLVMOpaqueMetadata").pointerTypes("LLVMMetadataRef"))
               .put(new Info("LLVMOpaqueNamedMDNode").pointerTypes("LLVMNamedMDNodeRef"))
               .put(new Info("LLVMOpaqueBuilder").pointerTypes("LLVMBuilderRef"))
               .put(new Info("LLVMOpaqueModuleProvider").pointerTypes("LLVMModuleProviderRef"))
               .put(new Info("LLVMOpaqueMemoryBuffer").pointerTypes("LLVMMemoryBufferRef"))
               .put(new Info("LLVMOpaquePassManager").pointerTypes("LLVMPassManagerRef"))
               .put(new Info("LLVMOpaquePassRegistry").pointerTypes("LLVMPassRegistryRef"))
               .put(new Info("LLVMOpaqueUse").pointerTypes("LLVMUseRef"))
               .put(new Info("LLVMOpaqueAttributeRef").pointerTypes("LLVMAttributeRef"))
               .put(new Info("LLVMOpaqueJITEventListener").pointerTypes("LLVMJITEventListenerRef"))
               .put(new Info("LLVMOpaqueBinary").pointerTypes("LLVMBinaryRef"))
               .put(new Info("LLVMOpaqueDiagnosticInfo").pointerTypes("LLVMDiagnosticInfoRef"))
               .put(new Info("LLVMOpaqueTargetData").pointerTypes("LLVMTargetDataRef"))
               .put(new Info("LLVMOpaqueTargetLibraryInfotData").pointerTypes("LLVMTargetLibraryInfoRef"))
               .put(new Info("LLVMOpaqueTargetMachine").pointerTypes("LLVMTargetMachineRef"))
               .put(new Info("LLVMTarget").pointerTypes("LLVMTargetRef"))
               .put(new Info("LLVMOpaqueGenericValue").pointerTypes("LLVMGenericValueRef"))
               .put(new Info("LLVMOpaqueExecutionEngine").pointerTypes("LLVMExecutionEngineRef"))
               .put(new Info("LLVMOpaqueMCJITMemoryManager").pointerTypes("LLVMMCJITMemoryManagerRef"))
               .put(new Info("LLVMComdat").pointerTypes("LLVMComdatRef"))
               .put(new Info("LLVMOpaqueDIBuilder").pointerTypes("LLVMDIBuilderRef"))
               .put(new Info("LLVMOpaqueError").pointerTypes("LLVMErrorRef"))
               .put(new Info("LLVMRemarkOpaqueString").pointerTypes("LLVMRemarkStringRef"))
               .put(new Info("LLVMRemarkOpaqueDebugLoc").pointerTypes("LLVMRemarkDebugLocRef"))
               .put(new Info("LLVMRemarkOpaqueArg").pointerTypes("LLVMRemarkArgRef"))
               .put(new Info("LLVMRemarkOpaqueEntry").pointerTypes("LLVMRemarkEntryRef"))
               .put(new Info("LLVMRemarkOpaqueParser").pointerTypes("LLVMRemarkParserRef"))
               .put(new Info("LLVMOpaqueLTOModule").pointerTypes("lto_module_t"))
               .put(new Info("LLVMOpaqueLTOCodeGenerator").pointerTypes("lto_code_gen_t"))
               .put(new Info("LLVMOpaqueThinLTOCodeGenerator").pointerTypes("thinlto_code_gen_t"))
               .put(new Info("LLVMOpaqueLTOInput").pointerTypes("lto_input_t"))
               .put(new Info("LLVMOpaqueObjectFile").pointerTypes("LLVMObjectFileRef"))
               .put(new Info("LLVMOpaqueSectionIterator").pointerTypes("LLVMSectionIteratorRef"))
               .put(new Info("LLVMOpaqueSymbolIterator").pointerTypes("LLVMSymbolIteratorRef"))
               .put(new Info("LLVMOpaqueRelocationIterator").pointerTypes("LLVMRelocationIteratorRef"))
               .put(new Info("LLVMOpaquePassManagerBuilder").pointerTypes("LLVMPassManagerBuilderRef"))
               .put(new Info("LLVMOrcOpaqueSymbolStringPool").pointerTypes("LLVMOrcSymbolStringPoolRef"))
               .put(new Info("LLVMOrcOpaqueDefinitionGenerator").pointerTypes("LLVMOrcDefinitionGeneratorRef"))
               .put(new Info("LLVMOrcOpaqueResourceTracker").pointerTypes("LLVMOrcResourceTrackerRef"))
               .put(new Info("LLVMOrcOpaqueLookupState").pointerTypes("LLVMOrcLookupStateRef"))
               .put(new Info("LLVMOrcOpaqueSymbolStringPoolEntry").pointerTypes("LLVMOrcSymbolStringPoolEntryRef"))
               .put(new Info("LLVMOrcOpaqueLLJIT").pointerTypes("LLVMOrcLLJITRef"))
               .put(new Info("LLVMOrcOpaqueLLJITBuilder").pointerTypes("LLVMOrcLLJITBuilderRef"))
               .put(new Info("LLVMOrcOpaqueMaterializationUnit").pointerTypes("LLVMOrcMaterializationUnitRef"))
               .put(new Info("LLVMOrcOpaqueJITDylib").pointerTypes("LLVMOrcJITDylibRef"))
               .put(new Info("LLVMOrcOpaqueLookupState").pointerTypes("LLVMOrcLookupStateRef"))
               .put(new Info("LLVMOrcOpaqueThreadSafeContext").pointerTypes("LLVMOrcThreadSafeContextRef"))
               .put(new Info("LLVMOrcOpaqueThreadSafeModule").pointerTypes("LLVMOrcThreadSafeModuleRef"))
               .put(new Info("LLVMOrcOpaqueJITTargetMachineBuilder").pointerTypes("LLVMOrcJITTargetMachineBuilderRef"))
               .put(new Info("LLVMOrcOpaqueObjectLayer").pointerTypes("LLVMOrcObjectLayerRef"))
               .put(new Info("LLVMOrcOpaqueExecutionSession").pointerTypes("LLVMOrcExecutionSessionRef"))
               .put(new Info("LLVMOrcOpaqueMaterializationResponsibility").pointerTypes("LLVMOrcMaterializationResponsibilityRef"))
               .put(new Info("LLVMOrcOpaqueIRTransformLayer").pointerTypes("LLVMOrcIRTransformLayerRef"))
               .put(new Info("LLVMOrcOpaqueObjectTransformLayer").pointerTypes("LLVMOrcObjectTransformLayerRef"))
               .put(new Info("LLVMOrcOpaqueIndirectStubsManager").pointerTypes("LLVMOrcIndirectStubsManagerRef"))
               .put(new Info("LLVMOrcOpaqueLazyCallThroughManager").pointerTypes("LLVMOrcLazyCallThroughManagerRef"))
               .put(new Info("LLVMOrcOpaqueDumpObjects").pointerTypes("LLVMOrcDumpObjectsRef"))
               .put(new Info("LLVMOpaquePassBuilderOptions").pointerTypes("LLVMPassBuilderOptionsRef"))

               .put(new Info("LLVMContextRef").valueTypes("LLVMContextRef").pointerTypes("@ByPtrPtr LLVMContextRef", "@Cast(\"LLVMContextRef*\") PointerPointer"))
               .put(new Info("LLVMModuleRef").valueTypes("LLVMModuleRef").pointerTypes("@ByPtrPtr LLVMModuleRef", "@Cast(\"LLVMModuleRef*\") PointerPointer"))
               .put(new Info("LLVMTypeRef").valueTypes("LLVMTypeRef").pointerTypes("@ByPtrPtr LLVMTypeRef", "@Cast(\"LLVMTypeRef*\") PointerPointer"))
               .put(new Info("LLVMValueRef").valueTypes("LLVMValueRef").pointerTypes("@ByPtrPtr LLVMValueRef", "@Cast(\"LLVMValueRef*\") PointerPointer"))
               .put(new Info("LLVMBasicBlockRef").valueTypes("LLVMBasicBlockRef").pointerTypes("@ByPtrPtr LLVMBasicBlockRef", "@Cast(\"LLVMBasicBlockRef*\") PointerPointer"))
               .put(new Info("LLVMMetadataRef").valueTypes("LLVMMetadataRef").pointerTypes("@ByPtrPtr LLVMMetadataRef", "@Cast(\"LLVMMetadataRef*\") PointerPointer"))
               .put(new Info("LLVMNamedMDNodeRef").valueTypes("LLVMNamedMDNodeRef").pointerTypes("@ByPtrPtr LLVMNamedMDNodeRef", "@Cast(\"LLVMNamedMDNodeRef*\") PointerPointer"))
               .put(new Info("LLVMBuilderRef").valueTypes("LLVMBuilderRef").pointerTypes("@ByPtrPtr LLVMBuilderRef", "@Cast(\"LLVMBuilderRef*\") PointerPointer"))
               .put(new Info("LLVMModuleProviderRef").valueTypes("LLVMModuleProviderRef").pointerTypes("@ByPtrPtr LLVMModuleProviderRef", "@Cast(\"LLVMModuleProviderRef*\") PointerPointer"))
               .put(new Info("LLVMMemoryBufferRef").valueTypes("LLVMMemoryBufferRef").pointerTypes("@ByPtrPtr LLVMMemoryBufferRef", "@Cast(\"LLVMMemoryBufferRef*\") PointerPointer"))
               .put(new Info("LLVMPassManagerRef").valueTypes("LLVMPassManagerRef").pointerTypes("@ByPtrPtr LLVMPassManagerRef", "@Cast(\"LLVMPassManagerRef*\") PointerPointer"))
               .put(new Info("LLVMPassRegistryRef").valueTypes("LLVMPassRegistryRef").pointerTypes("@ByPtrPtr LLVMPassRegistryRef", "@Cast(\"LLVMPassRegistryRef*\") PointerPointer"))
               .put(new Info("LLVMUseRef").valueTypes("LLVMUseRef").pointerTypes("@ByPtrPtr LLVMUseRef", "@Cast(\"LLVMUseRef*\") PointerPointer"))
               .put(new Info("LLVMAttributeRef").valueTypes("LLVMAttributeRef").pointerTypes("@ByPtrPtr LLVMAttributeRef", "@Cast(\"LLVMAttributeRef*\") PointerPointer"))
               .put(new Info("LLVMJITEventListenerRef").valueTypes("LLVMJITEventListenerRef").pointerTypes("@ByPtrPtr LLVMJITEventListenerRef", "@Cast(\"LLVMJITEventListenerRef*\") PointerPointer"))
               .put(new Info("LLVMBinaryRef").valueTypes("LLVMBinaryRef").pointerTypes("@ByPtrPtr LLVMBinaryRef", "@Cast(\"LLVMBinaryRef*\") PointerPointer"))
               .put(new Info("LLVMDiagnosticInfoRef").valueTypes("LLVMDiagnosticInfoRef").pointerTypes("@ByPtrPtr LLVMDiagnosticInfoRef", "@Cast(\"LLVMDiagnosticInfoRef*\") PointerPointer"))
               .put(new Info("LLVMTargetDataRef").valueTypes("LLVMTargetDataRef").pointerTypes("@ByPtrPtr LLVMTargetDataRef", "@Cast(\"LLVMTargetDataRef*\") PointerPointer"))
               .put(new Info("LLVMTargetLibraryInfoRef").valueTypes("LLVMTargetLibraryInfoRef").pointerTypes("@ByPtrPtr LLVMTargetLibraryInfoRef", "@Cast(\"LLVMTargetLibraryInfoRef*\") PointerPointer"))
               .put(new Info("LLVMTargetMachineRef").valueTypes("LLVMTargetMachineRef").pointerTypes("@ByPtrPtr LLVMTargetMachineRef", "@Cast(\"LLVMTargetMachineRef*\") PointerPointer"))
               .put(new Info("LLVMTargetRef").valueTypes("LLVMTargetRef").pointerTypes("@ByPtrPtr LLVMTargetRef", "@Cast(\"LLVMTargetRef*\") PointerPointer"))
               .put(new Info("LLVMGenericValueRef").valueTypes("LLVMGenericValueRef").pointerTypes("@ByPtrPtr LLVMGenericValueRef", "@Cast(\"LLVMGenericValueRef*\") PointerPointer"))
               .put(new Info("LLVMExecutionEngineRef").valueTypes("LLVMExecutionEngineRef").pointerTypes("@ByPtrPtr LLVMExecutionEngineRef", "@Cast(\"LLVMExecutionEngineRef*\") PointerPointer"))
               .put(new Info("LLVMMCJITMemoryManagerRef").valueTypes("LLVMMCJITMemoryManagerRef").pointerTypes("@ByPtrPtr LLVMMCJITMemoryManagerRef", "@Cast(\"LLVMMCJITMemoryManagerRef*\") PointerPointer"))
               .put(new Info("LLVMComdatRef").valueTypes("LLVMComdatRef").pointerTypes("@ByPtrPtr LLVMComdatRef", "@Cast(\"LLVMComdatRef*\") PointerPointer"))
               .put(new Info("LLVMDIBuilderRef").valueTypes("LLVMDIBuilderRef").pointerTypes("@ByPtrPtr LLVMDIBuilderRef", "@Cast(\"LLVMDIBuilderRef*\") PointerPointer"))
               .put(new Info("LLVMErrorRef").valueTypes("LLVMErrorRef").pointerTypes("@ByPtrPtr LLVMErrorRef", "@Cast(\"LLVMErrorRef*\") PointerPointer"))
               .put(new Info("LLVMRemarkStringRef").valueTypes("LLVMRemarkStringRef").pointerTypes("@ByPtrPtr LLVMRemarkStringRef", "@Cast(\"LLVMRemarkStringRef*\") PointerPointer"))
               .put(new Info("LLVMRemarkDebugLocRef").valueTypes("LLVMRemarkDebugLocRef").pointerTypes("@ByPtrPtr LLVMRemarkDebugLocRef", "@Cast(\"LLVMRemarkDebugLocRef*\") PointerPointer"))
               .put(new Info("LLVMRemarkArgRef").valueTypes("LLVMRemarkArgRef").pointerTypes("@ByPtrPtr LLVMRemarkArgRef", "@Cast(\"LLVMRemarkArgRef*\") PointerPointer"))
               .put(new Info("LLVMRemarkEntryRef").valueTypes("LLVMRemarkEntryRef").pointerTypes("@ByPtrPtr LLVMRemarkEntryRef", "@Cast(\"LLVMRemarkEntryRef*\") PointerPointer"))
               .put(new Info("LLVMRemarkParserRef").valueTypes("LLVMRemarkParserRef").pointerTypes("@ByPtrPtr LLVMRemarkParserRef", "@Cast(\"LLVMRemarkParserRef*\") PointerPointer"))
               .put(new Info("lto_module_t").valueTypes("lto_module_t").pointerTypes("@ByPtrPtr lto_module_t", "@Cast(\"lto_module_t*\") PointerPointer"))
               .put(new Info("lto_code_gen_t").valueTypes("lto_code_gen_t").pointerTypes("@ByPtrPtr lto_code_gen_t", "@Cast(\"lto_code_gen_t*\") PointerPointer"))
               .put(new Info("thinlto_code_gen_t").valueTypes("thinlto_code_gen_t").pointerTypes("@ByPtrPtr thinlto_code_gen_t", "@Cast(\"thinlto_code_gen_t*\") PointerPointer"))
               .put(new Info("lto_input_t").valueTypes("lto_input_t").pointerTypes("@ByPtrPtr lto_input_t", "@Cast(\"lto_input_t*\") PointerPointer"))
               .put(new Info("LLVMObjectFileRef").valueTypes("LLVMObjectFileRef").pointerTypes("@ByPtrPtr LLVMObjectFileRef", "@Cast(\"LLVMObjectFileRef*\") PointerPointer"))
               .put(new Info("LLVMSectionIteratorRef").valueTypes("LLVMSectionIteratorRef").pointerTypes("@ByPtrPtr LLVMSectionIteratorRef", "@Cast(\"LLVMSectionIteratorRef*\") PointerPointer"))
               .put(new Info("LLVMSymbolIteratorRef").valueTypes("LLVMSymbolIteratorRef").pointerTypes("@ByPtrPtr LLVMSymbolIteratorRef", "@Cast(\"LLVMSymbolIteratorRef*\") PointerPointer"))
               .put(new Info("LLVMRelocationIteratorRef").valueTypes("LLVMRelocationIteratorRef").pointerTypes("@ByPtrPtr LLVMRelocationIteratorRef", "@Cast(\"LLVMRelocationIteratorRef*\") PointerPointer"))
               .put(new Info("LLVMPassManagerBuilderRef").valueTypes("LLVMPassManagerBuilderRef").pointerTypes("@ByPtrPtr LLVMPassManagerBuilderRef", "@Cast(\"LLVMPassManagerBuilderRef*\") PointerPointer"))
               .put(new Info("LLVMOrcSymbolStringPoolRef").valueTypes("LLVMOrcSymbolStringPoolRef").pointerTypes("@ByPtrPtr LLVMOrcSymbolStringPoolRef", "@Cast(\"LLVMOrcSymbolStringPoolRef*\") PointerPointer"))
               .put(new Info("LLVMOrcDefinitionGeneratorRef").valueTypes("LLVMOrcDefinitionGeneratorRef").pointerTypes("@ByPtrPtr LLVMOrcDefinitionGeneratorRef", "@Cast(\"LLVMOrcDefinitionGeneratorRef*\") PointerPointer"))
               .put(new Info("LLVMOrcResourceTrackerRef").valueTypes("LLVMOrcResourceTrackerRef").pointerTypes("@ByPtrPtr LLVMOrcResourceTrackerRef", "@Cast(\"LLVMOrcResourceTrackerRef*\") PointerPointer"))
               .put(new Info("LLVMOrcLookupStateRef").valueTypes("LLVMOrcLookupStateRef").pointerTypes("@ByPtrPtr LLVMOrcLookupStateRef", "@Cast(\"LLVMOrcLookupStateRef*\") PointerPointer"))
               .put(new Info("LLVMOrcSymbolStringPoolEntryRef").valueTypes("LLVMOrcSymbolStringPoolEntryRef").pointerTypes("@ByPtrPtr LLVMOrcSymbolStringPoolEntryRef", "@Cast(\"LLVMOrcSymbolStringPoolEntryRef*\") PointerPointer"))
               .put(new Info("LLVMOrcLLJITRef").valueTypes("LLVMOrcLLJITRef").pointerTypes("@ByPtrPtr LLVMOrcLLJITRef", "@Cast(\"LLVMOrcLLJITRef*\") PointerPointer"))
               .put(new Info("LLVMOrcLLJITBuilderRef").valueTypes("LLVMOrcLLJITBuilderRef").pointerTypes("@ByPtrPtr LLVMOrcLLJITBuilderRef", "@Cast(\"LLVMOrcLLJITBuilderRef*\") PointerPointer"))
               .put(new Info("LLVMOrcMaterializationUnitRef").valueTypes("LLVMOrcMaterializationUnitRef").pointerTypes("@ByPtrPtr LLVMOrcMaterializationUnitRef", "@Cast(\"LLVMOrcMaterializationUnitRef*\") PointerPointer"))
               .put(new Info("LLVMOrcJITDylibRef").valueTypes("LLVMOrcJITDylibRef").pointerTypes("@ByPtrPtr LLVMOrcJITDylibRef", "@Cast(\"LLVMOrcJITDylibRef*\") PointerPointer"))
               .put(new Info("LLVMOrcLookupStateRef").valueTypes("LLVMOrcLookupStateRef").pointerTypes("@ByPtrPtr LLVMOrcLookupStateRef", "@Cast(\"LLVMOrcLookupStateRef*\") PointerPointer"))
               .put(new Info("LLVMOrcThreadSafeContextRef").valueTypes("LLVMOrcThreadSafeContextRef").pointerTypes("@ByPtrPtr LLVMOrcThreadSafeContextRef", "@Cast(\"LLVMOrcThreadSafeContextRef*\") PointerPointer"))
               .put(new Info("LLVMOrcThreadSafeModuleRef").valueTypes("LLVMOrcThreadSafeModuleRef").pointerTypes("@ByPtrPtr LLVMOrcThreadSafeModuleRef", "@Cast(\"LLVMOrcThreadSafeModuleRef*\") PointerPointer"))
               .put(new Info("LLVMOrcJITTargetMachineBuilderRef").valueTypes("LLVMOrcJITTargetMachineBuilderRef").pointerTypes("@ByPtrPtr LLVMOrcJITTargetMachineBuilderRef", "@Cast(\"LLVMOrcJITTargetMachineBuilderRef*\") PointerPointer"))
               .put(new Info("LLVMOrcObjectLayerRef").valueTypes("LLVMOrcObjectLayerRef").pointerTypes("@ByPtrPtr LLVMOrcObjectLayerRef", "@Cast(\"LLVMOrcObjectLayerRef*\") PointerPointer"))
               .put(new Info("LLVMOrcExecutionSessionRef").valueTypes("LLVMOrcExecutionSessionRef").pointerTypes("@ByPtrPtr LLVMOrcExecutionSessionRef", "@Cast(\"LLVMOrcExecutionSessionRef*\") PointerPointer"))
               .put(new Info("LLVMOrcMaterializationResponsibilityRef").valueTypes("LLVMOrcMaterializationResponsibilityRef").pointerTypes("@ByPtrPtr LLVMOrcMaterializationResponsibilityRef", "@Cast(\"LLVMOrcMaterializationResponsibilityRef*\") PointerPointer"))
               .put(new Info("LLVMOrcIRTransformLayerRef").valueTypes("LLVMOrcIRTransformLayerRef").pointerTypes("@ByPtrPtr LLVMOrcIRTransformLayerRef", "@Cast(\"LLVMOrcIRTransformLayerRef*\") PointerPointer"))
               .put(new Info("LLVMOrcObjectTransformLayerRef").valueTypes("LLVMOrcObjectTransformLayerRef").pointerTypes("@ByPtrPtr LLVMOrcObjectTransformLayerRef", "@Cast(\"LLVMOrcObjectTransformLayerRef*\") PointerPointer"))
               .put(new Info("LLVMOrcIndirectStubsManagerRef").valueTypes("LLVMOrcIndirectStubsManagerRef").pointerTypes("@ByPtrPtr LLVMOrcIndirectStubsManagerRef", "@Cast(\"LLVMOrcIndirectStubsManagerRef*\") PointerPointer"))
               .put(new Info("LLVMOrcLazyCallThroughManagerRef").valueTypes("LLVMOrcLazyCallThroughManagerRef").pointerTypes("@ByPtrPtr LLVMOrcLazyCallThroughManagerRef", "@Cast(\"LLVMOrcLazyCallThroughManagerRef*\") PointerPointer"))
               .put(new Info("LLVMOrcDumpObjectsRef").valueTypes("LLVMOrcDumpObjectsRef").pointerTypes("@ByPtrPtr LLVMOrcDumpObjectsRef", "@Cast(\"LLVMOrcDumpObjectsRef*\") PointerPointer"))
               .put(new Info("LLVMPassBuilderOptionsRef").valueTypes("LLVMPassBuilderOptionsRef").pointerTypes("@ByPtrPtr LLVMPassBuilderOptionsRef", "@Cast(\"LLVMPassBuilderOptionsRef*\") PointerPointer"))

               .put(new Info("LLVM_C_EXTERN_C_BEGIN").cppText("#define LLVM_C_EXTERN_C_BEGIN").cppTypes())
               .put(new Info("LLVM_C_EXTERN_C_END").cppText("#define LLVM_C_EXTERN_C_END").cppTypes())
               .put(new Info("INT64_MIN").cppTypes("long").translate())
               .put(new Info("HUGE_VALF").cppTypes("float").translate(false))
               .put(new Info("LLVMErrorTypeId").annotations("@Const").valueTypes("LLVMErrorTypeId"))
               .put(new Info("defined(_MSC_VER) && !defined(inline)").define(false))
               .put(new Info("GPU_CODEGEN").define(false))
               // These things were never actually implemented, see http://llvm.org/PR41362
               .put(new Info("LLVMConstGEP2", "LLVMConstInBoundsGEP2", "LLVMOrcObjectLayerAddObjectFileWithRT").skip());
    }
}
