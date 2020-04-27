/*
 * Copyright (C) 2020 Yu Kobayashi
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

#ifndef FULL_OPTIMIZATION_H
#define FULL_OPTIMIZATION_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm-c/Types.h"

using namespace llvm;

/**
 * This function does the standard LLVM optimization.
 * This function is based on main() of llvm/tools/opt/opt.cpp.
 * Use LLVMGetHostCPUName() for the cpu argument.
 */
void optimizeModule(
    LLVMModuleRef moduleRef,
    const char* cpu,
    unsigned optLevel,
    unsigned sizeLevel
) {
    Module *module = unwrap(moduleRef);

    std::string error;
    EngineBuilder engineBuilder;
    auto machine = std::unique_ptr<TargetMachine>(engineBuilder
        .setMCPU(cpu)
        .setErrorStr(&error)
        .selectTarget());
    if (!machine) {
        throw std::runtime_error(error);
    }

    module->setTargetTriple(machine->getTargetTriple().str());
    module->setDataLayout(machine->createDataLayout());

    legacy::PassManager passes;
    passes.add(new TargetLibraryInfoWrapperPass(machine->getTargetTriple()));
    passes.add(createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

    legacy::FunctionPassManager fnPasses(module);
    fnPasses.add(createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

    // AddOptimizationPasses
    PassManagerBuilder builder1;
    builder1.OptLevel = optLevel;
    builder1.SizeLevel = sizeLevel;
    builder1.Inliner = createFunctionInliningPass(optLevel, sizeLevel, false);
    builder1.LoopVectorize = optLevel > 1 && sizeLevel < 2;
    builder1.SLPVectorize = optLevel > 1 && sizeLevel < 2;
    machine->adjustPassManager(builder1);
    builder1.populateFunctionPassManager(fnPasses);
    builder1.populateModulePassManager(passes);

    // AddStandardLinkPasses
    PassManagerBuilder builder2;
    builder2.VerifyInput = true;
    builder2.Inliner = createFunctionInliningPass();
    builder2.populateLTOPassManager(passes);

    fnPasses.doInitialization();
    for (Function &func : *module) {
        fnPasses.run(func);
    }
    fnPasses.doFinalization();

    passes.add(createVerifierPass());
    passes.run(*module);
}

/**
 * This function is similar to LLVMCreateJITCompilerForModule() but does CPU specific optimization.
 * Use LLVMGetHostCPUName() for the cpu argument.
 */
void createOptimizedJITCompilerForModule(
    LLVMExecutionEngineRef *outJIT,
    LLVMModuleRef moduleRef,
    const char* cpu,
    unsigned optLevel
) {
    std::string error;
    EngineBuilder engineBuilder(std::unique_ptr<Module>(unwrap(moduleRef)));
    ExecutionEngine *ee = engineBuilder
        .setEngineKind(EngineKind::JIT)
        .setMCPU(cpu)
        .setOptLevel(static_cast<CodeGenOpt::Level>(optLevel))
        .setErrorStr(&error)
        .create();
    if (ee == nullptr) {
        throw std::runtime_error(error);
    }
    ee->finalizeObject();
    *outJIT = wrap(ee);
}

#endif
