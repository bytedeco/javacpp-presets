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

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.llvm.LLVM.LLVMBasicBlockRef;
import org.bytedeco.llvm.LLVM.LLVMBuilderRef;
import org.bytedeco.llvm.LLVM.LLVMContextRef;
import org.bytedeco.llvm.LLVM.LLVMModuleRef;
import org.bytedeco.llvm.LLVM.LLVMTargetMachineRef;
import org.bytedeco.llvm.LLVM.LLVMTargetRef;
import org.bytedeco.llvm.LLVM.LLVMTypeRef;
import org.bytedeco.llvm.LLVM.LLVMValueRef;

import static org.bytedeco.llvm.global.LLVM.*;

/**
 * Sample for generating both LLVM bitcode and relocatable object file from an LLVM module
 *
 * The generated module (and objec file) will have a single sum function, which returns
 * the sum of two integers.
 *
 * declare i32 @sum(i32 %lhs, i32 %rhs)
 *
 * This sample contains code for the following steps
 *
 * 1. Initializing required LLVM components
 * 2. Generating LLVM IR for a sum function
 * 3. Write the LLVM bitcode to a file on disk
 * 4. Write the relocatable object file to a file on disk
 * 5. Dispose of the allocated resources
 */
public class EmitBitcode {
    public static final BytePointer error = new BytePointer();

    public static void main(String[] args) {
        // Stage 1: Initialize LLVM components
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeDisassembler();
        LLVMInitializeNativeTarget();

        // Stage 2: Build the sum function
        LLVMContextRef context = LLVMContextCreate();
        LLVMModuleRef module = LLVMModuleCreateWithNameInContext("sum", context);
        LLVMBuilderRef builder = LLVMCreateBuilderInContext(context);
        LLVMTypeRef i32Type = LLVMInt32TypeInContext(context);
        PointerPointer<Pointer> sumArgumentTypes = new PointerPointer<>(2)
            .put(0, i32Type)
            .put(1, i32Type);
        LLVMTypeRef sumType = LLVMFunctionType(i32Type, sumArgumentTypes, /* argumentCount */ 2, /* isVariadic */ 0);

        LLVMValueRef sum = LLVMAddFunction(module, "sum", sumType);
        LLVMSetFunctionCallConv(sum, LLVMCCallConv);

        LLVMValueRef lhs = LLVMGetParam(sum, 0);
        LLVMValueRef rhs = LLVMGetParam(sum, 1);
        LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(context, sum, "entry");

        LLVMPositionBuilderAtEnd(builder, entry);
        LLVMValueRef result = LLVMBuildAdd(builder, lhs, rhs, "result = lhs + rhs");
        LLVMBuildRet(builder, result);

        LLVMDumpModule(module);
        if (LLVMVerifyModule(module, LLVMPrintMessageAction, error) != 0) {
            System.out.println("Failed to validate module: " + error.getString());
            return;
        }

        // Stage 3: Dump the module to file
        if (LLVMWriteBitcodeToFile(module, "./sum.bc") != 0) {
            System.err.println("Failed to write bitcode to file");
            return;
        }

        // Stage 4: Create the relocatable object file
        BytePointer triple = LLVMGetDefaultTargetTriple();
        LLVMTargetRef target = new LLVMTargetRef();

        if (LLVMGetTargetFromTriple(triple, target, error) != 0) {
            System.out.println("Failed to get target from triple: " + error.getString());
            LLVMDisposeMessage(error);
            return;
        }

        String cpu = "generic";
        String cpuFeatures = "";
        int optimizationLevel = 0;
        LLVMTargetMachineRef tm = LLVMCreateTargetMachine(
            target, triple.getString(), cpu, cpuFeatures, optimizationLevel,
            LLVMRelocDefault, LLVMCodeModelDefault
        );

        BytePointer outputFile = new BytePointer("./sum.o");
        if (LLVMTargetMachineEmitToFile(tm, module, outputFile, LLVMObjectFile, error) != 0) {
            System.err.println("Failed to emit relocatable object file: " + error.getString());
            LLVMDisposeMessage(error);
            return;
        }

        // Stage 5: Dispose of allocated resources
        outputFile.deallocate();
        LLVMDisposeMessage(triple);
        LLVMDisposeBuilder(builder);
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
    }
}
