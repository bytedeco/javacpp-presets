/*
 * Copyright (C) 2021 Mats Larsen
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

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.libffi.ffi_cif;
import org.bytedeco.llvm.LLVM.LLVMBasicBlockRef;
import org.bytedeco.llvm.LLVM.LLVMBuilderRef;
import org.bytedeco.llvm.LLVM.LLVMContextRef;
import org.bytedeco.llvm.LLVM.LLVMErrorRef;
import org.bytedeco.llvm.LLVM.LLVMModuleRef;
import org.bytedeco.llvm.LLVM.LLVMOrcJITDylibRef;
import org.bytedeco.llvm.LLVM.LLVMOrcLLJITBuilderRef;
import org.bytedeco.llvm.LLVM.LLVMOrcLLJITRef;
import org.bytedeco.llvm.LLVM.LLVMOrcThreadSafeContextRef;
import org.bytedeco.llvm.LLVM.LLVMOrcThreadSafeModuleRef;
import org.bytedeco.llvm.LLVM.LLVMTypeRef;
import org.bytedeco.llvm.LLVM.LLVMValueRef;

import static org.bytedeco.llvm.global.LLVM.*;
import static org.bytedeco.libffi.global.ffi.*;

/**
 * Sample code for using the OrcJIT v2 API to JIT compile and call arbitrary functions
 * <p>
 * This sample contains code for the following steps:
 * <p>
 * 1. Initializing required LLVM components
 * 2. Generating LLVM IR for a sum function
 * 3. Verify the module
 * 4. Load the module into OrcJIT and get the address of "sum"
 * 5. Call the sum function with libffi
 * 6. Dispose of the allocated resources
 */
public class OrcJit {

    public static void main(String[] args) {
        // Stage 1: Initialize LLVM components
        LLVMInitializeNativeTarget();
        LLVMInitializeNativeAsmPrinter();

        // Stage 2: Generate LLVM IR
        LLVMContextRef context = LLVMGetGlobalContext();
        LLVMOrcThreadSafeContextRef threadContext = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(context);
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

        // Print generated LLVM-IR to console (optional)
        LLVMDumpModule(module);

        // Stage 3: Verify the module (optional; recommended)
        BytePointer errorMessageVariable = new BytePointer();
        if (LLVMVerifyModule(module, LLVMPrintMessageAction, errorMessageVariable) != 0) {
            LLVMDisposeMessage(errorMessageVariable);
            return;
        }

        // Stage 4: Execute using OrcJIT
        LLVMOrcLLJITRef jit = new LLVMOrcLLJITRef();
        LLVMOrcLLJITBuilderRef jitBuilder = LLVMOrcCreateLLJITBuilder();
        LLVMErrorRef error;
        if ((error = LLVMOrcCreateLLJIT(jit, jitBuilder)) != null) {
            BytePointer errorMessage = LLVMGetErrorMessage(error);
            System.err.println("Failed to create LLJIT: " + errorMessage.getString());
            LLVMDisposeErrorMessage(errorMessage);
            return;
        }

        LLVMOrcJITDylibRef mainDylib = LLVMOrcLLJITGetMainJITDylib(jit);
        LLVMOrcThreadSafeModuleRef threadModule = LLVMOrcCreateNewThreadSafeModule(module, threadContext);
        if ((error = LLVMOrcLLJITAddLLVMIRModule(jit, mainDylib, threadModule)) != null) {
            BytePointer errorMessage = LLVMGetErrorMessage(error);
            System.err.println("Failed to add LLVM IR module: " + errorMessage.getString());
            LLVMDisposeErrorMessage(errorMessage);
            return;
        }

        final LongPointer res = new LongPointer(1);
        if ((error = LLVMOrcLLJITLookup(jit, res, "sum")) != null) {
            BytePointer errorMessage = LLVMGetErrorMessage(error);
            System.err.println("Failed to look up 'sum' symbol: " + errorMessage.getString());
            LLVMDisposeErrorMessage(errorMessage);
            return;
        }

        // Stage 5: Call the function with libffi
        ffi_cif callInterface = new ffi_cif();
        PointerPointer<Pointer> arguments = new PointerPointer<>(2)
            .put(0, ffi_type_sint())
            .put(1, ffi_type_sint());
        PointerPointer<Pointer> values = new PointerPointer<>(2)
            .put(0, new IntPointer(1).put(42))
            .put(1, new IntPointer(1).put(30));
        IntPointer resultVariable = new IntPointer(1);

        if (ffi_prep_cif(callInterface, FFI_DEFAULT_ABI(), 2, ffi_type_sint(), arguments) != FFI_OK) {
            System.err.println("Failed to prepare the libffi call interface");
            return;
        }
        Pointer function = new Pointer() {{
            address = res.get();
        }};
        ffi_call(callInterface, function, resultVariable, values);
        System.out.println("Evaluating sum(42, 30) through OrcJIT results in: " + resultVariable.get());

        // Stage 6: Dispose of the allocated resources
        LLVMOrcDisposeLLJIT(jit);
        LLVMShutdown();
    }
}
