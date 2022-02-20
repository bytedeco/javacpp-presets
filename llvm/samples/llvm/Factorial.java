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

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.llvm.LLVM.LLVMBasicBlockRef;
import org.bytedeco.llvm.LLVM.LLVMBuilderRef;
import org.bytedeco.llvm.LLVM.LLVMContextRef;
import org.bytedeco.llvm.LLVM.LLVMExecutionEngineRef;
import org.bytedeco.llvm.LLVM.LLVMGenericValueRef;
import org.bytedeco.llvm.LLVM.LLVMMCJITCompilerOptions;
import org.bytedeco.llvm.LLVM.LLVMModuleRef;
import org.bytedeco.llvm.LLVM.LLVMPassManagerRef;
import org.bytedeco.llvm.LLVM.LLVMTypeRef;
import org.bytedeco.llvm.LLVM.LLVMValueRef;

import static org.bytedeco.llvm.global.LLVM.*;

/**
 * Sample code for generating LLVM IR for a factorial function
 * <p>
 * The factorial function has the following LLVM IR signature:
 * <p>
 * declare i32 @factorial(i32 %n)
 * <p>
 * This sample contains code for the following stages:
 * <p>
 * 1. Initializing required LLVM components
 * 2. Generating LLVM IR for a factorial function
 * 3. Verifies that our generated code is well formed
 * 4. Creating a pass pipeline with the legacy pass manager
 * 5. Executing the code with the LLVM MCJIT Compiler
 * 6. Dispose of the allocated resources
 * <p>
 * TODO(supergrecko): Replace with new Pass Manager for LLVM 13
 */
public class Factorial {
    // a 'char *' used to retrieve error messages from LLVM
    private static final BytePointer error = new BytePointer();

    public static void main(String[] args) {
        // Stage 1: Initialize LLVM components
        LLVMInitializeCore(LLVMGetGlobalPassRegistry());
        LLVMLinkInMCJIT();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeTarget();

        // Stage 2: Build the factorial function.
        LLVMContextRef context = LLVMContextCreate();
        LLVMModuleRef module = LLVMModuleCreateWithNameInContext("factorial", context);
        LLVMBuilderRef builder = LLVMCreateBuilderInContext(context);
        LLVMTypeRef i32Type = LLVMInt32TypeInContext(context);
        LLVMTypeRef factorialType = LLVMFunctionType(i32Type, i32Type, /* argumentCount */ 1, /* isVariadic */ 0);

        LLVMValueRef factorial = LLVMAddFunction(module, "factorial", factorialType);
        LLVMSetFunctionCallConv(factorial, LLVMCCallConv);

        LLVMValueRef n = LLVMGetParam(factorial, /* parameterIndex */0);
        LLVMValueRef zero = LLVMConstInt(i32Type, 0, /* signExtend */ 0);
        LLVMValueRef one = LLVMConstInt(i32Type, 1, /* signExtend */ 0);
        LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(context, factorial, "entry");
        LLVMBasicBlockRef ifFalse = LLVMAppendBasicBlockInContext(context, factorial, "if_false");
        LLVMBasicBlockRef exit = LLVMAppendBasicBlockInContext(context, factorial, "exit");

        LLVMPositionBuilderAtEnd(builder, entry);
        LLVMValueRef condition = LLVMBuildICmp(builder, LLVMIntEQ, n, zero, "condition = n == 0");
        LLVMBuildCondBr(builder, condition, exit, ifFalse);

        LLVMPositionBuilderAtEnd(builder, ifFalse);
        LLVMValueRef nMinusOne = LLVMBuildSub(builder, n, one, "nMinusOne = n - 1");
        PointerPointer<Pointer> arguments = new PointerPointer<>(1)
            .put(0, nMinusOne);
        LLVMValueRef factorialResult = LLVMBuildCall(builder, factorial, arguments, 1, "factorialResult = factorial(nMinusOne)");
        LLVMValueRef resultIfFalse = LLVMBuildMul(builder, n, factorialResult, "resultIfFalse = n * factorialResult");
        LLVMBuildBr(builder, exit);

        LLVMPositionBuilderAtEnd(builder, exit);
        LLVMValueRef phi = LLVMBuildPhi(builder, i32Type, "result");
        PointerPointer<Pointer> phiValues = new PointerPointer<>(2)
            .put(0, one)
            .put(1, resultIfFalse);
        PointerPointer<Pointer> phiBlocks = new PointerPointer<>(2)
            .put(0, entry)
            .put(1, ifFalse);
        LLVMAddIncoming(phi, phiValues, phiBlocks, /* pairCount */ 2);
        LLVMBuildRet(builder, phi);

        // Stage 3: Verify the module using LLVMVerifier
        if (LLVMVerifyModule(module, LLVMPrintMessageAction, error) != 0) {
            LLVMDisposeMessage(error);
            return;
        }

        // Stage 4: Create a pass pipeline using the legacy pass manager
        LLVMPassManagerRef pm = LLVMCreatePassManager();
        LLVMAddAggressiveInstCombinerPass(pm);
        LLVMAddNewGVNPass(pm);
        LLVMAddCFGSimplificationPass(pm);
        LLVMRunPassManager(pm, module);
        LLVMDumpModule(module);

        // Stage 5: Execute the code using MCJIT
        LLVMExecutionEngineRef engine = new LLVMExecutionEngineRef();
        LLVMMCJITCompilerOptions options = new LLVMMCJITCompilerOptions();
        if (LLVMCreateMCJITCompilerForModule(engine, module, options, 3, error) != 0) {
            System.err.println("Failed to create JIT compiler: " + error.getString());
            LLVMDisposeMessage(error);
            return;
        }

        LLVMGenericValueRef argument = LLVMCreateGenericValueOfInt(i32Type, 10, /* signExtend */ 0);
        LLVMGenericValueRef result = LLVMRunFunction(engine, factorial, /* argumentCount */ 1, argument);
        System.out.println();
        System.out.println("; Running factorial(10) with MCJIT...");
        System.out.println("; Result: " + LLVMGenericValueToInt(result, /* signExtend */ 0));

        // Stage 6: Dispose of the allocated resources
        LLVMDisposeExecutionEngine(engine);
        LLVMDisposePassManager(pm);
        LLVMDisposeBuilder(builder);
        LLVMContextDispose(context);
    }
}
