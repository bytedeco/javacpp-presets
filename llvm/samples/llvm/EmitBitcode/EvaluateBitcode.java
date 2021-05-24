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
import org.bytedeco.llvm.LLVM.LLVMContextRef;
import org.bytedeco.llvm.LLVM.LLVMExecutionEngineRef;
import org.bytedeco.llvm.LLVM.LLVMGenericValueRef;
import org.bytedeco.llvm.LLVM.LLVMMemoryBufferRef;
import org.bytedeco.llvm.LLVM.LLVMModuleRef;
import org.bytedeco.llvm.LLVM.LLVMTypeRef;
import org.bytedeco.llvm.LLVM.LLVMValueRef;

import static org.bytedeco.llvm.global.LLVM.*;

/**
 * Sample code for importing a LLVM bitcode file and running a function
 * inside of the imported module
 *
 * This sample depends on EmitBitcode to produce the bitcode file. Make sure
 * you've ran the EmitBitcode sample and have the 'sum.bc' bitcode file.
 *
 * This sample contains code for the following steps:
 *
 * 1. Initializing required LLVM components
 * 2. Load and parse the bitcode
 * 3. Run the 'sum' function inside the module
 * 4. Dispose of the allocated resources
 */
public class EvaluateBitcode {
    public static final BytePointer error = new BytePointer();

    public static void main(String[] args) {
        // Stage 1: Initialize LLVM components
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeTarget();

        // Stage 2: Load and parse bitcode
        LLVMContextRef context = LLVMContextCreate();
        LLVMTypeRef i32Type = LLVMInt32TypeInContext(context);
        LLVMModuleRef module = new LLVMModuleRef();
        LLVMMemoryBufferRef membuf = new LLVMMemoryBufferRef();

        BytePointer inputFile = new BytePointer("./sum.bc");
        if (LLVMCreateMemoryBufferWithContentsOfFile(inputFile, membuf, error) != 0) {
            System.err.println("Failed to read file into memory buffer: " + error.getString());
            LLVMDisposeMessage(error);
            return;
        }

        if (LLVMParseBitcodeInContext2(context, membuf, module) != 0) {
            System.err.println("Failed to parser module from bitcode");
            return;
        }

        LLVMExecutionEngineRef engine = new LLVMExecutionEngineRef();
        if (LLVMCreateInterpreterForModule(engine, module, error) != 0) {
            System.err.println("Failed to create LLVM interpreter: " + error.getString());
            LLVMDisposeMessage(error);
            return;
        }

        LLVMValueRef sum = LLVMGetNamedFunction(module, "sum");
        PointerPointer<Pointer> arguments = new PointerPointer<>(2)
            .put(0, LLVMCreateGenericValueOfInt(i32Type, 42, /* signExtend */ 0))
            .put(1, LLVMCreateGenericValueOfInt(i32Type, 30, /* signExtend */ 0));
        LLVMGenericValueRef result = LLVMRunFunction(engine, sum, 2, arguments);

        System.out.println();
        System.out.print("The result of add(42, 32) imported from bitcode and executed with LLVM interpreter is: ");
        System.out.println(LLVMGenericValueToInt(result, /* signExtend */ 0));

        // Stage 4: Dispose of the allocated resources
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
    }
}
