import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.LLVM.*;

import static org.bytedeco.llvm.global.LLVM.*;

/**
 * This is an example which loads a LLVM module from a bitcode (.bc) file and
 * runs a function inside it using the LLVM interpreter.
 *
 * The sample should be used in conjunction with LLVMEmitObjectAndBitcodeFiles.java
 * which is capable of producing the bitcode file this example uses.
 *
 * It loads a function named "add" from the bitcode file which has a signature of
 * `i32 add(i32, i32)` which was created in the aforementioned Java file. It is then
 * executed via the LLVM interpreter and the result is printed to stdout.
 */
public class LLVMRunFunctionFromBitcode {
    public static void main(String[] unused) {
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeDisassembler();
        LLVMInitializeNativeTarget();

        LLVMModuleRef module = new LLVMModuleRef();
        LLVMMemoryBufferRef buffer = new LLVMMemoryBufferRef();
        BytePointer error = new BytePointer((Pointer) null);

        // Open file into memory buffer
        if (LLVMCreateMemoryBufferWithContentsOfFile(new BytePointer("./add.bc"), buffer, error) != 0) {
            String msg = error.getString();
            LLVMDisposeMessage(error);
            throw new RuntimeException(msg);
        }

        // Attempt to parse bitcode out of memory buffer
        if (LLVMParseBitcode2(buffer, module) != 0) {
            throw new RuntimeException("Failed to parse bitcode in module");
        }

        // Let's run the add function via the LLVM interpreter
        LLVMExecutionEngineRef engine = new LLVMExecutionEngineRef();
        if (LLVMCreateInterpreterForModule(engine, module, error) != 0) {
            String msg = error.getString();
            LLVMDisposeMessage(error);
            throw new RuntimeException(msg);
        }

        LLVMValueRef add = LLVMGetNamedFunction(module, "add");
        LLVMTypeRef i32 = LLVMInt32Type();
        LLVMGenericValueRef[] args = new LLVMGenericValueRef[]{
            LLVMCreateGenericValueOfInt(i32, 1000, 0),
            LLVMCreateGenericValueOfInt(i32, 1, 0)
        };
        PointerPointer<LLVMGenericValueRef> args_ptr = new PointerPointer<>(args);
        LLVMGenericValueRef result = LLVMRunFunction(engine, add, 2, args_ptr);
        args_ptr.deallocate();

        System.out.println("The result of add(1000, 1), loaded through membuffers and executed through LLVM " +
            "interpreter is: " + LLVMGenericValueToInt(result, 0));

        // Cleanup, the module is moved into the JIT compiler and released
        // when we release the engine
        LLVMDisposeMessage(error);
        LLVMDisposeExecutionEngine(engine);
        LLVMDisposeMemoryBuffer(buffer);
    }
}