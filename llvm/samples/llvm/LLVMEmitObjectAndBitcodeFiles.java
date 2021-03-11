import org.bytedeco.javacpp.*;
import org.bytedeco.llvm.LLVM.*;

import static org.bytedeco.llvm.global.LLVM.*;

/**
 * This is an example which shows how you can dump both object and bitcode file
 * from a LLVM module in JavaCPP.
 *
 * It produces two files, add.o and add.bc. The bitcode file can be used in the
 * LLVMRunFunctionFromBitcode.java sample. See that file for details.
 */
public class LLVMEmitObjectAndBitcodeFiles {
    public static void main(String[] unused) {
        // This example uses the target you're running on, enable
        // more targets as necessary
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeDisassembler();
        LLVMInitializeNativeTarget();

        LLVMContextRef context = LLVMContextCreate();
        LLVMModuleRef module = LLVMModuleCreateWithNameInContext("test", context);
        LLVMBuilderRef builder = LLVMCreateBuilderInContext(context);
        BytePointer error = new BytePointer((Pointer) null);

        // Construct the "i32 add(i32, i32)" function
        LLVMTypeRef i32 = LLVMInt32TypeInContext(context);
        LLVMTypeRef[] args = {i32, i32};
        PointerPointer<LLVMTypeRef> args_ptr = new PointerPointer<>(args);
        LLVMTypeRef add_type = LLVMFunctionType(i32, args_ptr, 2, 0);
        LLVMValueRef add = LLVMAddFunction(module, "add", add_type);
        LLVMSetFunctionCallConv(add, LLVMCCallConv);
        args_ptr.deallocate();

        // Build the IR for the add function
        LLVMBasicBlockRef entrypoint = LLVMAppendBasicBlock(add, "Entry");
        LLVMPositionBuilderAtEnd(builder, entrypoint);

        LLVMValueRef a = LLVMGetParam(add, 0);
        LLVMValueRef b = LLVMGetParam(add, 1);
        LLVMValueRef sum = LLVMBuildAdd(builder, a, b, "sum");
        LLVMBuildRet(builder, sum);

        LLVMDumpModule(module);

        // Emit the module to an object file
        BytePointer triple_ptr = LLVMGetDefaultTargetTriple();
        String triple = triple_ptr.getString();
        triple_ptr.deallocate();
        LLVMSetTarget(module, triple);

        LLVMVerifyModule(module, LLVMAbortProcessAction, error);
        // Don't handle any error because LLVMAbortProcessAction was used

        // Note: Replace with your own architecture if you're not running on x86 64bit
        LLVMTargetRef target = LLVMGetTargetFromName("x86-64");
        String cpu = "generic";
        String cpu_features = "";
        int optimization_level = 0;

        LLVMTargetMachineRef machine = LLVMCreateTargetMachine(target, triple, cpu, cpu_features, optimization_level,
            LLVMRelocDefault, LLVMCodeModelDefault);
        BytePointer output = new BytePointer("add.o");

        if (LLVMTargetMachineEmitToFile(machine, module, output, LLVMObjectFile, error) != 0) {
            String msg = error.getString();
            LLVMDisposeMessage(error);
            throw new RuntimeException(msg);
        }
        error.deallocate();

        // Emit to bitcode
        if (LLVMWriteBitcodeToFile(module, "./add.bc") != 0) {
            throw new RuntimeException("failed to write bitcode to file");
        }

        // Clean up resources
        LLVMDisposeMessage(error);
        LLVMDisposeBuilder(builder);
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
    }
}
