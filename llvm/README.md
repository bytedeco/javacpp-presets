JavaCPP Presets for LLVM
========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * LLVM 3.8.0  http://llvm.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/llvm/apidocs/

&lowast; Bindings are currently available only for the C API of LLVM.

Sample Usage
------------
Here is a simple example of LLVM ported to Java from this C source file:

 * https://github.com/wickedchicken/llvm-c-example/blob/master/fac.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/Fac.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.llvm</groupId>
    <artifactId>fac</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>Fac</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>llvm</artifactId>
            <version>3.8.0-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/Fac.java` source file
```java
// General stuff
import org.bytedeco.javacpp.*;

// Headers required by LLVM
import static org.bytedeco.javacpp.LLVM.*;

public class Fac {
    public static void main (String[] args) {
        BytePointer error = new BytePointer((Pointer)null); // Used to retrieve messages from functions
        LLVMLinkInMCJIT();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeDisassembler();
        LLVMInitializeNativeTarget();
        LLVMModuleRef mod = LLVMModuleCreateWithName("fac_module");
        LLVMTypeRef[] fac_args = { LLVMInt32Type() };
        LLVMValueRef fac = LLVMAddFunction(mod, "fac", LLVMFunctionType(LLVMInt32Type(), fac_args[0], 1, 0));
        LLVMSetFunctionCallConv(fac, LLVMCCallConv);
        LLVMValueRef n = LLVMGetParam(fac, 0);

        LLVMBasicBlockRef entry = LLVMAppendBasicBlock(fac, "entry");
        LLVMBasicBlockRef iftrue = LLVMAppendBasicBlock(fac, "iftrue");
        LLVMBasicBlockRef iffalse = LLVMAppendBasicBlock(fac, "iffalse");
        LLVMBasicBlockRef end = LLVMAppendBasicBlock(fac, "end");
        LLVMBuilderRef builder = LLVMCreateBuilder();

        LLVMPositionBuilderAtEnd(builder, entry);
        LLVMValueRef If = LLVMBuildICmp(builder, LLVMIntEQ, n, LLVMConstInt(LLVMInt32Type(), 0, 0), "n == 0");
        LLVMBuildCondBr(builder, If, iftrue, iffalse);

        LLVMPositionBuilderAtEnd(builder, iftrue);
        LLVMValueRef res_iftrue = LLVMConstInt(LLVMInt32Type(), 1, 0);
        LLVMBuildBr(builder, end);

        LLVMPositionBuilderAtEnd(builder, iffalse);
        LLVMValueRef n_minus = LLVMBuildSub(builder, n, LLVMConstInt(LLVMInt32Type(), 1, 0), "n - 1");
        LLVMValueRef[] call_fac_args = { n_minus };
        LLVMValueRef call_fac = LLVMBuildCall(builder, fac, new PointerPointer(call_fac_args), 1, "fac(n - 1)");
        LLVMValueRef res_iffalse = LLVMBuildMul(builder, n, call_fac, "n * fac(n - 1)");
        LLVMBuildBr(builder, end);

        LLVMPositionBuilderAtEnd(builder, end);
        LLVMValueRef res = LLVMBuildPhi(builder, LLVMInt32Type(), "result");
        LLVMValueRef[] phi_vals = { res_iftrue, res_iffalse };
        LLVMBasicBlockRef[] phi_blocks = { iftrue, iffalse };
        LLVMAddIncoming(res, new PointerPointer(phi_vals), new PointerPointer(phi_blocks), 2);
        LLVMBuildRet(builder, res);

        LLVMVerifyModule(mod, LLVMAbortProcessAction, error);
        LLVMDisposeMessage(error); // Handler == LLVMAbortProcessAction -> No need to check errors


        LLVMExecutionEngineRef engine = new LLVMExecutionEngineRef();
        LLVMModuleProviderRef provider = LLVMCreateModuleProviderForExistingModule(mod);
        error = new BytePointer((Pointer)null);
        if(LLVMCreateJITCompiler(engine, provider, 2, error) != 0) {
            System.err.println(error.getString());
            LLVMDisposeMessage(error);
            System.exit(-1);
        }

        LLVMPassManagerRef pass = LLVMCreatePassManager();
        LLVMAddTargetData(LLVMGetExecutionEngineTargetData(engine), pass);
        LLVMAddConstantPropagationPass(pass);
        LLVMAddInstructionCombiningPass(pass);
        LLVMAddPromoteMemoryToRegisterPass(pass);
        // LLVMAddDemoteMemoryToRegisterPass(pass); // Demotes every possible value to memory
        LLVMAddGVNPass(pass);
        LLVMAddCFGSimplificationPass(pass);
        LLVMRunPassManager(pass, mod);
        LLVMDumpModule(mod);

        LLVMGenericValueRef exec_args = LLVMCreateGenericValueOfInt(LLVMInt32Type(), 10, 0);
        LLVMGenericValueRef exec_res = LLVMRunFunction(engine, fac, 1, exec_args);
        System.err.println();
        System.err.println("; Running fac(10) with JIT...");
        System.err.println("; Result: " + LLVMGenericValueToInt(exec_res, 0));

        LLVMDisposePassManager(pass);
        LLVMDisposeBuilder(builder);
        LLVMDisposeExecutionEngine(engine);
    }
}
```
