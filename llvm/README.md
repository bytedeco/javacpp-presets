JavaCPP Presets for LLVM
========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/llvm/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/llvm) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/llvm.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![llvm](https://github.com/bytedeco/javacpp-presets/workflows/llvm/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Allvm)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * LLVM 11.0.1  http://llvm.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/llvm/apidocs/

&lowast; Bindings are currently available only for the C APIs of LLVM and Clang.  
&lowast; In the case of Clang, we might need to disable crash recovery with the `LIBCLANG_DISABLE_CRASH_RECOVERY=1` environment variable to prevent clashes with the JVM's own signal handlers.


Sample Usage
------------
Here is a simple example of LLVM ported to Java from this C source file:

 * https://github.com/wickedchicken/llvm-c-example/blob/master/fac.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Fac.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.llvm</groupId>
    <artifactId>fac</artifactId>
    <version>1.5.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>Fac</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>llvm-platform</artifactId>
            <version>11.0.1-1.5.5-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Fac.java` source file
```java
// General stuff
import org.bytedeco.javacpp.*;

// Headers required by LLVM
import org.bytedeco.llvm.LLVM.*;
import static org.bytedeco.llvm.global.LLVM.*;

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
        if(LLVMCreateJITCompilerForModule(engine, mod, 2, error) != 0) {
            System.err.println(error.getString());
            LLVMDisposeMessage(error);
            System.exit(-1);
        }

        LLVMPassManagerRef pass = LLVMCreatePassManager();
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
