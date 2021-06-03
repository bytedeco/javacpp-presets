import java.util.Random;
import org.bytedeco.javacpp.*;
import org.bytedeco.libffi.*;
import org.bytedeco.llvm.LLVM.*;

import static org.bytedeco.libffi.global.ffi.*;
import static org.bytedeco.llvm.global.LLVM.*;
import static org.bytedeco.mkl.global.mkl_rt.*;

/**
 * Matrix multiply benchmark.
 *
 * To run this sample, execute this command:
 * mvn clean compile exec:java -Djavacpp.platform.host
 *
 * If you set usePollyParallel, you may have to modify the file name of LLVMLoadLibraryPermanently().
 *
 * Note: This code is equivalent to this:
 * clang -O3 -march=native -mllvm -polly -mllvm -polly-vectorizer=stripmine
 *
 * Note: Instead of JNA or libffi, to obtain maximum performance, FunctionPointer should be used as shown here:
 * https://github.com/bytedeco/javacpp/blob/master/src/test/java/org/bytedeco/javacpp/PointerTest.java
 *
 * @author Yu Kobayashi
 */
public class MatMulBenchmark {
    static final int M = 2000, N = 2000, K = 2000;
    static final boolean usePolly = true;
    static final boolean usePollyParallel = true;
    static final boolean printResult = false;

    static final BytePointer cpu = LLVMGetHostCPUName();
    static LLVMTypeRef llvmVoidType;
    static LLVMTypeRef llvmInt32Type;
    static LLVMTypeRef llvmFloatType;
    static LLVMTypeRef llvmFloatPointerType;

    public static void main(String[] args) {
        float[] a = createRandomArray(M, K);
        float[] b = createRandomArray(K, N);
        float[] c = new float[M * N];

        initialize();

        benchmarkMKL(a, b, c);
        benchmarkLLVM(a, b, c);
        benchmarkPureJava(a, b, c);
    }

    static void benchmarkMKL(float[] a, float[] b, float[] c) {
        Pointer jitter = new Pointer();
        mkl_cblas_jit_create_sgemm(jitter, CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, K, N, 0.0f, N);
        sgemm_jit_kernel_t sgemm = mkl_jit_get_sgemm_ptr(jitter);

        FloatPointer A = new FloatPointer(a);
        FloatPointer B = new FloatPointer(b);
        FloatPointer C = new FloatPointer(c);

        int iterations = 10;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            sgemm.call(jitter, A, B, C);
        }
        long end = System.nanoTime();
        C.get(c);
        System.out.printf("MKL: %fms. c[0] = %f\n", (end - start) / (iterations * 1000d * 1000d), c[0]);
        printArray(c);

        mkl_jit_destroy(jitter);
    }

    static void benchmarkPureJava(float[] a, float[] b, float[] c) {
        assert a.length == M * K;
        assert b.length == K * N;
        assert c.length == M * N;

        System.out.println("Now starting pure Java benchmark: This may take a minute.");
        long start = System.nanoTime();
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float s = 0;
                for (int k = 0; k < K; k++) {
                    s += a[m * K + k] * b[k * N + n];
                }
                c[m * N + n] = s;
            }
        }
        long end = System.nanoTime();
        System.out.printf("Pure Java: %fms. c[0] = %f\n", (end - start) / (1000d * 1000d), c[0]);
        printArray(c);
    }

    static void initialize() {
        if (usePolly) {
            if (usePollyParallel) {
                String platform = Loader.getPlatform();
                String omplib = platform.startsWith("linux") ? "libiomp5.so"
                              : platform.startsWith("macosx") ? "libiomp5.dylib"
                              : platform.startsWith("windows") ? "libiomp5md.dll"
                              : null;
                if (omplib != null) {
                    LLVMLoadLibraryPermanently(omplib);
                }
                setLLVMCommandLineOptions("",
                        "-mllvm", "-polly",
                        "-mllvm", "-polly-parallel",
                        "-mllvm", "-polly-vectorizer=stripmine");
            } else {
                setLLVMCommandLineOptions("",
                        "-mllvm", "-polly",
                        "-mllvm", "-polly-vectorizer=stripmine");
            }
        }

        LLVMLinkInMCJIT();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        LLVMInitializeNativeDisassembler();
        LLVMInitializeNativeTarget();

        llvmVoidType = LLVMVoidType();
        llvmInt32Type = LLVMInt32Type();
        llvmFloatType = LLVMFloatType();
        llvmFloatPointerType = LLVMPointerType(llvmFloatType, 0);
    }

    static void benchmarkLLVM(float[] a, float[] b, float[] c) {
        assert a.length == M * K;
        assert b.length == K * N;
        assert c.length == M * N;

        LLVMExecutionEngineRef engine = new LLVMExecutionEngineRef();
        try {
            LLVMModuleRef module = build();
            verify(module, false);
            optimize(module);
            jitCompile(engine, module);

            long fnAddr = LLVMGetFunctionAddress(engine, "matmul");
            // Using libffi directly reduces overhead, but we can also do the equivalent with JNA as follows:
            //  com.sun.jna.Function func = com.sun.jna.Function.getFunction(new com.sun.jna.Pointer(fnAddr));
            //  func.invoke(Void.class, new Object[]{a, b, c});
            ffi_cif cif = new ffi_cif();
            Pointer fn = new Pointer() {{ address = fnAddr; }};
            PointerPointer<ffi_type> args = new PointerPointer<>(ffi_type_pointer(), ffi_type_pointer(), ffi_type_pointer());
            PointerPointer<FloatPointer> pointers = new PointerPointer<>(a, b, c);
            PointerPointer<PointerPointer> values = new PointerPointer<>(pointers.getPointer(0), pointers.getPointer(1), pointers.getPointer(2));
            ffi_prep_cif(cif, FFI_DEFAULT_ABI(), 3, ffi_type_void(), args);
            long start = System.nanoTime();
            ffi_call(cif, fn, null, values);
            long end = System.nanoTime();
            pointers.get(FloatPointer.class, 2).get(c);
            System.out.printf("LLVM%s: %fms. c[0] = %f\n",
                    usePolly ? " with Polly" : " without Polly",
                    (end - start) / (1000d * 1000d),
                    c[0]);
            printArray(c);
        } finally {
            LLVMDisposeExecutionEngine(engine);
        }
    }

    static LLVMModuleRef build() {
        LLVMBuilderRef builder = LLVMCreateBuilder();

        LLVMModuleRef module = LLVMModuleCreateWithName("matmulModule");

        // Create function
        LLVMTypeRef funcType = LLVMFunctionType(
                llvmVoidType,
                new PointerPointer<>(llvmFloatPointerType, llvmFloatPointerType, llvmFloatPointerType),
                3,
                0);
        LLVMValueRef func = LLVMAddFunction(module, "matmul", funcType);
        LLVMSetFunctionCallConv(func, LLVMCCallConv);

        LLVMValueRef paramA = LLVMGetParam(func, 0);
        LLVMValueRef paramB = LLVMGetParam(func, 1);
        LLVMValueRef paramC = LLVMGetParam(func, 2);

        // entry basic block
        LLVMBasicBlockRef entryBB = LLVMAppendBasicBlock(func, "entry");
        LLVMPositionBuilderAtEnd(builder, entryBB);

        // loop M basic block
        LLVMBasicBlockRef loopMBB = LLVMAppendBasicBlock(func, "loopM");
        LLVMBuildBr(builder, loopMBB);
        LLVMPositionBuilderAtEnd(builder, loopMBB);

        // loop M index variable
        LLVMValueRef loopMIdx = LLVMBuildPhi(builder, llvmInt32Type, "m");
        LLVMAddIncoming(loopMIdx, toConstInt(0), entryBB, 1);

        // loop N basic block
        LLVMBasicBlockRef loopNBB = LLVMAppendBasicBlock(func, "loopN");
        LLVMBuildBr(builder, loopNBB);
        LLVMPositionBuilderAtEnd(builder, loopNBB);

        // loop N index variable
        LLVMValueRef loopNIdx = LLVMBuildPhi(builder, llvmInt32Type, "n");
        LLVMAddIncoming(loopNIdx, toConstInt(0), loopMBB, 1);

        // loop K basic block
        LLVMBasicBlockRef loopKBB = LLVMAppendBasicBlock(func, "loopK");
        LLVMBuildBr(builder, loopKBB);
        LLVMPositionBuilderAtEnd(builder, loopKBB);

        // loop K index variable
        LLVMValueRef loopKIdx = LLVMBuildPhi(builder, llvmInt32Type, "k");
        LLVMAddIncoming(loopKIdx, toConstInt(0), loopNBB, 1);

        // s = 0
        LLVMValueRef s = LLVMBuildPhi(builder, llvmFloatType, "s");
        LLVMAddIncoming(s, toConstFloat(0), loopNBB, 1);

        // s += a[m * K + k] * b[k * N + n]
        LLVMValueRef mMulK = LLVMBuildMul(builder, loopMIdx, toConstInt(K), "m * K");
        LLVMValueRef mMulKAddK = LLVMBuildAdd(builder, mMulK, loopKIdx, "m * K + k");
        LLVMValueRef aAryPtr = LLVMBuildInBoundsGEP(builder, paramA, mMulKAddK, 1, new BytePointer("&a[m * K + k]"));
        LLVMValueRef aAryValue = LLVMBuildLoad(builder, aAryPtr, "a[m * K + k]");
        LLVMValueRef kMulN = LLVMBuildMul(builder, loopKIdx, toConstInt(N), "k * N");
        LLVMValueRef kMulNAddN = LLVMBuildAdd(builder, kMulN, loopNIdx, "k * N + n");
        LLVMValueRef bAryPtr = LLVMBuildInBoundsGEP(builder, paramB, kMulNAddN, 1, new BytePointer("&b[k * N + n]"));
        LLVMValueRef bAryValue = LLVMBuildLoad(builder, bAryPtr, "b[k * N + n]");
        LLVMValueRef aMulB = LLVMBuildFMul(builder, aAryValue, bAryValue, "a[m * K + k] * b[k * N + n]");
        LLVMValueRef sAddAMulB = LLVMBuildFAdd(builder, s, aMulB, "s + a[m * K + k] * b[k * N + n]");

        // k++
        LLVMValueRef nextLoopKIdx = LLVMBuildAdd(builder, loopKIdx, toConstInt(1), "k + 1");

        // k == K
        LLVMValueRef kEndCond = LLVMBuildICmp(builder, LLVMIntEQ, nextLoopKIdx, toConstInt(K), "k == K");

        LLVMBasicBlockRef loopKEndBB = LLVMGetInsertBlock(builder);
        LLVMBasicBlockRef afterKBB = LLVMAppendBasicBlock(func, "afterK");
        LLVMBuildCondBr(builder, kEndCond, afterKBB, loopKBB);
        LLVMPositionBuilderAtEnd(builder, afterKBB);
        LLVMAddIncoming(loopKIdx, nextLoopKIdx, loopKEndBB, 1);
        LLVMAddIncoming(s, sAddAMulB, loopKEndBB, 1);

        // c[m * N + n] = s
        LLVMValueRef mMulN = LLVMBuildMul(builder, loopMIdx, toConstInt(N), "m * N");
        LLVMValueRef mMulNAddN = LLVMBuildAdd(builder, mMulN, loopNIdx, "m * N + n");
        LLVMValueRef cAryPtr = LLVMBuildInBoundsGEP(builder, paramC, mMulNAddN, 1, new BytePointer("&c[m * N + n]"));
        LLVMBuildStore(builder, sAddAMulB, cAryPtr);

        // n++
        LLVMValueRef nextLoopNIdx = LLVMBuildAdd(builder, loopNIdx, toConstInt(1), "n + 1");

        // n == N
        LLVMValueRef nEndCond = LLVMBuildICmp(builder, LLVMIntEQ, nextLoopNIdx, toConstInt(N), "n == N");

        LLVMBasicBlockRef loopNEndBB = LLVMGetInsertBlock(builder);
        LLVMBasicBlockRef afterNBB = LLVMAppendBasicBlock(func, "afterN");
        LLVMBuildCondBr(builder, nEndCond, afterNBB, loopNBB);
        LLVMPositionBuilderAtEnd(builder, afterNBB);
        LLVMAddIncoming(loopNIdx, nextLoopNIdx, loopNEndBB, 1);

        // m++
        LLVMValueRef nextLoopMIdx = LLVMBuildAdd(builder, loopMIdx, toConstInt(1), "m + 1");

        // m == M
        LLVMValueRef mEndCond = LLVMBuildICmp(builder, LLVMIntEQ, nextLoopMIdx, toConstInt(M), "m == M");

        LLVMBasicBlockRef loopMEndBB = LLVMGetInsertBlock(builder);
        LLVMBasicBlockRef afterMBB = LLVMAppendBasicBlock(func, "afterM");
        LLVMBuildCondBr(builder, mEndCond, afterMBB, loopMBB);
        LLVMPositionBuilderAtEnd(builder, afterMBB);
        LLVMAddIncoming(loopMIdx, nextLoopMIdx, loopMEndBB, 1);

        // return
        LLVMBuildRetVoid(builder);

        LLVMDisposeBuilder(builder);

        return module;
    }

    static void handleError(LLVMErrorRef e) {
        if (e != null && !e.isNull()) {
            BytePointer p = LLVMGetErrorMessage(e);
            String s = p.getString();
            LLVMDisposeErrorMessage(p);
            throw new RuntimeException(s);
        }
    }

    static void optimize(LLVMModuleRef module) {
        handleError(optimizeModule(module, cpu, 3, 0));
    }

    static void verify(LLVMModuleRef module, boolean dumpModule) {
        BytePointer error = new BytePointer((Pointer) null);
        try {
            if (dumpModule) {
                LLVMDumpModule(module);
            }
            if (LLVMVerifyModule(module, LLVMPrintMessageAction, error) != 0) {
                throw new RuntimeException(error.getString());
            }
        } finally {
            LLVMDisposeMessage(error);
        }
    }

    static void jitCompile(LLVMExecutionEngineRef engine, LLVMModuleRef module) {
        handleError(createOptimizedJITCompilerForModule(engine, module, cpu, 3));
    }

    static LLVMValueRef toConstInt(int v) {
        return LLVMConstInt(llvmInt32Type, v, 0);
    }

    static LLVMValueRef toConstFloat(float v) {
        return LLVMConstReal(llvmFloatType, v);
    }

    static void setLLVMCommandLineOptions(String... args) {
        LLVMParseCommandLineOptions(args.length, new PointerPointer<>(args), null);
    }

    static float[] createRandomArray(int m, int n) {
        Random rand = new Random();
        float[] ary = new float[m * n];
        for (int i = 0; i < ary.length; i++) {
            ary[i] = rand.nextFloat();
        }
        return ary;
    }

    static void printArray(float[] ary) {
        if (printResult) {
            for (float v : ary) {
                System.out.println(v);
            }
        }
    }
}
