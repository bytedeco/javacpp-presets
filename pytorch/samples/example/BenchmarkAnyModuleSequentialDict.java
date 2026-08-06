package example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.AnyModule;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.modules.container.StringAnyModuleDict;

public class BenchmarkAnyModuleSequentialDict {
    private static final int LOOPS = 2000;

    private static void check(boolean ok, String msg) {
        if (!ok) throw new AssertionError(msg);
    }

    private static void verifyAnyModuleConversion() {
        try (PointerScope scope = new PointerScope()) {
            Module linear = new LinearImpl(8L, 8L);
            AnyModule a1 = new AnyModule(linear);
            AnyModule a2 = AnyModule.from_module(linear);
            Module r1 = a1.toModule();
            Module r2 = a2.toModule();
            check(r1 != null && !r1.isNull(), "AnyModule(Module) toModule failed");
            check(r2 != null && !r2.isNull(), "AnyModule.from_module toModule failed");
            check(r1.as(LinearImpl.class) != null, "AnyModule(Module) type mismatch");
            check(r2.as(LinearImpl.class) != null, "AnyModule.from_module type mismatch");
        }
    }

    private static SequentialImpl buildFromAnyDict() {
        StringAnyModuleDict dict = new StringAnyModuleDict();
        dict.insert("fc1", AnyModule.from_module(new LinearImpl(8L, 16L)));
        dict.insert("act1", AnyModule.from_module(new ReLUImpl()));
        dict.insert("fc2", AnyModule.from_module(new LinearImpl(16L, 4L)));
        return new SequentialImpl(dict);
    }

    private static void verifySequentialCtorFromAnyDict() {
        try (PointerScope scope = new PointerScope()) {
            SequentialImpl seq = buildFromAnyDict();
            check(seq.size() == 3, "SequentialImpl(StringAnyModuleDict) size mismatch");
            Module[] children = seq.children().get();
            check(children.length == 3, "children length mismatch");
            check(children[0].as(LinearImpl.class) != null, "child0 should be LinearImpl");
            check(children[1].as(ReLUImpl.class) != null, "child1 should be ReLUImpl");
            check(children[2].as(LinearImpl.class) != null, "child2 should be LinearImpl");
        }
    }

    private static void runBenchmarks() {
        long startCtor = System.nanoTime();
        for (int i = 0; i < LOOPS; i++) {
            try (PointerScope scope = new PointerScope()) {
                Module m = new LinearImpl(8L, 8L);
                AnyModule.from_module(m).toModule();
            }
        }
        long ctorNs = System.nanoTime() - startCtor;

        long checksum = 0;
        long startSeq = System.nanoTime();
        for (int i = 0; i < LOOPS; i++) {
            try (PointerScope scope = new PointerScope()) {
                SequentialImpl seq = buildFromAnyDict();
                checksum += seq.size();
            }
        }
        long seqNs = System.nanoTime() - startSeq;

        check(checksum == (long) LOOPS * 3L, "benchmark checksum mismatch");
        System.out.println("BenchmarkAnyModuleSequentialDict");
        System.out.println("loops=" + LOOPS);
        System.out.println("AnyModule.from_module + toModule: " + (ctorNs / 1_000_000.0) + " ms");
        System.out.println("SequentialImpl(StringAnyModuleDict) build+introspection: " + (seqNs / 1_000_000.0) + " ms");
        System.out.println("BENCHMARK OK");
    }

    public static void main(String[] args) {
        Loader.load(torch.class);
        torch.manual_seed(123);
        verifyAnyModuleConversion();
        verifySequentialCtorFromAnyDict();
        runBenchmarks();
    }
}
