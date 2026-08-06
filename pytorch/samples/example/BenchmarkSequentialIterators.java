package example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.AnyModule;
import org.bytedeco.pytorch.nn.modules.container.AnyModuleVector;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.modules.container.SharedModuleVector;

public class BenchmarkSequentialIterators {
    private static final int LOOPS = 20_000;

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static SequentialImpl buildSequential() {
        SequentialImpl seq = new SequentialImpl();
        seq.push_back("fc1", new LinearImpl(16L, 16L));
        seq.push_back("act1", new ReLUImpl());
        seq.push_back("fc2", new LinearImpl(16L, 16L));
        seq.push_back("act2", new ReLUImpl());
        seq.push_back("fc3", new LinearImpl(16L, 16L));
        seq.push_back("act3", new ReLUImpl());
        return seq;
    }

    private static int iterateSharedOnce(SequentialImpl seq) {
        int count = 0;
        SharedModuleVector.Iterator it = seq.begin();
        SharedModuleVector.Iterator end = seq.end();
        for (; !it.equals(end); it.increment()) {
            Module m = it.get();
            if(!m.asLinear().isNull()){
                System.out.println("begin");
                System.out.println(m.asLinear().getClass().getName());
                System.out.println("end");
            }
//            System.out.println(m.getClass().getName());
            check(m != null && !m.isNull(), "shared iterator returned null module");
            count++;
        }
        return count;
    }

    private static int iterateAnyOnce(SequentialImpl seq) {
        int count = 0;
        AnyModuleVector.Iterator it = seq.begin_any();
        AnyModuleVector.Iterator end = seq.end_any();
        for (; !it.equals(end); it.increment()) {
            AnyModule any = it.get();

            Module m = any.toModule();

//            System.out.println(m.getClass().getName());
            check(m != null && !m.isNull(), "any iterator returned null module");
            count++;
        }
        return count;
    }

    private static long benchSharedIterator(SequentialImpl seq) {
        long total = 0;
        long start = System.nanoTime();
        for (int i = 0; i < LOOPS; i++) {
            total += iterateSharedOnce(seq);
        }
        long elapsedNs = System.nanoTime() - start;
        check(total == (long) LOOPS * seq.size(), "shared iterator total mismatch");
        return elapsedNs;
    }

    private static long benchAnyIterator(SequentialImpl seq) {
        long total = 0;
        long start = System.nanoTime();
        for (int i = 0; i < LOOPS; i++) {
            total += iterateAnyOnce(seq);
        }
        long elapsedNs = System.nanoTime() - start;
        check(total == (long) LOOPS * seq.size(), "any iterator total mismatch");
        return elapsedNs;
    }

    private static void verifyIteratorElements(SequentialImpl seq) {
        Module firstShared = seq.begin().get();
        check(firstShared.as(LinearImpl.class) != null, "shared iterator first element should be LinearImpl");

        AnyModule firstAny = seq.begin_any().get();
        Module fromAny = firstAny.toModule();
        check(fromAny != null && !fromAny.isNull(), "any iterator toModule failed");
        check(fromAny.as(LinearImpl.class) != null, "any iterator first element should be LinearImpl");
    }

    public static void main(String[] args) {
        Loader.load(torch.class);
        torch.manual_seed(7);
        try (PointerScope scope = new PointerScope()) {
            SequentialImpl seq = buildSequential();
            check(seq.size() == 6, "sequential size should be 6");
            verifyIteratorElements(seq);

            long sharedNs = benchSharedIterator(seq);
            long anyNs = benchAnyIterator(seq);

            System.out.println("BenchmarkSequentialIterators "+sharedNs+" "+ anyNs);
            System.out.println("size=" + seq.size() + ", loops=" + LOOPS);
            System.out.println("shared(begin/end): " + (sharedNs / 1_000_000.0) + " ms");
            System.out.println("any(begin_any/end_any): " + (anyNs / 1_000_000.0) + " ms");
            System.out.println("ITERATORS OK");
        }
    }
}
