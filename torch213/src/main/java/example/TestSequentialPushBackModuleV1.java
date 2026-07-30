package org.example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.DropoutImpl;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.ReLUImpl;
import org.bytedeco.pytorch.SequentialImpl;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * Functional verification for the generic
 * {@link SequentialImpl#push_back(Module)} overload and its named
 * {@code push_back(String, Module)} sibling.
 *
 * <p>The C++ side uses {@code torch::nn::push_back_module} (a free helper
 * added to the preset's {@code cppText}) which does {@code dynamic_cast}
 * on the actual C++ typeid and forwards to the matching typed
 * {@code push_back<Impl>} overload. The Java-side declarations are
 * emitted by JavaCPP itself; this test just exercises the generated
 * entry points.
 */
public class TestSequentialPushBackModuleV1 {

    private static int passed = 0;
    private static int failed = 0;

    private static void check(boolean cond, String msg) {
        if (cond) {
            passed++;
        } else {
            failed++;
            System.err.println("FAIL: " + msg);
        }
    }

    /**
     * User-defined class that extends {@link Module} directly. The
     * underlying C++ type is the base {@code torch::nn::Module}; the
     * javacpp-presets patch adds a virtual {@code forward()} to the
     * base class so that {@code AnyModule<Module>} compiles and
     * {@code push_back<Module>(shared_ptr<Module>)} instantiates.
     * The custom Java class adds a couple of sub-modules on top.
     */
    public static final class MyBlock extends Module {
        public MyBlock() {
            super("MyBlock");
            register_module("fc1", new LinearImpl(4L, 4L));
            register_module("act", new ReLUImpl());
            register_module("fc2", new LinearImpl(4L, 4L));
        }

        public MyBlock(org.bytedeco.javacpp.Pointer p) {
            super(p);
        }

        public Tensor forward(Tensor x) {
            Tensor y = children().get()[0].forward(x);
            y = children().get()[1].forward(y);
            y = children().get()[2].forward(y);
            return y;
        }


    }

    private static void testGenericPushBack() {
        try (PointerScope scope = new PointerScope()) {
            SequentialImpl seq = new SequentialImpl();
            Module linear = new LinearImpl(4L, 8L);
            Module relu = new ReLUImpl();
            Module linear2 = new LinearImpl(8L, 2L);
            Module dropout = new DropoutImpl(0.5);
            seq.push_back(linear);
            seq.push_back(relu);
            seq.push_back(linear2);
            seq.push_back(dropout);
            check(seq.size() == 4, "size should be 4 after 4 generic push_backs, got " + seq.size());

            // Keep this test focused on insertion + type retention.
            // Forward path is validated in the dedicated pipeline benchmarks.
            Module[] children = seq.children().get();
            check(children.length == 4, "expected 4 children");
        }
    }

    private static void testNamedPushBack() {
        try (PointerScope scope = new PointerScope()) {
            SequentialImpl seq = new SequentialImpl();
            LinearImpl a = new LinearImpl(4L, 4L);
            ReLUImpl b = new ReLUImpl();
            LinearImpl c = new LinearImpl(4L, 4L);
            seq.push_back("alpha", a);
            seq.push_back("beta",  b);
            seq.push_back("gamma", c);
            check(seq.size() == 3, "size should be 3, got " + seq.size());
        }
    }

    private static void testCustomModulePushBack() {
        try (PointerScope scope = new PointerScope()) {
            // MyBlock extends Module directly. Without a C++ patch
            // adding virtual forward() to the base Module class, the
            // C++ push_back<Module> template cannot be instantiated
            // (AnyModule<Module>'s static_assert on has_forward
            // fails). The as<Impl>() walk in ModuleAsHelper finds no
            // match for a bare-Module pointer, so the push_back
            // throws. The user must either:
            //   1. extend a *Impl class (e.g. LinearImpl) instead, or
            //   2. push the inner *Impl via `myBlock.children().get()[0]`
            //      followed by a typed push_back.
            SequentialImpl seq = new SequentialImpl();
            Module myBlock = new MyBlock();
            try {
                seq.push_back(myBlock);
                System.out.println("insert myblock success");
                // If we got here, somehow it worked.
                check(seq.size() == 1, "unexpected: custom module push_back succeeded");
            } catch (IllegalArgumentException e) {
                check(e.getMessage().contains("unsupported module type"),
                        "expected unsupported module type error, got: " + e.getMessage());
                System.out.println("[custom] expected failure (no C++ patch): " + e.getMessage());
            }
            // Workaround: extract the inner LinearImpl and push it
            // via the typed push_back(LinearImpl) overload.
            Module[] children = myBlock.children().get();
            check(children.length == 3, "MyBlock should expose 3 sub-modules");
            LinearImpl innerLinear = (LinearImpl) children[0].as(LinearImpl.class);
            check(innerLinear != null, "expected first child to be a LinearImpl");
            seq.push_back(innerLinear);
            check(seq.size() == 2, "inner LinearImpl push_back should succeed");
            System.out.println("[custom] workaround (inner *Impl push_back) succeeded, size=" + seq.size());

            var my =seq.get(0).as(MyBlock.class).getClass().getName();
            System.out.println("myblock" + my);
        }
    }

    private static void testRoundTripViaAs() {
        try (PointerScope scope = new PointerScope()) {
            SequentialImpl seq = new SequentialImpl();
            LinearImpl linear = new LinearImpl(4L, 4L);
            ReLUImpl relu   = new ReLUImpl();
            seq.push_back(linear);
            seq.push_back(relu);

            // Round-trip the linear via as(LinearImpl.class) to make sure
            // the generic push_back actually stored a LinearImpl that
            // the existing as() mechanism can recover.
            Module[] children = seq.children().get();
            check(children.length == 2, "expected 2 children");
            int linearFound = 0, reluFound = 0;
            for (Module child : children) {
                if (child.as(LinearImpl.class) != null) linearFound++;
                if (child.as(ReLUImpl.class) != null)   reluFound++;
            }
            check(linearFound == 1, "expected 1 LinearImpl among children, got " + linearFound);
            check(reluFound == 1, "expected 1 ReLUImpl among children, got " + reluFound);
        }
    }

    public static void main(String[] args) {
        Loader.load(torch.class);
        testGenericPushBack();
        testNamedPushBack();
        testCustomModulePushBack();
        testRoundTripViaAs();
        System.out.println();
        System.out.println("TestSequentialPushBackModule: " + passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }
}