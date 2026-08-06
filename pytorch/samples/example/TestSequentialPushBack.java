package example;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

public class TestSequentialPushBack {
    private static final class InputStem extends Module {
        private final LinearImpl fc;

        InputStem(long inFeatures, long outFeatures) {
//            super("InputStem");
            fc = register_module("fc", new LinearImpl(inFeatures, outFeatures));
        }

        @Override
        public Tensor forward(Tensor x) {
            return fc.forward(x).relu();
        }
    }

    private static final class ResidualBlock extends Module {
        private final LinearImpl fc1;
        private final LinearImpl fc2;

        ResidualBlock(long features) {
            super("ResidualBlock");
            fc1 = register_module("fc1", new LinearImpl(features, features));
            fc2 = register_module("fc2", new LinearImpl(features, features));
        }

        public Tensor forward(Tensor x) {
            Tensor residual = x;
            Tensor y = fc1.forward(x).relu();
            y = fc2.forward(y);
            return residual.add(y).relu();
        }
    }

    private static final class ResidualBackbone extends Module {
        private final ResidualBlock block1;
        private final ResidualBlock block2;

        ResidualBackbone(long features) {
            super("ResidualBackbone");
            block1 = register_module("block1", new ResidualBlock(features));
            block2 = register_module("block2", new ResidualBlock(features));
        }

        public Tensor forward(Tensor x) {
            x = block1.forward(x);
            x = block2.forward(x);
            return x;
        }
    }

    private static final class ClassifierHead extends Module {
        private final LinearImpl fc1;
        private final LinearImpl fc2;

        ClassifierHead(long inFeatures, long hiddenFeatures, long outFeatures) {
            super("ClassifierHead");
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hiddenFeatures));
            fc2 = register_module("fc2", new LinearImpl(hiddenFeatures, outFeatures));
        }

        public Tensor forward(Tensor x) {
            x = fc1.forward(x).relu();
            return fc2.forward(x);
        }
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void checkShape(Tensor tensor, long rows, long cols, String label) {
        check(tensor.dim() == 2, label + " should be rank-2");
        check(tensor.sizes().get(0) == rows, label + " should have " + rows + " rows");
        check(tensor.sizes().get(1) == cols, label + " should have " + cols + " cols");
    }

    private static SequentialImpl buildSequential() {
        SequentialImpl seq = new SequentialImpl();
        seq.push_back(new InputStem(4L, 8L));
        seq.push_back(new BytePointer("resnet_like"), new ResidualBackbone(8L));
        seq.push_back("classifier", new ClassifierHead(8L, 6L, 2L));
        return seq;
    }

    private static void verifySequentialWithCustomModules() {
        try (PointerScope scope = new PointerScope()) {
            SequentialImpl seq = buildSequential();
            check(seq.size() == 3, "SequentialImpl should contain three custom modules");

            Tensor input = torch.randn(new long[]{5, 4});
            Tensor output = seq.forward(input);
            checkShape(output, 5L, 2L, "sequential output");
            System.out.println("forward output sum = " + output.sum().item_double());

            ResidualBackbone backbone = new ResidualBackbone(8L);
            StringTensorDict params = backbone.named_parameters();
            check(params.size() > 0, "residual backbone should expose registered parameters");

            SequentialImpl seq2 = new SequentialImpl();
            seq2.push_back("stem", new InputStem(4L, 8L));
            seq2.push_back(new BytePointer("backbone"), new ResidualBackbone(8L));
            seq2.push_back(new ClassifierHead(8L, 6L, 2L));
            check(seq2.size() == 3, "second SequentialImpl should also contain three modules");

            Tensor output2 = seq2.forward(torch.randn(new long[]{2, 4}));
            checkShape(output2, 2L, 2L, "second sequential output");

            System.out.println("residual backbone named_parameters = " + params.size());
            System.out.println("ALL OK");
        }
    }

    private static void benchmarkComplexInsertion() {
        final int iterations = 128;
        long start = System.nanoTime();
        double checksum = 0.0;

        try (PointerScope scope = new PointerScope()) {
            for (int i = 0; i < iterations; i++) {
                SequentialImpl seq = buildSequential();
                Tensor output = seq.forward(torch.randn(new long[]{4, 4}));
                checksum += output.sum().item_double();

                check(seq.size() == 3, "benchmark sequence should keep three modules");
                checkShape(output, 4L, 2L, "benchmark output");
            }
        }

        long elapsedMicros = (System.nanoTime() - start) / 1_000L;
        System.out.println("benchmark iterations = " + iterations
                + ", elapsedMicros = " + elapsedMicros
                + ", checksum = " + checksum);
    }

    public static void main(String[] args) {
        Loader.load(torch.class);
        verifySequentialWithCustomModules();
        benchmarkComplexInsertion();
    }
}
