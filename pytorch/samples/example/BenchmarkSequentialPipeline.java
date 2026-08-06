package example;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;

import static org.bytedeco.pytorch.global.torch.manual_seed;
import static org.bytedeco.pytorch.global.torch.randn;

public class BenchmarkSequentialPipeline {
    private static final class InputStem extends Module {
        private final LinearImpl fc;
        InputStem(long inFeatures, long outFeatures) {
            super("InputStem");
            fc = register_module("fc", new LinearImpl(inFeatures, outFeatures));
        }
        public Tensor forward(Tensor x) {
            return fc.forward(x).relu();
        }
    }

    private static final class ResidualBlock extends Module {
        private final LinearImpl fc1, fc2;
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
        private final ResidualBlock block1, block2;
        ResidualBackbone(long features) {
            super("ResidualBackbone");
            block1 = register_module("block1", new ResidualBlock(features));
            block2 = register_module("block2", new ResidualBlock(features));
        }
        public Tensor forward(Tensor x) {
            return block2.forward(block1.forward(x));
        }
    }

    private static final class RegressorHead extends Module {
        private final LinearImpl fc1, fc2;
        RegressorHead(long inFeatures, long hiddenFeatures, long outFeatures) {
            super("RegressorHead");
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hiddenFeatures));
            fc2 = register_module("fc2", new LinearImpl(hiddenFeatures, outFeatures));
        }
        public Tensor forward(Tensor x) {
            return fc2.forward(fc1.forward(x).relu());
        }
    }

    private static void check(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }

    private static SequentialImpl buildModel() {
        SequentialImpl seq = new SequentialImpl();
        seq.push_back("stem", new InputStem(8, 16));
        seq.push_back("backbone", new ResidualBackbone(16));
        seq.push_back("head", new RegressorHead(16, 16, 4));
        return seq;
    }

    private static ModuleListImpl buildListModel() {
        ModuleListImpl seq = new ModuleListImpl();
        seq.push_back(new InputStem(8, 16));
        seq.push_back(new ResidualBackbone(16));
        seq.push_back(new RegressorHead(16, 16, 4));
        return seq;
    }

    private static Tensor syntheticTargets(Tensor x, Tensor teacherW, Tensor teacherB) {
        return x.matmul(teacherW).add(teacherB).tanh();
    }

    public static void main(String[] args) {
        Loader.load(torch.class);
        manual_seed(1234);

        final long batch = 64;
        final int trainSteps = 300;

        try (PointerScope scope = new PointerScope()) {
            Tensor teacherW = randn(new long[]{8, 4});
            Tensor teacherB = randn(new long[]{4});

            SequentialImpl model = buildModel();
//            ModuleListImpl model = buildListModel();
            SGD optimizer = new SGD(model.parameters(), new SGDOptions(0.03));

            Tensor xVal0 = randn(new long[]{batch, 8});
            Tensor yVal0 = syntheticTargets(xVal0, teacherW, teacherB);
            Tensor pVal0 = model.forward(xVal0);
            float initialLoss = pVal0.sub(yVal0).mul(pVal0.sub(yVal0)).mean().item_float();

            long startNs = System.nanoTime();
            float lastLoss = initialLoss;
            for (int step = 1; step <= trainSteps; step++) {
                Tensor x = randn(new long[]{batch, 8});
                Tensor y = syntheticTargets(x, teacherW, teacherB);

                optimizer.zero_grad();
                Tensor pred = model.forward(x);
                Tensor diff = pred.sub(y);
                Tensor loss = diff.mul(diff).mean();
                loss.backward();
                optimizer.step();
                lastLoss = loss.item_float();
                System.out.println(lastLoss);

            }
            long elapsedMs = (System.nanoTime() - startNs) / 1_000_000L;

            Tensor xVal = randn(new long[]{batch, 8});
            Tensor yVal = syntheticTargets(xVal, teacherW, teacherB);

            Tensor pVal = model.forward(xVal);
            float finalLoss = pVal.sub(yVal).mul(pVal.sub(yVal)).mean().item_float();

            check(model.size() == 3, "custom sequential should keep 3 modules");
            check(finalLoss < initialLoss, "final loss should decrease");

            System.out.println("initialLoss=" + initialLoss
                    + ", lastTrainLoss=" + lastLoss
                    + ", finalValLoss=" + finalLoss);
            System.out.println("steps=" + trainSteps + ", elapsedMs=" + elapsedMs);
            System.out.println("PIPELINE OK");
        }
    }
}
