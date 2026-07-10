package org.example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.ReLUImpl;
import org.bytedeco.pytorch.SGD;
import org.bytedeco.pytorch.SGDOptions;
import org.bytedeco.pytorch.SequentialImpl;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

public class BenchmarkSequentialContainersPipeline {
    public static final class ComplexLayerA extends Module {
        private final LinearImpl fc1, fc2;

        public ComplexLayerA(long inFeatures, long hiddenFeatures) {
            super("ComplexLayerA");
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hiddenFeatures));
            fc2 = register_module("fc2", new LinearImpl(hiddenFeatures, inFeatures));
        }

        public Tensor forward(Tensor x) {
            Tensor y = fc1.forward(x).relu();
            y = fc2.forward(y);
            return x.add(y).relu();
        }
    }

    public static final class ComplexLayerB extends Module {
        private final LinearImpl gate, proj;

        public ComplexLayerB(long features) {
            super("ComplexLayerB");
            gate = register_module("gate", new LinearImpl(features, features));
            proj = register_module("proj", new LinearImpl(features, features));
        }

        public Tensor forward(Tensor x) {
            Tensor g = gate.forward(x).sigmoid();
            Tensor p = proj.forward(x);
            return p.mul(g).add(x).relu();
        }
    }

    public static final class PairTupleLayer extends Module {
        private final LinearImpl left, right;

        public PairTupleLayer(long inFeatures, long outFeatures) {
            super("PairTupleLayer");
            left = register_module("left", new LinearImpl(inFeatures, outFeatures));
            right = register_module("right", new LinearImpl(inFeatures, outFeatures));
        }

        public T_TensorTensor_T forwardT_TensorTensor_T(Tensor a, Tensor b) {
            Tensor y1 = left.forward(a).add(right.forward(b)).tanh();
            Tensor y2 = left.forward(b).sub(right.forward(a)).tanh();
            return new T_TensorTensor_T(y1, y2);
        }
    }

    public static final class SequentialModelA extends Module {
        private final SequentialImpl blocks;
        private final ComplexLayerA block0;
        private final ComplexLayerA block1;
        private final LinearImpl head;

        public SequentialModelA() {
            super("SequentialModelA");
            blocks = register_module("blocks", new SequentialImpl());
            block0 = new ComplexLayerA(8, 16);
            block1 = new ComplexLayerA(8, 16);
            blocks.push_back(block0);
            blocks.push_back(block1);
            head = register_module("head", new LinearImpl(8, 4));
        }

        public Tensor forward(Tensor x) {
            x = block0.forward(x);
            x = block1.forward(x);
            return head.forward(x);
        }
    }

    public static final class SequentialModelB extends Module {
        private final SequentialImpl blocks;
        private final ComplexLayerB block0;
        private final ComplexLayerB block1;
        private final ComplexLayerA block2;
        private final LinearImpl head;

        public SequentialModelB() {
            super("SequentialModelB");
            blocks = register_module("blocks", new SequentialImpl());
            block0 = new ComplexLayerB(8);
            block1 = new ComplexLayerB(8);
            block2 = new ComplexLayerA(8, 12);
            blocks.push_back(block0);
            blocks.push_back(block1);
            blocks.push_back(block2);
            head = register_module("head", new LinearImpl(8, 4));
        }

        public Tensor forward(Tensor x) {
            x = block0.forward(x);
            x = block1.forward(x);
            x = block2.forward(x);
            return head.forward(x);
        }
    }

    public static final class SequentialDictModelA extends Module {
        private final SequentialImpl dict;
        private final ComplexLayerA stem;
        private final ComplexLayerB body;
        private final LinearImpl head;

        public SequentialDictModelA() {
            super("SequentialDictModelA");
            dict = register_module("dict", new SequentialImpl());
            stem = new ComplexLayerA(8, 16);
            body = new ComplexLayerB(8);
            head = new LinearImpl(8, 4);
            dict.push_back("stem", stem);
            dict.push_back("body", body);
            dict.push_back("head", head);
        }

        public Tensor forward(Tensor x) {
            x = stem.forward(x);
            x = body.forward(x);
            return head.forward(x);
        }
    }

    public static final class SequentialDictModelPair extends Module {
        private final SequentialImpl dict;
        private final ComplexLayerA leftEnc;
        private final ComplexLayerB rightEnc;
        private final PairTupleLayer pair;

        public SequentialDictModelPair() {
            super("SequentialDictModelPair");
            dict = register_module("dict", new SequentialImpl());
            leftEnc = new ComplexLayerA(8, 16);
            rightEnc = new ComplexLayerB(8);
            pair = new PairTupleLayer(8, 4);
            dict.push_back("left_enc", leftEnc);
            dict.push_back("right_enc", rightEnc);
            dict.push_back("pair", pair);
        }

        public T_TensorTensor_T forwardT_TensorTensor_T(Tensor a, Tensor b) {
            Tensor ea = leftEnc.forward(a);
            Tensor eb = rightEnc.forward(b);
            return pair.forwardT_TensorTensor_T(ea, eb);
        }
    }

    private static Tensor deterministicMatrix(long rows, long cols, long offset) {
        Tensor x = arange(new org.bytedeco.pytorch.Scalar(offset), new org.bytedeco.pytorch.Scalar(offset + rows * cols));
        return x.reshape(rows, cols).div(new org.bytedeco.pytorch.Scalar(rows * cols));
    }

    private static Tensor deterministicTarget(Tensor x, Tensor w, Tensor b) {
        return x.matmul(w).add(b).tanh();
    }

    private static Tensor mse(Tensor a, Tensor b) {
        Tensor d = a.sub(b);
        return d.mul(d).mean();
    }

    private static void check(boolean ok, String msg) {
        if (!ok) throw new AssertionError(msg);
    }

    private static void runSequentialBenchA() {
        final int steps = 220;
        SequentialModelA model = new SequentialModelA();
        SGD opt = new SGD(model.parameters(), new SGDOptions(0.08));
        Tensor w = deterministicMatrix(8, 4, 10);
        Tensor b = deterministicMatrix(1, 4, 100).reshape(4);
        Tensor x = deterministicMatrix(96, 8, 1000);
        Tensor y = deterministicTarget(x, w, b);
        float init = mse(model.forward(x), y).item_float();
        long t0 = System.nanoTime();
        float last = init;
        for (int i = 0; i < steps; i++) {
            opt.zero_grad();
            Tensor loss = mse(model.forward(x), y);
            loss.backward();
            opt.step();
            last = loss.item_float();
        }
        float fin = mse(model.forward(x), y).item_float();
        check(fin < init, "Sequential bench A loss did not decrease");
        System.out.println("Sequential-A: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runSequentialBenchB() {
        final int steps = 240;
        SequentialModelB model = new SequentialModelB();
        SGD opt = new SGD(model.parameters(), new SGDOptions(0.06));
        Tensor w = deterministicMatrix(8, 4, 200);
        Tensor b = deterministicMatrix(1, 4, 300).reshape(4);
        Tensor x = deterministicMatrix(120, 8, 2000);
        Tensor y = deterministicTarget(x, w, b);
        float init = mse(model.forward(x), y).item_float();
        long t0 = System.nanoTime();
        float last = init;
        for (int i = 0; i < steps; i++) {
            opt.zero_grad();
            Tensor loss = mse(model.forward(x), y);
            loss.backward();
            opt.step();
            last = loss.item_float();
        }
        float fin = mse(model.forward(x), y).item_float();
        check(fin < init, "Sequential bench B loss did not decrease");
        System.out.println("Sequential-B: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runSequentialDictBenchA() {
        final int steps = 220;
        SequentialDictModelA model = new SequentialDictModelA();
        SGD opt = new SGD(model.parameters(), new SGDOptions(0.07));
        Tensor w = deterministicMatrix(8, 4, 400);
        Tensor b = deterministicMatrix(1, 4, 500).reshape(4);
        Tensor x = deterministicMatrix(96, 8, 3000);
        Tensor y = deterministicTarget(x, w, b);
        float init = mse(model.forward(x), y).item_float();
        long t0 = System.nanoTime();
        float last = init;
        for (int i = 0; i < steps; i++) {
            opt.zero_grad();
            Tensor loss = mse(model.forward(x), y);
            loss.backward();
            opt.step();
            last = loss.item_float();
        }
        float fin = mse(model.forward(x), y).item_float();
        check(fin < init, "Sequential dict bench A loss did not decrease");
        System.out.println("Sequential-Dict-A: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runSequentialDictBenchPair() {
        final int steps = 260;
        SequentialDictModelPair model = new SequentialDictModelPair();
        SGD opt = new SGD(model.parameters(), new SGDOptions(0.05));

        Tensor xa = deterministicMatrix(128, 8, 5000);
        Tensor xb = deterministicMatrix(128, 8, 8000);
        Tensor wa = deterministicMatrix(8, 4, 9000);
        Tensor wb = deterministicMatrix(8, 4, 9500);
        Tensor y1 = deterministicTarget(xa, wa, deterministicMatrix(1, 4, 9800).reshape(4))
                .add(deterministicTarget(xb, wb, deterministicMatrix(1, 4, 9900).reshape(4)));
        Tensor y2 = deterministicTarget(xa.sub(xb), deterministicMatrix(8, 4, 10000),
                deterministicMatrix(1, 4, 10100).reshape(4));

        T_TensorTensor_T p0 = model.forwardT_TensorTensor_T(xa, xb);
        float init = mse(p0.get0(), y1).add(mse(p0.get1(), y2)).item_float();

        long t0 = System.nanoTime();
        float last = init;
        for (int i = 0; i < steps; i++) {
            opt.zero_grad();
            T_TensorTensor_T p = model.forwardT_TensorTensor_T(xa, xb);
            Tensor loss = mse(p.get0(), y1).add(mse(p.get1(), y2));
            loss.backward();
            opt.step();
            last = loss.item_float();
        }
        T_TensorTensor_T pf = model.forwardT_TensorTensor_T(xa, xb);
        float fin = mse(pf.get0(), y1).add(mse(pf.get1(), y2)).item_float();
        check(fin < init, "Sequential dict pair bench loss did not decrease");
        System.out.println("Sequential-Dict-Pair(T_TensorTensor_T): init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    public static void main(String[] args) {
        Loader.load(org.bytedeco.pytorch.global.torch.class);
        manual_seed(2026);
        try (PointerScope scope = new PointerScope()) {
            runSequentialBenchA();
            runSequentialBenchB();
            runSequentialDictBenchA();
            runSequentialDictBenchPair();
            System.out.println("SEQUENTIAL CONTAINER PIPELINE OK");
        }
    }
}
