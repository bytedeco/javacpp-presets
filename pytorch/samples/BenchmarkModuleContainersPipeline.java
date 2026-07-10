package org.example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.ModuleDictImpl;
import org.bytedeco.pytorch.ModuleListImpl;
import org.bytedeco.pytorch.SGD;
import org.bytedeco.pytorch.SGDOptions;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

public class BenchmarkModuleContainersPipeline {
    public static final class ComplexLayerA extends Module {
        private LinearImpl fc1, fc2;
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
        private LinearImpl gate, proj;
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
        private LinearImpl left, right;
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

    public static final class ModuleListModelA extends Module {
        private final ModuleListImpl blocks;
        private final LinearImpl head;
        public ModuleListModelA() {
            super("ModuleListModelA");
            blocks = register_module("blocks", new ModuleListImpl());
            blocks.insert(0, new ComplexLayerA(8, 16));
            blocks.insert(1, new ComplexLayerA(8, 16));
            head = register_module("head", new LinearImpl(8, 4));
        }

        public Tensor forward(Tensor x) {
            x = blocks.get(0).as(ComplexLayerA.class).forward(x);
            x = blocks.get(1).as(ComplexLayerA.class).forward(x);
            return head.forward(x);
        }
    }

    public static final class ModuleListModelB extends Module {
        private final ModuleListImpl blocks;
        private final LinearImpl head;
        public ModuleListModelB() {
            super("ModuleListModelB");
            blocks = register_module("blocks", new ModuleListImpl());
            blocks.insert(0, new ComplexLayerB(8));
            blocks.insert(1, new ComplexLayerB(8));
            blocks.insert(2, new ComplexLayerA(8, 12));
            head = register_module("head", new LinearImpl(8, 4));
        }

        public Tensor forward(Tensor x) {
            x = blocks.get(0).as(ComplexLayerB.class).forward(x);
            x = blocks.get(1).as(ComplexLayerB.class).forward(x);
            x = blocks.get(2).as(ComplexLayerA.class).forward(x);
            return head.forward(x);
        }
    }

    public static final class ModuleDictModelA extends Module {
        private final ModuleDictImpl dict;
        public ModuleDictModelA() {
            super("ModuleDictModelA");
            dict = register_module("dict", new ModuleDictImpl());
            dict.insert("stem", new ComplexLayerA(8, 16));
            dict.insert("body", new ComplexLayerB(8));
            dict.insert("head", new LinearImpl(8, 4));
        }

        public Tensor forward(Tensor x) {
            x = dict.get("stem").as(ComplexLayerA.class).forward(x);
            x = dict.get("body").as(ComplexLayerB.class).forward(x);
            return dict.get("head").as(LinearImpl.class).forward(x);
        }
    }

    public static final class ModuleDictModelPair extends Module {
        private final ModuleDictImpl dict;
        public ModuleDictModelPair() {
            super("ModuleDictModelPair");
            dict = register_module("dict", new ModuleDictImpl());
            dict.insert("left_enc", new ComplexLayerA(8, 16));
            dict.insert("right_enc", new ComplexLayerB(8));
            dict.insert("pair", new PairTupleLayer(8, 4));
        }

        public T_TensorTensor_T forwardT_TensorTensor_T(Tensor a, Tensor b) {
            Tensor ea = dict.get("left_enc").as(ComplexLayerA.class).forward(a);
            Tensor eb = dict.get("right_enc").as(ComplexLayerB.class).forward(b);
            return dict.get("pair").as(PairTupleLayer.class).forwardT_TensorTensor_T(ea, eb);
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

    private static void runModuleListBenchA() {
        final int steps = 220;
        ModuleListModelA model = new ModuleListModelA();
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
        check(fin < init, "ModuleList bench A loss did not decrease");
        System.out.println("ModuleList-A: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runModuleListBenchB() {
        final int steps = 240;
        ModuleListModelB model = new ModuleListModelB();
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
        check(fin < init, "ModuleList bench B loss did not decrease");
        System.out.println("ModuleList-B: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runModuleDictBenchA() {
        final int steps = 220;
        ModuleDictModelA model = new ModuleDictModelA();
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
        check(fin < init, "ModuleDict bench A loss did not decrease");
        System.out.println("ModuleDict-A: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runModuleDictBenchPair() {
        final int steps = 260;
        ModuleDictModelPair model = new ModuleDictModelPair();
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
        check(fin < init, "ModuleDict pair bench loss did not decrease");
        System.out.println("ModuleDict-Pair(T_TensorTensor_T): init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    public static void main(String[] args) {
        Loader.load(org.bytedeco.pytorch.global.torch.class);
        manual_seed(2026);
        try (PointerScope scope = new PointerScope()) {
            runModuleListBenchA();
            runModuleListBenchB();
            runModuleDictBenchA();
            runModuleDictBenchPair();
            System.out.println("MODULE CONTAINER PIPELINE OK");
        }
    }
}
