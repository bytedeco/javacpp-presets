package org.example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.DropoutImpl;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.ModuleDictImpl;
import org.bytedeco.pytorch.ModuleListImpl;
import org.bytedeco.pytorch.ReLUImpl;
import org.bytedeco.pytorch.SequentialImpl;
import org.bytedeco.pytorch.SoftmaxImpl;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorDataset;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

/**
 * Verifies System.out.println on Module / Tensor / TensorDataset /
 * DataLoader / ModuleDict / ModuleList / MMoE-style model.
 */
public class BenchmarkDebugPrintV1 {

    public static void main(String[] args) {
        Loader.load(torch.class);

        System.out.println("=== MLP (Sequential of Linear/ReLU/Dropout) ===");
        System.out.println(buildMlp());

        System.out.println();
        System.out.println("=== MMoE-style model (gates + experts) ===");
        System.out.println(buildMmoeLike());

        System.out.println();
        System.out.println("=== ModuleDict ===");
        System.out.println(buildModuleDict());

        System.out.println();
        System.out.println("=== ModuleList ===");
        System.out.println(buildModuleList());

        System.out.println();
        System.out.println("=== TensorDataset (TensorVector) ===");
        System.out.println(buildTensorDatasetVec());

        System.out.println();
        System.out.println("=== DataLoader (JavaRandomDataLoader over TensorDataset) ===");
//        System.out.println(buildDataLoader());

        System.out.println();
        System.out.println("=== Functional check (original Sequential+Dropout) ===");
        SequentialImpl mlp = buildMlp();
        Tensor x = torch.randn(new long[]{4, 64});
        Tensor y = mlp.forward(x);
        System.out.println("x:");
        System.out.println(x);
        System.out.println("y:");
        System.out.println(y);
    }

    private static SequentialImpl buildMlp() {
        SequentialImpl seq = new SequentialImpl();
        seq.push_back(new LinearImpl(64L, 128L));
        seq.push_back(new ReLUImpl());
        seq.push_back(new DropoutImpl(0.1));
        seq.push_back(new LinearImpl(128L, 64L));
        seq.push_back(new ReLUImpl());
        seq.push_back(new LinearImpl(64L, 10L));
        return seq;
    }

    /** Fake MMoE-style: 3 experts, each a Linear; one gate (Linear+Softmax). */
    private static Module buildMmoeLike() {
        ModuleListImpl experts = new ModuleListImpl();
        experts.push_back(new LinearImpl(64L, 32L));
        experts.push_back(new LinearImpl(64L, 32L));
        experts.push_back(new LinearImpl(64L, 32L));

        SequentialImpl gate = new SequentialImpl();
        gate.push_back(new LinearImpl(64L, 3));
        gate.push_back(new SoftmaxImpl(1));

        return new MmoeLikeBlock(experts, gate);
    }

    public static final class MmoeLikeBlock extends Module {
        public MmoeLikeBlock(ModuleListImpl experts, SequentialImpl gate) {
            super("MmoeLikeBlock");
            register_module("experts", experts);
            register_module("gate", gate);
        }
    }

    private static ModuleDictImpl buildModuleDict() {
        ModuleDictImpl d = new ModuleDictImpl();
        d.insert("stem", new LinearImpl(64L, 128L));
        SequentialImpl gate = new SequentialImpl();
        gate.push_back(new LinearImpl(64L, 3));
        gate.push_back(new SoftmaxImpl(1));
        d.insert("body", gate);
        d.insert("head", new LinearImpl(64L, 10L));
        return d;
    }

    private static ModuleListImpl buildModuleList() {
        ModuleListImpl l = new ModuleListImpl();
        l.push_back(new LinearImpl(64L, 128L));
        l.push_back(new ReLUImpl());
        l.push_back(new LinearImpl(128L, 10L));
        return l;
    }

    private static TensorDataset buildTensorDatasetVec() {
        int n = 8;
        Tensor x = torch.randn(new long[]{n, 3});
        Tensor y = torch.randint(0, 2, new long[]{n,3},
                                 new org.bytedeco.pytorch.TensorOptions()
                                     .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(torch.kLong())));
        TensorVector tv = new TensorVector();
        tv.push_back(x);
        tv.push_back(y);
        return new TensorDataset(tv);
    }

//    private static JavaRandomDataLoader buildDataLoader() {
//        TensorDataset ds = buildTensorDatasetVec();
//        // JavaRandomDataLoader(dataset, batch_size)
//        return new JavaRandomDataLoader(ds, 4L);
//    }
}
