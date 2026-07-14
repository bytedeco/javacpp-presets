package samples;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Adam;
import org.bytedeco.pytorch.AdamOptions;
import org.bytedeco.pytorch.CrossEntropyLossImpl;
import org.bytedeco.pytorch.CrossEntropyLossOptions;
import org.bytedeco.pytorch.DropoutImpl;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.MSELossImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.ModuleDictImpl;
import org.bytedeco.pytorch.ModuleListImpl;
import org.bytedeco.pytorch.ReLUImpl;
import org.bytedeco.pytorch.MSELossImpl;
import org.bytedeco.pytorch.Parameter;
import org.bytedeco.pytorch.ParameterListImpl;
import org.bytedeco.pytorch.RandomSampler;
import org.bytedeco.pytorch.SGD;
import org.bytedeco.pytorch.SGDOptions;
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
public class BenchmarkDebugPrint {

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
        System.out.println("=== Optimizer: Adam ===");
        System.out.println(buildAdam());

        System.out.println();
        System.out.println("=== Optimizer: SGD (with momentum) ===");
        System.out.println(buildSGD());

        System.out.println();
        System.out.println("=== Loss: MSELoss ===");
        System.out.println(new MSELossImpl());

        System.out.println();
        System.out.println("=== Loss: CrossEntropyLoss (with label_smoothing, ignore_index) ===");
        System.out.println(new CrossEntropyLossImpl(new CrossEntropyLossOptions()
                .label_smoothing(0.1).ignore_index(0L)));

        System.out.println();
        System.out.println("=== DataLoaderOptions (batch=32, num_workers=4) ===");
        org.bytedeco.pytorch.DataLoaderOptions dlo =
            new org.bytedeco.pytorch.DataLoaderOptions(32L);
        System.out.println(dlo);

        System.out.println();
        System.out.println("=== FullDataLoaderOptions (built from DataLoaderOptions) ===");
        // timeout(Milliseconds) takes a c10::optional<Milliseconds>;
        // JavaCPP exposes Milliseconds as a 64-bit long. max_jobs and
        // timeout take SizeTOptional / std::optional<Milliseconds>
        // respectively, so we leave them unset and just set workers.
        org.bytedeco.pytorch.FullDataLoaderOptions fdlo =
            new org.bytedeco.pytorch.FullDataLoaderOptions(
                dlo.workers(4L));
        System.out.println(fdlo);

        System.out.println();
        System.out.println("=== RandomSampler (size=100) ===");
        System.out.println(new RandomSampler(100L));

        System.out.println();
        System.out.println("=== Parameter + ParameterList (mirroring Python nn.Parameter) ===");
        ParameterListImpl plist = new ParameterListImpl();
        // Use StringTensorDictItem (name, tensor) for named entries.
        org.bytedeco.pytorch.StringTensorDictItem weight =
            new org.bytedeco.pytorch.StringTensorDictItem("weight",
                Parameter.create(torch.randn(new long[]{64, 128})));
        org.bytedeco.pytorch.StringTensorDictItem bias =
            new org.bytedeco.pytorch.StringTensorDictItem("bias",
                Parameter.create(torch.zeros(new long[]{128})));
        plist.append(weight);
        plist.append(bias);
        System.out.println(plist);

        System.out.println();
        System.out.println("=== Loss: CrossEntropyLoss (with label_smoothing=0.1, ignore_index=100) ===");
        // Non-default ignore_index so the attrs are non-trivial and
        // show in the toString output.
        System.out.println(new CrossEntropyLossImpl(new CrossEntropyLossOptions()
                .label_smoothing(0.1).ignore_index(100L)));
        System.out.println("=== Functional check (original Sequential+Dropout) ===");
        SequentialImpl mlp = buildMlp();
        Tensor x = torch.randn(new long[]{4, 64});
        Tensor y = mlp.forward(x);
        System.out.println("x:");
        System.out.println(x);
        System.out.println("y:");
        System.out.println(y);
    }

    private static Adam buildAdam() {
        // Adam(params, AdamOptions(lr=1e-3, betas=(0.9, 0.999), weight_decay=0))
        TensorVector params = new TensorVector();
        params.push_back(torch.randn(new long[]{64, 128}));
        params.push_back(torch.randn(new long[]{128}));
        // betas() takes a DoublePointer (c10::ExpandingArray<double, 2>).
        // Allocate, fill two doubles, pass to the setter.
        org.bytedeco.javacpp.DoublePointer betas = new org.bytedeco.javacpp.DoublePointer(2);
        betas.put(0, 0.9);
        betas.put(1, 0.999);
        AdamOptions opts = new AdamOptions()
                .lr(0.001)
                .betas(betas)
                .weight_decay(0.0);
        return new Adam(params, opts);
    }

    private static SGD buildSGD() {
        TensorVector params = new TensorVector();
        params.push_back(torch.randn(new long[]{64, 64}));
        // SGDOptions(double lr) is the only public ctor besides the
        // Pointer-cast one; subsequent fields are set via fluent
        // setters.
        SGDOptions opts = new SGDOptions(0.01)
                .momentum(0.9)
                .dampening(0.0)
                .weight_decay(0.0001)
                .nesterov(false);
        return new SGD(params, opts);
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
        d.insert("body", new SequentialImpl());
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
        // C++ TensorDataset stacks the tensors along dim 0, so all
        // tensors must share the same shape. Use 2D [n, 1] for both.
        Tensor x = torch.randn(new long[]{n, 1});
        Tensor y = torch.randint(0, 2, new long[]{n, 1},
                                 new org.bytedeco.pytorch.TensorOptions()
                                     .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(torch.kLong())));
        TensorVector tv = new TensorVector();
        tv.push_back(x);
        tv.push_back(y);
        return new TensorDataset(tv);
    }

}
