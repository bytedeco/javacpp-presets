import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.EmbeddingBagFromPretrainedOptions;
import org.bytedeco.pytorch.EmbeddingBagImpl;
import org.bytedeco.pytorch.EmbeddingFromPretrainedOptions;
import org.bytedeco.pytorch.EmbeddingImpl;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;
/*
//   1. SequentialImpl 支持 insert push_back 任意的继承自Module的复杂类了 和构造函数 参数字典支持Module，这个需要把cpp libtorch 的 sequential.h any.h 可能需要部分修改，可以追加到 cppbuild.h
     2. Module 支持 大部分forward 虚方法了，可以直接在ModuleDict ModuleList 不用强转实际类型调用forward，不过对7个layer的forward造成冲突破坏，被迫重命名
     3. Module 支持 as 强转 任意实际类型的 layer model
     4. ModuleDict 支持 insert 继承自 Module任意复杂layer，这个需要把cpp libtorch 的 ModuleDict.h private 修改为public, 可以追加到 cppbuild.h
     5. Embedding EmbeddingBag 支持 from_pretrained
     6. ModuleDict ModuleList 支持 forward 直接调用子module的forward

*/
public class BenchmarkEmbeddingFromPretrained {
    private static Tensor deterministicMatrix(long rows, long cols, long offset) {
        Tensor x = arange(new org.bytedeco.pytorch.Scalar(offset), new org.bytedeco.pytorch.Scalar(offset + rows * cols));
        return x.reshape(rows, cols).div(new org.bytedeco.pytorch.Scalar(rows * cols));
    }

    private static void check(boolean ok, String msg) {
        if (!ok) {
            throw new AssertionError(msg);
        }
    }

    private static void runEmbeddingBench() {
        final int steps = 200;
        Tensor weights = deterministicMatrix(2048, 64, 10_000);
        Tensor indices = arange(new org.bytedeco.pytorch.Scalar(0), new org.bytedeco.pytorch.Scalar(8192))
                .remainder(new org.bytedeco.pytorch.Scalar(2048)).reshape(128, 64);

        long t0 = System.nanoTime();
        EmbeddingImpl embedding = EmbeddingImpl.from_pretrained(weights, new EmbeddingFromPretrainedOptions());
        float init = embedding.forward(indices).sum().item_float();
        float last = init;
        for (int i = 0; i < steps; i++) {
            last = embedding.forward(indices).sum().item_float();
        }
        float fin = embedding.forward(indices).sum().item_float();
        check(fin == init, "Embedding from_pretrained output changed");
        System.out.println("Embedding-from_pretrained: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    private static void runEmbeddingBagBench() {
        final int steps = 200;
        Tensor weights = deterministicMatrix(2048, 64, 20_000);
        Tensor indices = arange(new org.bytedeco.pytorch.Scalar(0), new org.bytedeco.pytorch.Scalar(8192))
                .remainder(new org.bytedeco.pytorch.Scalar(2048));
        Tensor offsets = arange(new org.bytedeco.pytorch.Scalar(0), new org.bytedeco.pytorch.Scalar(128))
                .mul(new org.bytedeco.pytorch.Scalar(64));

        long t0 = System.nanoTime();
        EmbeddingBagImpl embeddingBag = EmbeddingBagImpl.from_pretrained(weights, new EmbeddingBagFromPretrainedOptions());
        float init = embeddingBag.forward(indices, offsets).sum().item_float();
        float last = init;
        for (int i = 0; i < steps; i++) {
            last = embeddingBag.forward(indices, offsets).sum().item_float();
        }
        float fin = embeddingBag.forward(indices, offsets).sum().item_float();
        check(fin == init, "EmbeddingBag from_pretrained output changed");
        System.out.println("EmbeddingBag-from_pretrained: init=" + init + ", last=" + last + ", final=" + fin
                + ", ms=" + ((System.nanoTime() - t0) / 1_000_000L));
    }

    public static void main(String[] args) {
        Loader.load(org.bytedeco.pytorch.global.torch.class);
        manual_seed(2026);
        try (PointerScope scope = new PointerScope()) {
            runEmbeddingBench();
            runEmbeddingBagBench();
            System.out.println("EMBEDDING FROM_PRETRAINED BENCH OK");
        }
    }
}
