package example;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.EmbeddingBagImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingBagFromPretrainedOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingFromPretrainedOptions;

import static org.bytedeco.pytorch.global.torch.arange;
import static org.bytedeco.pytorch.global.torch.manual_seed;

public class BenchmarkEmbeddingFromPretrainedV1 {
    private static Tensor deterministicMatrix(long rows, long cols, long offset) {
        Tensor x = arange(new Scalar(offset), new Scalar(offset + rows * cols));
        return x.reshape(rows, cols).div(new Scalar(rows * cols));
    }

    private static void check(boolean ok, String msg) {
        if (!ok) {
            throw new AssertionError(msg);
        }
    }

    private static void runEmbeddingBench() {
        final int steps = 200;
        Tensor weights = deterministicMatrix(2048, 64, 10_000);
        Tensor indices = arange(new Scalar(0), new Scalar(8192))
                .remainder(new Scalar(2048)).reshape(128, 64);

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
        Tensor indices = arange(new Scalar(0), new Scalar(8192)).remainder(new Scalar(2048));
        Tensor offsets = arange(new Scalar(0), new Scalar(128)).mul(new Scalar(64));

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
        Loader.load(torch.class);
        manual_seed(2026);
        try (PointerScope scope = new PointerScope()) {
            runEmbeddingBench();
            runEmbeddingBagBench();
            System.out.println("EMBEDDING FROM_PRETRAINED BENCH OK");
        }
    }
}
