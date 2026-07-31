/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/OuterProductNetwork.scala
 *
 * Outer Product Network for cross-feature interactions.
 * Kernel types: "mat", "vec", "num".
 * Reference: "Product-based Neural Networks for User Response Prediction" (Song et al., 2016)
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class OuterProductNetwork extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final String kernelType;
    private final int numPairs;
    private final long[] pairRowIndices;
    private final long[] pairColIndices;
    private final Tensor kernel;

    public OuterProductNetwork(int numFields, int embedDim) {
        this(numFields, embedDim, "mat", DeviceSupport.backend());
    }

    public OuterProductNetwork(int numFields, int embedDim, String kernelType) {
        this(numFields, embedDim, kernelType, DeviceSupport.backend());
    }

    public OuterProductNetwork(int numFields, int embedDim, String kernelType, String device) {
        super("OuterProductNetwork");
        if (numFields < 2) {
            throw new IllegalArgumentException("numFields must be >= 2, got " + numFields);
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive, got " + embedDim);
        }
        if (!"mat".equals(kernelType) && !"vec".equals(kernelType) && !"num".equals(kernelType)) {
            throw new IllegalArgumentException(
                    "kernelType must be 'mat', 'vec', or 'num', got " + kernelType);
        }
        this.numFields = numFields;
        this.embedDim = embedDim;
        this.kernelType = kernelType;
        this.numPairs = (numFields * (numFields - 1)) / 2;

        List<Long> rows = new ArrayList<>();
        List<Long> cols = new ArrayList<>();
        for (int i = 0; i < numFields - 1; i++) {
            for (int j = i + 1; j < numFields; j++) {
                rows.add((long) i);
                cols.add((long) j);
            }
        }
        this.pairRowIndices = rows.stream().mapToLong(Long::longValue).toArray();
        this.pairColIndices = cols.stream().mapToLong(Long::longValue).toArray();

        Tensor kernelInit;
        if ("mat".equals(kernelType)) {
            float scale = (float) Math.sqrt(2.0 / (embedDim * 2));
            kernelInit = torch.randn(new long[]{numPairs, embedDim, embedDim})
                    .mul(new Scalar(scale));
        } else {
            float scale = (float) Math.sqrt(2.0 / embedDim);
            kernelInit = torch.randn(new long[]{numPairs, embedDim})
                    .mul(new Scalar(scale));
        }

        // register_parameter ByRef: keep original handle
        Tensor p = new Tensor();
        p.copy_(kernelInit);
        register_parameter("kernel", p);
        this.kernel = p;

        if (device != null && !"cpu".equals(device)) {
            kernel.to(new Device(device), ScalarType.Float);
        }
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        int batchSize = (int) embeddings.size(0);
        Device dev = embeddings.device();

        // Build index tensors — Scala used float array then Long dtype; use direct long.
        Tensor rowT = torch.tensor(pairRowIndices,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                .to(dev, ScalarType.Long);
        Tensor colT = torch.tensor(pairColIndices,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                .to(dev, ScalarType.Long);

        Tensor p = embeddings.index_select(1, rowT);
        Tensor q = embeddings.index_select(1, colT);

        switch (kernelType) {
            case "mat": {
                Tensor k = kernel.to(dev, ScalarType.Float);
                Tensor result = torch.zeros(new long[]{batchSize, numPairs},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                        .to(dev, ScalarType.Float);
                for (int b = 0; b < batchSize; b++) {
                    for (int pIdx = 0; pIdx < numPairs; pIdx++) {
                        Tensor pVec = p.select(0, b).select(0, pIdx);
                        Tensor qVec = q.select(0, b).select(0, pIdx);
                        Tensor wMat = k.select(0, pIdx);
                        Tensor wq = torch.matmul(wMat, qVec);
                        Tensor dot = pVec.dot(wq);
                        result.select(0, b).select(0, pIdx).copy_(dot);
                    }
                }
                return result;
            }
            case "vec": {
                Tensor k = kernel.to(dev, ScalarType.Float);
                Tensor pq = p.mul(q);
                Tensor kB = k.unsqueeze(0).expand(batchSize, numPairs, embedDim);
                return pq.mul(kB).sum(2L);
            }
            case "num": {
                // Scala kernel for non-mat is (numPairs, embedDim); ported as-is.
                Tensor k = kernel.to(dev, ScalarType.Float);
                Tensor pq = p.mul(q).sum(2L);
                Tensor kB = k.unsqueeze(0).expand(batchSize, numPairs);
                return pq.mul(kB);
            }
            default:
                throw new IllegalStateException("Unknown kernelType: " + kernelType);
        }
    }
}
