/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/AnovaKernel.scala
 *
 * Anova Kernel for high-order polynomial interactions.
 * Reference: "Factorization Machines" (Rendle, 2010) - High-order extension
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
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
public class AnovaKernel extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int order;
    private final int embedDim;
    private final boolean reduceSum;
    private final String device;

    public AnovaKernel(int order, int embedDim) {
        this(order, embedDim, true, DeviceSupport.backend());
    }

    public AnovaKernel(int order, int embedDim, boolean reduceSum) {
        this(order, embedDim, reduceSum, DeviceSupport.backend());
    }

    public AnovaKernel(int order, int embedDim, boolean reduceSum, String device) {
        super("AnovaKernel");
        if (order < 2) {
            throw new IllegalArgumentException("order must be >= 2, got " + order);
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive, got " + embedDim);
        }
        this.order = order;
        this.embedDim = embedDim;
        this.reduceSum = reduceSum;
        this.device = device;
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        long batchSize = embeddings.size(0);
        int numFields = (int) embeddings.size(1);
        long eDim = embeddings.size(2);
        Device dev = embeddings.device();

        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor acc = torch.zeros(new long[]{batchSize, eDim}, floatOpts).to(dev, ScalarType.Float);

        int[] indices = new int[numFields];
        for (int i = 0; i < numFields; i++) {
            indices[i] = i;
        }
        List<int[]> comb = combinations(indices, order);
        for (int[] c : comb) {
            Tensor prod = embeddings.narrow(1, c[0], 1).squeeze(1);
            for (int idx = 1; idx < c.length; idx++) {
                Tensor t = embeddings.narrow(1, c[idx], 1).squeeze(1);
                prod = prod.mul(t);
            }
            acc = acc.add(prod);
        }

        if (reduceSum) {
            return acc.sum(1L).unsqueeze(1);
        }
        return acc;
    }

    private static List<int[]> combinations(int[] arr, int k) {
        List<int[]> result = new ArrayList<>();
        if (k == 0) {
            result.add(new int[0]);
            return result;
        }
        if (arr.length < k) {
            return result;
        }
        for (int i = 0; i < arr.length; i++) {
            int[] rest = new int[arr.length - i - 1];
            System.arraycopy(arr, i + 1, rest, 0, rest.length);
            for (int[] r : combinations(rest, k - 1)) {
                int[] combo = new int[r.length + 1];
                combo[0] = arr[i];
                System.arraycopy(r, 0, combo, 1, r.length);
                result.add(combo);
            }
        }
        return result;
    }
}
