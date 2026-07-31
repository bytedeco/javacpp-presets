/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/FieldAwareFactorizationMachine.scala
 *
 * Reference: "Field-aware Factorization Machines for CTR Prediction" (Criteo, RecSys 2016)
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
public class FieldAwareFactorizationMachine extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final int numPairs;
    private final long[] pairRowIndices;
    private final long[] pairColIndices;
    private final Tensor ffee;

    public FieldAwareFactorizationMachine(int numFields, int embedDim) {
        this(numFields, embedDim, DeviceSupport.backend());
    }

    public FieldAwareFactorizationMachine(int numFields, int embedDim, String device) {
        super("FieldAwareFactorizationMachine");
        if (numFields < 2) {
            throw new IllegalArgumentException("numFields must be >= 2, got " + numFields);
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive, got " + embedDim);
        }
        this.numFields = numFields;
        this.embedDim = embedDim;
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

        float scale = (float) Math.sqrt(2.0 / embedDim);
        Tensor ffeeInit = torch.randn(new long[]{numFields, numFields - 1, embedDim})
                .mul(new Scalar(scale));
        Tensor p = new Tensor();
        p.copy_(ffeeInit);
        register_parameter("field_aware_embeddings", p);
        this.ffee = p;

        if (device != null && !"cpu".equals(device)) {
            ffee.to(new Device(device), ScalarType.Float);
        }
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        int batchSize = (int) embeddings.size(0);
        Device dev = embeddings.device();
        Tensor ffeeDev = ffee.to(dev, ScalarType.Float);

        Tensor result = torch.zeros(new long[]{batchSize, numPairs},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .to(dev, ScalarType.Float);

        for (int pairIdx = 0; pairIdx < numPairs; pairIdx++) {
            int i = (int) pairRowIndices[pairIdx];
            int j = (int) pairColIndices[pairIdx];

            Tensor vi = embeddings.select(1, i);
            Tensor vj = embeddings.select(1, j);

            // Note: Scala indexes ffee[j, i] and ffee[i, j] but second dim is only (numFields-1).
            // Port mirrors Scala indexing exactly.
            Tensor viAsJ = ffeeDev.select(0, j).select(0, i);
            Tensor vjAsI = ffeeDev.select(0, i).select(0, j);

            Tensor direct = vi.mul(vj).sum(1L);
            Tensor fieldAware = viAsJ.mul(vjAsI).sum();
            Tensor interaction = direct.mul(new Scalar(fieldAware.item_float()));

            result.select(1, pairIdx).copy_(interaction);
        }

        return result.sum(1L);
    }
}
