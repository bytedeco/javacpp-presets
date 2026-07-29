/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/InnerProductNetwork.scala
 *
 * Inner Product Network — pairwise inner products of field embeddings.
 * Reference: "Product-based Neural Networks for User Response Prediction" (SJTU, 2016)
 *
 * Input:  (batch, num_fields, embed_dim)
 * Output: (batch, num_pairs)
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class InnerProductNetwork extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String device;

    public InnerProductNetwork() {
        this(DeviceSupport.backend());
    }

    public InnerProductNetwork(String device) {
        super("InnerProductNetwork");
        this.device = device;
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        long numFields = embeddings.size(1);
        List<Tensor> outputs = new ArrayList<>();

        for (long i = 0; i < numFields; i++) {
            for (long j = i + 1; j < numFields; j++) {
                Tensor vi = embeddings.narrow(1, i, 1).squeeze(1);  // (batch, embed_dim)
                Tensor vj = embeddings.narrow(1, j, 1).squeeze(1);  // (batch, embed_dim)
                Tensor ip = vi.mul(vj).sum(1).unsqueeze(1);  // (batch, 1)
                outputs.add(ip);
            }
        }

        if (outputs.isEmpty()) {
            TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
            return torch.zeros(new long[]{embeddings.size(0), 1}, opts)
                    .to(embeddings.device(), ScalarType.Float);
        } else if (outputs.size() == 1) {
            return outputs.get(0);
        } else {
            // Avoid torch.cat over TensorVector which may fail if any intermediate lacks device.
            int numOut = outputs.size();
            TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
            Tensor result = torch.zeros(new long[]{embeddings.size(0), numOut}, opts)
                    .to(embeddings.device(), ScalarType.Float);
            for (int k = 0; k < numOut; k++) {
                result.narrow(1, k, 1).copy_(outputs.get(k));
            }
            return result;
        }
    }
}
