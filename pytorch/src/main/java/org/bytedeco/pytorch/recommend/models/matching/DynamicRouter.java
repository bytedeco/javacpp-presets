/*
 * Ported from torch-rechub-scala: torchrec/models/matching/ComirecDR.scala
 * (DynamicRouter + ComirecDR)
 *
 * Comirec-DR: Dynamic Routing Multi-Interest Framework. Reference: RecSys 2020
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;

/** Dynamic Router for Multiple Interests. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DynamicRouter extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numInterests;
    private final int numRoutings;

    public DynamicRouter(int embedDim, int numInterests) {
        this(embedDim, numInterests, 3);
    }

    public DynamicRouter(int embedDim, int numInterests, int numRoutings) {
        super("DynamicRouter");
        this.embedDim = embedDim;
        this.numInterests = numInterests;
        this.numRoutings = numRoutings;
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, seq_len, embed_dim)
        int batchSize = (int) x.size(0);

        Tensor capsules = torch.randn(batchSize, numInterests, embedDim)
                .mul(new Scalar(0.1f))
                .to(x.device(), ScalarType.Float);

        Tensor weightedItems = capsules;
        for (int r = 0; r < numRoutings; r++) {
            Tensor expandedCaps = capsules.unsqueeze(1); // (batch, 1, num_interests, embed_dim)
            Tensor expandedItems = x.unsqueeze(2); // (batch, seq_len, 1, embed_dim)
            Tensor similarities = expandedCaps.mul(expandedItems).sum(3); // (batch, seq_len, num_interests)

            Tensor weights = similarities.softmax(2);

            weightedItems = x.unsqueeze(2).mul(weights.unsqueeze(3)).sum(1); // (batch, num_interests, embed_dim)

            if (r < numRoutings - 1) {
                capsules = weightedItems;
            }
        }
        return weightedItems;
    }
}
