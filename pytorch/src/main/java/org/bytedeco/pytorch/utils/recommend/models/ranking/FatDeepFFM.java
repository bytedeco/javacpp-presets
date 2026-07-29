/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DeepFFM.scala (FatDeepFFM)
 *
 * Field-Attentive Deep Field-weighted FM — uses all sparse features as fields.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;

import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FatDeepFFM extends DeepFFM {

    public FatDeepFFM(List<? extends Feature> features) {
        this(features, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public FatDeepFFM(List<? extends Feature> features, int embedDim, long[] mlpDims,
                      float dropout, String device) {
        super(features, embedDim, countSparse(features), mlpDims, dropout, device);
    }

    private static int countSparse(List<? extends Feature> features) {
        int n = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) n++;
        }
        return n;
    }
}
