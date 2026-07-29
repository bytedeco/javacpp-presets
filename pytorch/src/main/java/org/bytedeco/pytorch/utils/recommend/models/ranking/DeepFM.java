/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DeepFM.scala
 *
 * DeepFM: Combine FM (2nd-order) with DNN. Reference: IJCAI 2017.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.FM;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DeepFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> deepFeatures;
    private final List<Feature> fmFeatures;
    private final EmbeddingLayer embeddingLayer;
    private final LinearImpl linear;
    private final FM fm;
    private final MLP mlp;

    public DeepFM(List<? extends Feature> deepFeatures, List<? extends Feature> fmFeatures) {
        this(deepFeatures, fmFeatures, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public DeepFM(List<? extends Feature> deepFeatures, List<? extends Feature> fmFeatures,
                  int embedDim, long[] mlpDims, float dropout, String device) {
        super("DeepFM");
        this.deepFeatures = new ArrayList<>(deepFeatures);
        this.fmFeatures = new ArrayList<>(fmFeatures);

        List<Feature> allFeatures = new ArrayList<>();
        allFeatures.addAll(this.deepFeatures);
        allFeatures.addAll(this.fmFeatures);
        this.embeddingLayer = new EmbeddingLayer(allFeatures, embedDim, device);
        register_module("embedding", embeddingLayer);

        long fmDims = 0L;
        for (Feature f : this.fmFeatures) {
            fmDims += f.embedDim();
        }
        this.linear = new LinearImpl(fmDims, 1);
        register_module("linear", linear);

        this.fm = new FM(embedDim, device);
        register_module("fm", fm);

        long deepDims = 0L;
        for (Feature f : this.deepFeatures) {
            deepDims += f.embedDim();
        }
        this.mlp = new MLP(deepDims, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            linear.to(dev, false);
            fm.to(dev, false);
            mlp.to(dev, false);
            embeddingLayer.toDevice(device);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Map<String, Tensor> deepSparse = new LinkedHashMap<>();
        for (Feature f : deepFeatures) {
            deepSparse.put(f.name(), sparseFeats.get(f.name()));
        }
        Map<String, Tensor> fmSparse = new LinkedHashMap<>();
        for (Feature f : fmFeatures) {
            fmSparse.put(f.name(), sparseFeats.get(f.name()));
        }

        Tensor deepEmbeddings = embeddingLayer.forward3D(deepSparse, Collections.emptyMap());
        Tensor fmEmbeddings = embeddingLayer.forward3D(fmSparse, Collections.emptyMap());
        int batchSize = (int) deepEmbeddings.size(0);

        Tensor fmFlattened = fmEmbeddings.view(batchSize, -1);
        Tensor linearOut = linear.forward(fmFlattened);

        Tensor fmOut = fm.forward(fmEmbeddings);

        Tensor deepFlattened = deepEmbeddings.view(batchSize, -1);
        Tensor mlpOut = mlp.forward(deepFlattened);

        return linearOut.add(fmOut).add(mlpOut);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
