/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/EDCN.scala
 *
 * Enhanced Deep & Cross Network with Bridge and Regulation (EDCN, KDD'2021).
 * Reference: https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.CrossLayer;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.LR;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.RegulationModule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class EDCN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int dims;
    private final int nCrossLayers;
    private final EmbeddingLayer embedding;
//    private final List<CrossLayer> crossLayers = new ArrayList<>();
//    private final List<MLP> mlps = new ArrayList<>();
//    private final List<BridgeModule> bridges = new ArrayList<>();
//    private final List<RegulationModule> regulationModules = new ArrayList<>();
    private final ModuleListImpl crossLayers = new ModuleListImpl();
    private final ModuleListImpl mlps = new ModuleListImpl();
    private final ModuleListImpl bridges = new ModuleListImpl();
    private final ModuleListImpl regulationModules = new ModuleListImpl();
    private final LR finalLinear;

    public EDCN(List<? extends Feature> features) {
        this(features, 3, defaultMlpParams(), "hadamard_product", true, 1.0f, DeviceSupport.backend());
    }

    public EDCN(List<? extends Feature> features, int nCrossLayers, Map<String, Object> mlpParams,
                String bridgeType, boolean useRegulationModule, float temperature, String device) {
        super("EDCN");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("EDCN: features cannot be empty");
        }
        if (nCrossLayers <= 0) {
            throw new IllegalArgumentException("EDCN: nCrossLayers must be > 0, got " + nCrossLayers);
        }
        List<String> allowed = Arrays.asList(
                "hadamard_product", "pointwise_addition", "concatenation", "attention_pooling");
        if (!allowed.contains(bridgeType)) {
            throw new IllegalArgumentException("EDCN: bridgeType '" + bridgeType + "' not supported");
        }

        List<Feature> featList = new ArrayList<>(features);
        this.numFields = featList.size();
        int d = 0;
        int[] feaDimsArr = new int[featList.size()];
        for (int i = 0; i < featList.size(); i++) {
            feaDimsArr[i] = featList.get(i).embedDim();
            d += feaDimsArr[i];
        }
        this.dims = d;
        this.nCrossLayers = nCrossLayers;

        this.embedding = new EmbeddingLayer(featList, 8, device);
        register_module("embedding", embedding);

        if (mlpParams == null) {
            mlpParams = defaultMlpParams();
        }
        String activation = mlpParams.containsKey("activation")
                ? mlpParams.get("activation").toString() : "relu";
        float dropout = mlpParams.containsKey("dropout")
                ? ((Number) mlpParams.get("dropout")).floatValue() : 0.2f;

        for (int i = 0; i < nCrossLayers; i++) {
            CrossLayer crossLayer = new CrossLayer(dims, device);
            register_module("cross_" + i, crossLayer);
            crossLayers.insert(i,crossLayer);

            // Python: mlp dims forced to [dims, dims], outputLayer=false
            MLP mlp = new MLP(dims, new long[]{dims, dims}, dims, activation, dropout,
                    false, false, false, device);
            register_module("mlp_" + i, mlp);
            mlps.insert(i,mlp);

            BridgeModule bridge = new BridgeModule(dims, bridgeType, device);
            register_module("bridge_" + i, bridge);
            bridges.insert(i,bridge);

            RegulationModule reg = new RegulationModule(numFields, feaDimsArr, temperature, useRegulationModule);
            register_module("regulation_" + i, reg);
            regulationModules.insert(i,reg);
        }

        this.finalLinear = new LR(dims * 3L, false, device);
        register_module("final_linear", finalLinear);
    }

    private static Map<String, Object> defaultMlpParams() {
        Map<String, Object> m = new HashMap<>();
        m.put("dims", Arrays.asList(256L, 128L));
        m.put("activation", "relu");
        m.put("dropout", 0.2f);
        return m;
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        Tensor embedX = embedding.forward(sparseFeats, java.util.Collections.emptyMap(), true); // (B, dims)

        T_TensorTensor_T firstReg = ((RegulationModule)regulationModules.get(0)).forwardReg(embedX);
        Tensor crossI = firstReg.get0();
        Tensor deepI = firstReg.get1();
        Tensor cross0 = crossI;
        Tensor bridgeI = crossI.clone();

        for (int i = 0; i < nCrossLayers; i++) {
            if (i > 0) {
                T_TensorTensor_T reg = ((RegulationModule)regulationModules.get(i)).forwardReg(bridgeI);
                crossI = reg.get0();
                deepI = reg.get1();
            }
            crossI = crossI.add(crossLayers.get(i).forward(cross0, crossI));
            deepI = mlps.get(i).forward(deepI);
            bridgeI = bridges.get(i).forward(crossI, deepI);
        }

        TensorVector vec = new TensorVector();
        vec.push_back(crossI);
        vec.push_back(deepI);
        vec.push_back(bridgeI);
        Tensor xStack = torch.cat(vec, 1L);
        Tensor y = finalLinear.forward(xStack);
        return torch.sigmoid(y.squeeze(1));
    }
}
