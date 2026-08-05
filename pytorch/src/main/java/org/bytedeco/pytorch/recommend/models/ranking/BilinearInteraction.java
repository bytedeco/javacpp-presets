/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/FiBiNet.scala (BilinearInteraction)
 *
 * Ranking-local bilinear feature interaction used by FiBiNet.
 * Different from basic.layers.BiLinearInteractionLayer (pair layout / API).
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

/**
 * Bilinear Feature Interaction for FiBiNet.
 * bilinearType: field_all | field_each | field_interaction
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BilinearInteraction extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numFields;
    private final String bilinearType;
    private final LinearImpl sharedWeight;          // field_all
//    private final List<LinearImpl> fieldWeights;    // field_each / field_interaction
    private final ModuleListImpl fieldWeights = new ModuleListImpl();

    public BilinearInteraction(int embedDim, int numFields, String bilinearType) {
        this(embedDim, numFields, bilinearType, DeviceSupport.backend());
    }

    public BilinearInteraction(int embedDim, int numFields, String bilinearType, String device) {
        super("BilinearInteraction");
        this.embedDim = embedDim;
        this.numFields = numFields;
        this.bilinearType = bilinearType;

        if ("field_all".equals(bilinearType)) {
            this.sharedWeight = new LinearImpl(embedDim, embedDim);
            register_module("bilinear_weight", sharedWeight);
//            this.fieldWeights = null;
            if (device != null && !"cpu".equals(device)) {
                sharedWeight.to(new Device(device), false);
            }
        } else {
            this.sharedWeight = null;
//            this.fieldWeights = new ArrayList<>(numFields);
            for (int i = 0; i < numFields; i++) {
                LinearImpl w = new LinearImpl(embedDim, embedDim);
                register_module("bilinear_weight_" + i, w);
                fieldWeights.insert(i,w);
                if (device != null && !"cpu".equals(device)) {
                    w.to(new Device(device), false);
                }
            }
        }
    }

    /**
     * Pairwise bilinear interactions.
     * @param f1 (batch, num_fields, embed_dim)
     * @param f2 (batch, num_fields, embed_dim)
     * @return list of (batch, embed_dim) tensors
     */
    public List<Tensor> forwardPair(Tensor f1, Tensor f2) {
        List<Tensor> out = new ArrayList<>();
        switch (bilinearType) {
            case "field_all": {
                // field_all: single shared bilinear matrix → num_fields^2 pairs
                for (int i = 0; i < numFields; i++) {
                    for (int j = 0; j < numFields; j++) {
                        Tensor vi = f1.select(1, i);
                        Tensor vj = sharedWeight.forward(f2.select(1, j));
                        out.add(vi.mul(vj));
                    }
                }
                break;
            }
            case "field_each": {
                // field_each: each field has its own bilinear matrix → num_fields^2 pairs
                for (int i = 0; i < numFields; i++) {
                    for (int j = 0; j < numFields; j++) {
                        Tensor vi = f1.select(1, i);
                        Tensor vj = fieldWeights.get(i).forward(f2.select(1, j));
                        out.add(vi.mul(vj));
                    }
                }
                break;
            }
            case "field_interaction":
            default: {
                // field_interaction: only i<j pairs
                for (int i = 0; i < numFields; i++) {
                    for (int j = i + 1; j < numFields; j++) {
                        Tensor vi = f1.select(1, i);
                        Tensor vj = fieldWeights.get(i).forward(f2.select(1, j));
                        out.add(vi.mul(vj));
                    }
                }
                break;
            }
        }
        return out;
    }
}
