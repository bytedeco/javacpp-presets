/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/BiLinearInteractionLayer.scala
 *
 * Bilinear feature interaction (FFM-style / FiBiNet).
 * bilinearType: field_all | field_each | field_interaction
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BiLinearInteractionLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long inputDim;
    private final int numFields;
    private final String bilinearType;
    private final int[][] pairs;
    private final List<LinearImpl> bilinearLayers = new ArrayList<>();

    public BiLinearInteractionLayer(long inputDim, int numFields) {
        this(inputDim, numFields, "field_interaction", DeviceSupport.backend());
    }

    public BiLinearInteractionLayer(long inputDim, int numFields, String bilinearType) {
        this(inputDim, numFields, bilinearType, DeviceSupport.backend());
    }

    public BiLinearInteractionLayer(long inputDim, int numFields, String bilinearType, String device) {
        super("BiLinearInteractionLayer");
        if (numFields < 2) {
            throw new IllegalArgumentException(
                    "BiLinearInteractionLayer needs numFields >= 2, got " + numFields);
        }
        this.inputDim = inputDim;
        this.numFields = numFields;
        this.bilinearType = bilinearType;

        List<int[]> pairList = new ArrayList<>();
        for (int i = 0; i < numFields; i++) {
            for (int j = i + 1; j < numFields; j++) {
                pairList.add(new int[]{i, j});
            }
        }
        this.pairs = pairList.toArray(new int[0][]);

        // Skip Module.to(device) for bias=false LinearImpl (Scala notes crash risk).
        switch (bilinearType) {
            case "field_all": {
                LinearImpl layer = new LinearImpl(new LinearOptions(inputDim, inputDim).bias(false));
                register_module("bilinear_layer", layer);
                bilinearLayers.add(layer);
                break;
            }
            case "field_each": {
                for (int i = 0; i < numFields; i++) {
                    LinearImpl layer = new LinearImpl(new LinearOptions(inputDim, inputDim).bias(false));
                    register_module("bilinear_layer_" + i, layer);
                    bilinearLayers.add(layer);
                }
                break;
            }
            case "field_interaction": {
                for (int idx = 0; idx < pairs.length; idx++) {
                    LinearImpl layer = new LinearImpl(new LinearOptions(inputDim, inputDim).bias(false));
                    register_module("bilinear_layer_" + idx, layer);
                    bilinearLayers.add(layer);
                }
                break;
            }
            default:
                throw new UnsupportedOperationException("bilinearType " + bilinearType + " not implemented");
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        int nf = (int) x.size(1);
        Tensor[] fields = new Tensor[nf];
        for (int i = 0; i < nf; i++) {
            fields[i] = x.select(1, i);
        }

        List<Tensor> out = new ArrayList<>();
        switch (bilinearType) {
            case "field_all": {
                LinearImpl shared = bilinearLayers.get(0);
                for (int[] pair : pairs) {
                    out.add(shared.forward(fields[pair[0]]).mul(fields[pair[1]]));
                }
                break;
            }
            case "field_each": {
                for (int[] pair : pairs) {
                    int i = pair[0];
                    int j = pair[1];
                    out.add(bilinearLayers.get(i).forward(fields[i]).mul(fields[j]));
                }
                break;
            }
            case "field_interaction": {
                for (int idx = 0; idx < pairs.length; idx++) {
                    int i = pairs[idx][0];
                    int j = pairs[idx][1];
                    out.add(bilinearLayers.get(idx).forward(fields[i]).mul(fields[j]));
                }
                break;
            }
            default:
                throw new UnsupportedOperationException("bilinearType " + bilinearType + " not implemented");
        }

        if (out.isEmpty()) {
            return torch.empty(0L);
        }
        TensorVector vec = new TensorVector();
        for (Tensor t : out) {
            vec.push_back(t);
        }
        return torch.cat(vec, 1L);
    }
}
