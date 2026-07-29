/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/EDCN.scala (BridgeModule)
 *
 * Bridge Module for connecting cross and deep networks in EDCN.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

/**
 * Bridge types: hadamard_product | pointwise_addition | concatenation | attention_pooling
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BridgeModule extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int inputDim;
    private final String bridgeType;
    private final SequentialImpl concatPooling;   // nullable
    private final SequentialImpl attentionX;      // nullable
    private final SequentialImpl attentionH;      // nullable

    public BridgeModule(int inputDim, String bridgeType) {
        this(inputDim, bridgeType, DeviceSupport.backend());
    }

    public BridgeModule(int inputDim, String bridgeType, String device) {
        super("BridgeModule");
        this.inputDim = inputDim;
        this.bridgeType = bridgeType;

        if ("concatenation".equals(bridgeType)) {
            SequentialImpl seq = new SequentialImpl();
            seq.push_back("linear", new LinearImpl(inputDim * 2L, inputDim));
            seq.push_back("relu", new ReLUImpl());
            this.concatPooling = seq;
            register_module("concat_pooling", seq);
            this.attentionX = null;
            this.attentionH = null;
        } else if ("attention_pooling".equals(bridgeType)) {
            SequentialImpl attX = new SequentialImpl();
            attX.push_back("linear1", new LinearImpl(inputDim, inputDim));
            attX.push_back("relu", new ReLUImpl());
            attX.push_back("linear2", new LinearImpl(inputDim, inputDim));
            register_module("attention_x", attX);
            this.attentionX = attX;

            SequentialImpl attH = new SequentialImpl();
            attH.push_back("linear1", new LinearImpl(inputDim, inputDim));
            attH.push_back("relu", new ReLUImpl());
            attH.push_back("linear2", new LinearImpl(inputDim, inputDim));
            register_module("attention_h", attH);
            this.attentionH = attH;

            this.concatPooling = null;
        } else {
            this.concatPooling = null;
            this.attentionX = null;
            this.attentionH = null;
        }
    }

    public String bridgeTypeName() {
        return bridgeType;
    }

    public Tensor forward(Tensor x, Tensor h) {
        switch (bridgeType) {
            case "hadamard_product":
                return x.mul(h);
            case "pointwise_addition":
                return x.add(h);
            case "concatenation": {
                TensorVector vec = new TensorVector();
                vec.push_back(x);
                vec.push_back(h);
                Tensor concat = torch.cat(vec, 1L);
                return concatPooling.forward(concat);
            }
            case "attention_pooling": {
                Tensor weightX = torch.softmax(attentionX.forward(x), 1L);
                Tensor weightH = torch.softmax(attentionH.forward(h), 1L);
                return weightX.mul(x).add(weightH.mul(h));
            }
            default:
                return h;
        }
    }
}
