/*
 * TwinTowerPPI — twin-tower protein–protein interaction scorer.
 *
 * Uses ProteinSeqEncoder for each partner; score = MLP([z1; z2; z1⊙z2; |z1-z2|]).
 * Common formulation in PPI prediction benchmarks (yeast, human PPI datasets).
 *
 * References (representative):
 *   - Chen et al., "Multifaceted protein-protein interaction prediction based
 *     on Siamese residual RCNN", Bioinformatics 2019
 *   - ESM / ProtTrans embeddings + MLP heads in recent PPI transfer learning
 */
package org.bytedeco.pytorch.utils.recommend.models.bio;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TwinTowerPPI extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final ProteinSeqEncoder encoder; // shared weights (Siamese)
    private final MLP head;
    private final boolean shareEncoder;
    private final ProteinSeqEncoder encoder2; // only if not shared

    public TwinTowerPPI(int vocabSize, int maxLen) {
        this(vocabSize, maxLen, 128, 4, 2, true, DeviceSupport.backend());
    }

    public TwinTowerPPI(int vocabSize, int maxLen, int embedDim, int numHeads,
                        int numLayers, boolean shareEncoder, String device) {
        super("TwinTowerPPI");
        this.shareEncoder = shareEncoder;
        this.encoder = new ProteinSeqEncoder(vocabSize, maxLen, embedDim, numHeads,
                numLayers, embedDim * 2, 0.1f, device);
        register_module("encoder", encoder);
        if (shareEncoder) {
            this.encoder2 = null;
        } else {
            this.encoder2 = new ProteinSeqEncoder(vocabSize, maxLen, embedDim, numHeads,
                    numLayers, embedDim * 2, 0.1f, device);
            register_module("encoder2", encoder2);
        }
        // [z1; z2; z1*z2; |z1-z2|] = 4 * embedDim
        this.head = new MLP(embedDim * 4L, new long[]{256L, 128L}, 1L, "relu", 0.1f,
                false, false, true, device);
        register_module("head", head);
    }

    /**
     * @param seqA [B, L]
     * @param seqB [B, L]
     * @return interaction probability [B]
     */
    public Tensor forward(Tensor seqA, Tensor seqB) {
        Tensor z1 = encoder.forward(seqA);
        Tensor z2 = (shareEncoder ? encoder : encoder2).forward(seqB);
        Tensor hadamard = z1.mul(z2);
        Tensor absDiff = z1.sub(z2).abs();
        TensorVector cat = new TensorVector();
        cat.push_back(z1);
        cat.push_back(z2);
        cat.push_back(hadamard);
        cat.push_back(absDiff);
        return head.forward(torch.cat(cat, 1L)).squeeze(1L).sigmoid();
    }

    public ProteinSeqEncoder encoder() {
        return encoder;
    }
}
