/*
 * DeepDTA — Deep Drug-Target Binding Affinity prediction.
 *
 * Reference:
 *   Öztürk et al., "DeepDTA: deep drug-target binding affinity prediction",
 *   Bioinformatics 2018. https://doi.org/10.1093/bioinformatics/bty593
 *   Code: https://github.com/hkmztrk/DeepDTA
 *
 * Architecture (paper-faithful):
 *   Drug SMILES character sequence  -> CNN (multi-filter) -> max-pool -> FC
 *   Protein amino-acid sequence     -> CNN (multi-filter) -> max-pool -> FC
 *   Concatenate drug & protein representations -> FC layers -> affinity score
 *
 * Affinity is typically pKd / pKi / pIC50 regression (MSE loss).
 * Widely used baseline in computational drug discovery and pharma AI pipelines.
 *
 * Input:
 *   drugTokens    [B, Ld] long  (SMILES char ids, 0=pad)
 *   proteinTokens [B, Lp] long  (amino-acid ids, 0=pad)
 * Output:
 *   affinity [B]  (regression; no sigmoid)
 */
package org.bytedeco.pytorch.utils.recommend.models.pharma;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DeepDTA extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl drugEmbedding;
    private final EmbeddingImpl proteinEmbedding;
    private final List<Conv1dImpl> drugConvs = new ArrayList<>();
    private final List<Conv1dImpl> proteinConvs = new ArrayList<>();
    private final LinearImpl drugFc;
    private final LinearImpl proteinFc;
    private final MLP affinityHead;
    private final ReLUImpl relu;
    private final int numFilters;
    private final int fcDim;

    public DeepDTA(int drugVocabSize, int proteinVocabSize) {
        this(drugVocabSize, proteinVocabSize, 128, 32, new int[]{4, 6, 8},
                256, new long[]{1024L, 1024L, 512L}, DeviceSupport.backend());
    }

    /**
     * @param drugVocabSize     SMILES character vocab (paper ~64 incl. pad)
     * @param proteinVocabSize  amino-acid vocab (paper ~25 incl. pad)
     * @param embedDim          char embedding dim
     * @param numFilters        CNN filters per kernel
     * @param kernelSizes       CNN kernel widths (paper uses 4,6,8 stacked as sequential CNN;
     *                          we use parallel multi-scale then concat, a common robust variant)
     * @param fcDim             drug/protein representation dim before concat
     */
    public DeepDTA(int drugVocabSize, int proteinVocabSize, int embedDim, int numFilters,
                   int[] kernelSizes, int fcDim, long[] headHidden, String device) {
        super("DeepDTA");
        this.numFilters = numFilters;
        this.fcDim = fcDim;
        this.relu = new ReLUImpl();
        int[] kernels = kernelSizes != null ? kernelSizes : new int[]{4, 6, 8};

        EmbeddingOptions dOpts = new EmbeddingOptions(Math.max(drugVocabSize, 2), embedDim);
        dOpts.padding_idx().put(new LongOptional(0L));
        this.drugEmbedding = new EmbeddingImpl(dOpts);
        register_module("drug_embedding", drugEmbedding);

        EmbeddingOptions pOpts = new EmbeddingOptions(Math.max(proteinVocabSize, 2), embedDim);
        pOpts.padding_idx().put(new LongOptional(0L));
        this.proteinEmbedding = new EmbeddingImpl(pOpts);
        register_module("protein_embedding", proteinEmbedding);

        for (int i = 0; i < kernels.length; i++) {
            int k = kernels[i];
            LongPointer dk = new LongPointer(new long[]{k});
            Conv1dOptions dco = new Conv1dOptions(embedDim, numFilters, dk);
            dco.padding().put(new LongPointer(new long[]{Math.max(k / 2, 0)}));
            Conv1dImpl dc = new Conv1dImpl(dco);
            register_module("drug_conv_" + i, dc);
            drugConvs.add(dc);

            LongPointer pk = new LongPointer(new long[]{k});
            Conv1dOptions pco = new Conv1dOptions(embedDim, numFilters, pk);
            pco.padding().put(new LongPointer(new long[]{Math.max(k / 2, 0)}));
            Conv1dImpl pc = new Conv1dImpl(pco);
            register_module("protein_conv_" + i, pc);
            proteinConvs.add(pc);
        }

        int cnnOut = numFilters * kernels.length;
        this.drugFc = new LinearImpl(cnnOut, fcDim);
        this.proteinFc = new LinearImpl(cnnOut, fcDim);
        register_module("drug_fc", drugFc);
        register_module("protein_fc", proteinFc);

        this.affinityHead = new MLP(fcDim * 2L, headHidden, 1L, "relu", 0.1f,
                false, false, true, device);
        register_module("affinity_head", affinityHead);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            drugEmbedding.to(dev, false);
            proteinEmbedding.to(dev, false);
            drugFc.to(dev, false);
            proteinFc.to(dev, false);
        }
    }

    private Tensor encodeSequence(Tensor tokenIds, EmbeddingImpl emb, List<Conv1dImpl> convs,
                                  LinearImpl fc) {
        Tensor x = emb.forward(tokenIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, L, E]
        Tensor xt = x.transpose(1, 2); // [B, E, L]
        List<Tensor> pooled = new ArrayList<>();
        for (Conv1dImpl conv : convs) {
            Tensor h = relu.forward(conv.forward(xt)); // [B, F, L']
            pooled.add(torch.max(h, 2L).get0());       // [B, F]
        }
        TensorVector vec = new TensorVector();
        for (Tensor p : pooled) vec.push_back(p);
        return relu.forward(fc.forward(torch.cat(vec, 1L))); // [B, fcDim]
    }

    /**
     * @return predicted binding affinity [B] (regression)
     */
    public Tensor forward(Tensor drugTokens, Tensor proteinTokens) {
        Tensor drugRep = encodeSequence(drugTokens, drugEmbedding, drugConvs, drugFc);
        Tensor protRep = encodeSequence(proteinTokens, proteinEmbedding, proteinConvs, proteinFc);
        TensorVector cat = new TensorVector();
        cat.push_back(drugRep);
        cat.push_back(protRep);
        return affinityHead.forward(torch.cat(cat, 1L)).squeeze(1L);
    }

    /** MSE affinity loss helper. */
    public static Tensor mseLoss(Tensor pred, Tensor target) {
        return pred.sub(target).pow(new Scalar(2.0f)).mean();
    }

    public int fcDim() {
        return fcDim;
    }
}
