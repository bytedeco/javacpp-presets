/*
 * DrugBAN — Bilinear Attention Network for drug-target interaction.
 *
 * Reference:
 *   Bai et al., "Interpretable bilinear attention network improves
 *   drug-target interaction prediction accuracy and drug repositioning",
 *   Nature Machine Intelligence / related DTI work; BAN bilinear attention
 *   is also popularized by Kim et al. (visual QA) and adapted to DTI in:
 *   "DrugBAN: Drug-target interaction prediction by bilinear attention
 *   network" style implementations (see https://github.com/peizhenbai/DrugBAN).
 *
 * Core idea:
 *   Encode drug (graph/sequence) and protein (sequence) separately,
 *   then compute bilinear attention affinity matrix between every drug
 *   fragment and protein residue; attend both ways; fuse for classification.
 *
 * This library version uses sequence encoders (CNN/Transformer-lite) for both
 * modalities so it runs without a full molecular graph stack. Pair with
 * geometric GNN modules elsewhere in the project when graph drug features
 * are available.
 *
 * Input:  drugTokens [B, Ld], proteinTokens [B, Lp]
 * Output: binding probability [B]
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
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DrugBAN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl drugEmbedding;
    private final EmbeddingImpl proteinEmbedding;
    private final Conv1dImpl drugConv;
    private final Conv1dImpl proteinConv;
    private final LinearImpl bilinear; // maps drug_dim -> protein_dim for bilinear score
    private final LinearImpl drugAttnProj;
    private final LinearImpl proteinAttnProj;
    private final MLP head;
    private final ReLUImpl relu;
    private final int hiddenDim;

    public DrugBAN(int drugVocabSize, int proteinVocabSize) {
        this(drugVocabSize, proteinVocabSize, 128, 64, 3, DeviceSupport.backend());
    }

    public DrugBAN(int drugVocabSize, int proteinVocabSize, int embedDim, int hiddenDim,
                   int kernelSize, String device) {
        super("DrugBAN");
        this.hiddenDim = hiddenDim;
        this.relu = new ReLUImpl();

        EmbeddingOptions dOpts = new EmbeddingOptions(Math.max(drugVocabSize, 2), embedDim);
        dOpts.padding_idx().put(new LongOptional(0L));
        this.drugEmbedding = new EmbeddingImpl(dOpts);
        register_module("drug_embedding", drugEmbedding);

        EmbeddingOptions pOpts = new EmbeddingOptions(Math.max(proteinVocabSize, 2), embedDim);
        pOpts.padding_idx().put(new LongOptional(0L));
        this.proteinEmbedding = new EmbeddingImpl(pOpts);
        register_module("protein_embedding", proteinEmbedding);

        LongPointer k = new LongPointer(new long[]{kernelSize});
        Conv1dOptions dco = new Conv1dOptions(embedDim, hiddenDim, k);
        dco.padding().put(new LongPointer(new long[]{Math.max(kernelSize / 2, 0)}));
        this.drugConv = new Conv1dImpl(dco);
        register_module("drug_conv", drugConv);

        LongPointer pk = new LongPointer(new long[]{kernelSize});
        Conv1dOptions pco = new Conv1dOptions(embedDim, hiddenDim, pk);
        pco.padding().put(new LongPointer(new long[]{Math.max(kernelSize / 2, 0)}));
        this.proteinConv = new Conv1dImpl(pco);
        register_module("protein_conv", proteinConv);

        // Bilinear: score_ij = drug_i^T W protein_j  <=> (drug_i W) · protein_j
        this.bilinear = new LinearImpl(new LinearOptions(hiddenDim, hiddenDim).bias(false));
        this.drugAttnProj = new LinearImpl(hiddenDim, hiddenDim);
        this.proteinAttnProj = new LinearImpl(hiddenDim, hiddenDim);
        register_module("bilinear", bilinear);
        register_module("drug_attn_proj", drugAttnProj);
        register_module("protein_attn_proj", proteinAttnProj);

        // Fused: attended drug + attended protein + Hadamard
        this.head = new MLP(hiddenDim * 3L, new long[]{256L, 128L}, 1L, "relu", 0.1f,
                false, false, true, device);
        register_module("head", head);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            drugEmbedding.to(dev, false);
            proteinEmbedding.to(dev, false);
            bilinear.to(dev, false);
            drugAttnProj.to(dev, false);
            proteinAttnProj.to(dev, false);
        }
    }

    /**
     * @return binding probability [B]
     */
    public Tensor forward(Tensor drugTokens, Tensor proteinTokens) {
        Tensor dMask = drugTokens.ne(new Scalar(0L)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor pMask = proteinTokens.ne(new Scalar(0L)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);

        Tensor d = drugEmbedding.forward(drugTokens.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, Ld, E]
        Tensor p = proteinEmbedding.forward(proteinTokens.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, Lp, E]

        Tensor dH = relu.forward(drugConv.forward(d.transpose(1, 2))).transpose(1, 2); // [B, Ld, H]
        Tensor pH = relu.forward(proteinConv.forward(p.transpose(1, 2))).transpose(1, 2); // [B, Lp, H]

        // Bilinear attention affinity: [B, Ld, Lp]
        Tensor dW = bilinear.forward(dH);                         // [B, Ld, H]
        Tensor affinity = torch.matmul(dW, pH.transpose(1, 2));   // [B, Ld, Lp]

        // Mask padded positions
        Tensor dM = dMask.unsqueeze(2L); // [B, Ld, 1]
        Tensor pM = pMask.unsqueeze(1L); // [B, 1, Lp]
        Tensor neg = torch.full_like(affinity, new org.bytedeco.pytorch.Scalar(-1e9f));
        Tensor valid = dM.mul(pM);
        affinity = affinity.mul(valid).add(neg.mul(torch.sub(torch.ones_like(valid), valid)));

        // Attend protein from drug, and drug from protein
        Tensor attnDrugToProt = affinity.softmax(2L); // [B, Ld, Lp]
        Tensor attnProtToDrug = affinity.transpose(1, 2).softmax(2L); // [B, Lp, Ld]

        // Cross-attended sequences (lengths match the *query* side)
        Tensor drugFromProt = torch.matmul(attnDrugToProt, pH); // [B, Ld, H] protein ctx per drug token
        Tensor protFromDrug = torch.matmul(attnProtToDrug, dH); // [B, Lp, H] drug ctx per protein token

        // Residual on matching lengths, then masked mean pool
        Tensor drugPool = maskedMean(dH.add(drugFromProt), dMask);
        Tensor protPool = maskedMean(pH.add(protFromDrug), pMask);
        Tensor hadamard = drugPool.mul(protPool);

        TensorVector cat = new TensorVector();
        cat.push_back(drugPool);
        cat.push_back(protPool);
        cat.push_back(hadamard);
        return head.forward(torch.cat(cat, 1L)).squeeze(1L).sigmoid();
    }

    private static Tensor maskedMean(Tensor seq, Tensor mask) {
        // seq [B, L, H], mask [B, L]
        Tensor m = mask.unsqueeze(2L);
        Tensor summed = seq.mul(m).sum(1L);
        Tensor denom = mask.sum(1L).unsqueeze(1L).clamp_min(new Scalar(1.0f));
        return summed.div(denom);
    }

    public int hiddenDim() {
        return hiddenDim;
    }
}
