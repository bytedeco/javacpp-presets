/*
 * MolTrans — Molecular Interaction Transformer for drug-target binding.
 *
 * Reference:
 *   Huang et al., "MolTrans: Molecular Interaction Transformer for Drug Target
 *   Interaction Prediction", Bioinformatics 2021.
 *   https://doi.org/10.1093/bioinformatics/btaa880
 *   Code: https://github.com/kexinhuang12345/moltrans
 *
 * Architecture (core ideas preserved):
 *   1. Sub-structure / character embeddings for drug and protein
 *   2. Independent Transformer encoders
 *   3. Interaction map via outer product (or bilinear) of drug & protein tokens
 *   4. CNN over interaction map + FC → binding probability / affinity
 *
 * Industrial relevance: drug–target interaction (DTI) screening in pharma
 * virtual screening pipelines; complements DeepDTA (CNN) with explicit
 * interaction modeling.
 */
package org.bytedeco.pytorch.recommend.models.pharma;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.industry.MultiHeadSelfAttention;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MolTrans extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl drugEmbedding;
    private final EmbeddingImpl proteinEmbedding;
    private final List<MultiHeadSelfAttention> drugLayers = new ArrayList<>();
    private final List<MultiHeadSelfAttention> proteinLayers = new ArrayList<>();
    private final List<LayerNormImpl> drugNorms = new ArrayList<>();
    private final List<LayerNormImpl> proteinNorms = new ArrayList<>();
    private final Conv2dImpl interactionConv;
    private final MLP head;
    private final ReLUImpl relu;
    private final int embedDim;
    private final int maxDrugLen;
    private final int maxProteinLen;

    public MolTrans(int drugVocabSize, int proteinVocabSize, int maxDrugLen, int maxProteinLen) {
        this(drugVocabSize, proteinVocabSize, maxDrugLen, maxProteinLen,
                64, 4, 2, 0.1f, DeviceSupport.backend());
    }

    public MolTrans(int drugVocabSize, int proteinVocabSize, int maxDrugLen, int maxProteinLen,
                    int embedDim, int numHeads, int numLayers, float dropout, String device) {
        super("MolTrans");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.maxDrugLen = maxDrugLen;
        this.maxProteinLen = maxProteinLen;
        this.relu = new ReLUImpl();

        EmbeddingOptions dOpts = new EmbeddingOptions(Math.max(drugVocabSize, 2), embedDim);
        dOpts.padding_idx().put(new LongOptional(0L));
        this.drugEmbedding = new EmbeddingImpl(dOpts);
        register_module("drug_embedding", drugEmbedding);

        EmbeddingOptions pOpts = new EmbeddingOptions(Math.max(proteinVocabSize, 2), embedDim);
        pOpts.padding_idx().put(new LongOptional(0L));
        this.proteinEmbedding = new EmbeddingImpl(pOpts);
        register_module("protein_embedding", proteinEmbedding);

        for (int i = 0; i < numLayers; i++) {
            MultiHeadSelfAttention da = new MultiHeadSelfAttention(embedDim, numHeads, dropout, device);
            MultiHeadSelfAttention pa = new MultiHeadSelfAttention(embedDim, numHeads, dropout, device);
            LongVector ds = new LongVector(1); ds.put(0, embedDim);
            LongVector ps = new LongVector(1); ps.put(0, embedDim);
            LayerNormImpl dn = new LayerNormImpl(ds);
            LayerNormImpl pn = new LayerNormImpl(ps);
            register_module("drug_attn_" + i, da);
            register_module("protein_attn_" + i, pa);
            register_module("drug_norm_" + i, dn);
            register_module("protein_norm_" + i, pn);
            drugLayers.add(da);
            proteinLayers.add(pa);
            drugNorms.add(dn);
            proteinNorms.add(pn);
        }

        // Interaction map CNN: input is [B, 1, Ld, Lp] (or embedDim channels if multi-channel)
        // We use rank-1 interaction per embedding dim then mean, or single-channel outer of pooled.
        // Paper: interaction matrix I = drug @ protein^T  -> CNN
        LongPointer k = new LongPointer(new long[]{3L, 3L});
        Conv2dOptions copt = new Conv2dOptions(1, 16, k);
        copt.padding().put(new LongPointer(new long[]{1L, 1L}));
        this.interactionConv = new Conv2dImpl(copt);
        register_module("interaction_conv", interactionConv);

        // After conv+pool roughly 16 * (Ld/2) * (Lp/2) — use adaptive via flatten of max-pool
        // Head input = 16 (global max pool over spatial)
        this.head = new MLP(16L, new long[]{64L, 32L}, 1L, "relu", dropout, false, false, true, device);
        register_module("head", head);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            drugEmbedding.to(dev, false);
            proteinEmbedding.to(dev, false);
            for (LayerNormImpl n : drugNorms) n.to(dev, false);
            for (LayerNormImpl n : proteinNorms) n.to(dev, false);
        }
    }

    private Tensor encode(Tensor tokens, EmbeddingImpl emb,
                          List<MultiHeadSelfAttention> layers, List<LayerNormImpl> norms) {
        Tensor mask = tokens.ne(new Scalar(0L)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor h = emb.forward(tokens.toType(org.bytedeco.pytorch.global.torch.ScalarType.Long));
        for (int i = 0; i < layers.size(); i++) {
            h = h.add(layers.get(i).forward(norms.get(i).forward(h), mask));
        }
        return h; // [B, L, D]
    }

    /**
     * @param drugTokens    [B, Ld]
     * @param proteinTokens [B, Lp]
     * @return binding probability [B]
     */
    public Tensor forward(Tensor drugTokens, Tensor proteinTokens) {
        Tensor drugH = encode(drugTokens, drugEmbedding, drugLayers, drugNorms);       // [B, Ld, D]
        Tensor protH = encode(proteinTokens, proteinEmbedding, proteinLayers, proteinNorms); // [B, Lp, D]

        // Interaction map: mean over embed dim of outer products
        // I[b, i, j] = sum_d drugH[b,i,d] * protH[b,j,d]
        Tensor inter = torch.matmul(drugH, protH.transpose(1, 2)); // [B, Ld, Lp]
        Tensor interMap = inter.unsqueeze(1L); // [B, 1, Ld, Lp]

        Tensor convOut = relu.forward(interactionConv.forward(interMap)); // [B, 16, Ld, Lp]
        // Global max pool over spatial dims
        Tensor pooled = torch.amax(convOut, new long[]{2L, 3L}, false); // [B, 16]
        return head.forward(pooled).squeeze(1L).sigmoid();
    }

    /** Regression variant (no sigmoid) for continuous affinity labels. */
    public Tensor forwardAffinity(Tensor drugTokens, Tensor proteinTokens) {
        Tensor drugH = encode(drugTokens, drugEmbedding, drugLayers, drugNorms);
        Tensor protH = encode(proteinTokens, proteinEmbedding, proteinLayers, proteinNorms);
        Tensor inter = torch.matmul(drugH, protH.transpose(1, 2)).unsqueeze(1L);
        Tensor convOut = relu.forward(interactionConv.forward(inter));
        Tensor pooled = torch.amax(convOut, new long[]{2L, 3L}, false);
        return head.forward(pooled).squeeze(1L);
    }

    public int embedDim() {
        return embedDim;
    }
}
