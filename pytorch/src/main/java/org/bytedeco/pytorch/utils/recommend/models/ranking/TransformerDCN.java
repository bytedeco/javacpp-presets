/*
 * Ported from torchSa: torchrec/model/TransformerDCNModel.scala
 * and WWW2025_MMCTR Python Transformer_DCN (FuxiCTR-aligned).
 *
 * Transformer_DCN ranking model for MicroLens / MMCTR:
 *
 *   batchFeat  = likes_emb || views_emb                     // [B, 2*embDim]
 *   itemFeat   = item_id_emb || tags_pooled || emb_d128_proj // [B,S+1, itemInfoDim]
 *   targetEmb  = itemFeat[:, -1, :]                         // [B, itemInfoDim]
 *   seqEmb     = itemFeat[:, :-1, :]                        // [B, S, itemInfoDim]
 *   tfmrOut    = SequenceTransformer(target, seq, mask)     // [B, seqOutDim]
 *   dcnIn      = batchFeat || targetEmb || tfmrOut           // [B, dcnInDim]
 *   y          = MLP(concat(CrossNetV2(dcnIn), DNN(dcnIn))) // [B, 1] logits
 *
 * Loss: BCEWithLogitsLoss (Python binary_crossentropy with logits path).
 *
 * Cross path uses FuxiCTR CrossNetV2:
 *   X_{i+1} = X_i + X_0 * Linear_i(X_i)   (Linear has bias=true)
 * which differs slightly from basic.layers.CrossNetV2 (bias outside the mul).
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BCEWithLogitsLossImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.basic.layers.SequenceTransformer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.TransformerDCNFeatureEmbedding;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TransformerDCN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long embDim;
    private final long itemInfoDim;
    private final long dcnInDim;
    private final int dcnCrossLayers;
    private final long[] dcnHiddenUnits;
    private final long[] mlpHiddenUnits;

    private final TransformerDCNFeatureEmbedding featEmb;
    private final SequenceTransformer seqTransformer;
    private final LinearImpl[] crossLayers;
    private final MLP parallelDNN;
    private final MLP finalMLP;
    private final BCEWithLogitsLossImpl bceLoss;

    /**
     * Full constructor matching Python Transformer_DCN / Scala TransformerDCNModel.
     */
    public TransformerDCN(
            long itemVocabSize,
            long embDim,
            long pretrainDim,
            long embDimPretrain,
            long likesVocabSize,
            long viewsVocabSize,
            long tagsVocabSize,
            long numItems,
            int tagsLen,
            float[] pretrainedEmbFlat,
            long[] itemTagsFlat,
            long numHeads,
            int transformerLayers,
            double transformerDropout,
            long dimFeedforward,
            int firstKCols,
            boolean concatMaxPool,
            int dcnCrossLayers,
            long[] dcnHiddenUnits,
            long[] mlpHiddenUnits,
            double netDropout,
            String device) {
        super("TransformerDCN");
        this.embDim = embDim;
        this.dcnCrossLayers = dcnCrossLayers;
        this.dcnHiddenUnits = dcnHiddenUnits != null ? dcnHiddenUnits.clone() : new long[]{1024L, 512L, 256L};
        this.mlpHiddenUnits = mlpHiddenUnits != null ? mlpHiddenUnits.clone() : new long[]{64L, 32L};

        String dev = device != null ? device : DeviceSupport.backend();

        this.featEmb = new TransformerDCNFeatureEmbedding(
                itemVocabSize, embDim, likesVocabSize, viewsVocabSize, tagsVocabSize,
                pretrainDim, embDimPretrain, numItems, tagsLen,
                pretrainedEmbFlat, itemTagsFlat, dev);
        register_module("feat_emb", featEmb);
        this.itemInfoDim = featEmb.itemInfoDim();

        // Transformer input dim = 2 * itemInfoDim (concat target + seq features)
        long seqInDim = itemInfoDim * 2;
        this.seqTransformer = new SequenceTransformer(
                seqInDim, numHeads, dimFeedforward, transformerDropout,
                transformerLayers, firstKCols, concatMaxPool, dev);
        register_module("seq_transformer", seqTransformer);

        // dcn_in = batchFeat(2*emb) + target(itemInfoDim) + seqOut
        long batchFeatDim = embDim * 2;
        this.dcnInDim = batchFeatDim + itemInfoDim + seqTransformer.outputDim();

        // FuxiCTR CrossNetV2: Linear with bias, X += X0 * Linear(X)
        this.crossLayers = new LinearImpl[this.dcnCrossLayers];
        for (int i = 0; i < this.dcnCrossLayers; i++) {
            LinearImpl layer = new LinearImpl(
                    new LinearOptions(dcnInDim, dcnInDim).bias(true));
            register_module("cross_" + i, layer);
            crossLayers[i] = layer;
        }

        // parallel_dnn: hidden_units=[1024,512,256], output_dim=None
        // -> last layer is 256 with ReLU+Dropout (no final Linear to 1)
        this.parallelDNN = new MLP(
                dcnInDim,
                this.dcnHiddenUnits,
                /*outputDim*/ 1L,          // ignored when outputLayer=false
                "relu",
                (float) netDropout,
                /*useBatchNorm*/ false,
                /*useLayerNorm*/ false,
                /*outputLayer*/ false,
                dev);
        register_module("parallel_dnn", parallelDNN);

        long dnnOut = this.dcnHiddenUnits.length > 0
                ? this.dcnHiddenUnits[this.dcnHiddenUnits.length - 1]
                : dcnInDim;
        long dcnOutDim = dcnInDim + dnnOut; // concat [cross, dnn]

        // final mlp: hidden=[64,32], output_dim=1, no sigmoid (BCEWithLogits)
        this.finalMLP = new MLP(
                dcnOutDim,
                this.mlpHiddenUnits,
                1L,
                "relu",
                (float) netDropout,
                false,
                false,
                true,
                dev);
        register_module("final_mlp", finalMLP);

        this.bceLoss = new BCEWithLogitsLossImpl();
        register_module("bce", bceLoss);

        if (dev != null && !"cpu".equals(dev)) {
            this.to(new Device(dev), false);
        }
    }

    /** Python config defaults for MicroLens MMCTR. */
    public TransformerDCN(
            long itemVocabSize,
            long embDim,
            long pretrainDim,
            long embDimPretrain,
            long likesVocabSize,
            long viewsVocabSize,
            long tagsVocabSize,
            long numItems,
            int tagsLen,
            float[] pretrainedEmbFlat,
            long[] itemTagsFlat) {
        this(itemVocabSize, embDim, pretrainDim, embDimPretrain,
                likesVocabSize, viewsVocabSize, tagsVocabSize, numItems, tagsLen,
                pretrainedEmbFlat, itemTagsFlat,
                /*numHeads*/ 1L,
                /*transformerLayers*/ 2,
                /*transformerDropout*/ 0.2,
                /*dimFeedforward*/ 256L,
                /*firstKCols*/ 16,
                /*concatMaxPool*/ true,
                /*dcnCrossLayers*/ 3,
                new long[]{1024L, 512L, 256L},
                new long[]{64L, 32L},
                /*netDropout*/ 0.2,
                DeviceSupport.backend());
    }

    /**
     * Forward matching Scala / Python:
     *
     * @param history    item_seq   [B, S] long
     * @param target     item_id    [B] long
     * @param mask       pad mask   [B, S] float 1=valid
     * @param likesLevel likes      [B] long
     * @param viewsLevel views      [B] long
     * @return logits [B, 1]
     */
    public Tensor forward(
            Tensor history,
            Tensor target,
            Tensor mask,
            Tensor likesLevel,
            Tensor viewsLevel) {
        long batchSize = history.size(0);
        long maxSeqLen = history.size(1);

        // Batch-level features: likes || views  -> [B, 2*embDim]
        Tensor batchFeat = featEmb.encodeBatch(likesLevel, viewsLevel);

        // Full sequence = history || target  -> [B, S+1]
        Tensor target2D = target.reshape(batchSize, 1L);
        Tensor fullSeq = TensorHelpers.cat(new Tensor[]{history, target2D}, 1);

        // Item-level features for every position: [B, S+1, itemInfoDim]
        Tensor itemFeat = featEmb.encodeItems(fullSeq);

        // Target = last item, sequence = history
        Tensor targetEmb = itemFeat.narrow(1, maxSeqLen, 1).squeeze(1); // [B, itemInfoDim]
        Tensor seqEmb = itemFeat.narrow(1, 0, maxSeqLen);               // [B, S, itemInfoDim]

        // Transformer over history with target conditioning
        Tensor seqOut = seqTransformer.forward(targetEmb.unsqueeze(1), seqEmb, mask);

        // DCN input: batchFeat || targetEmb || seqOut
        Tensor dcnIn = TensorHelpers.cat(new Tensor[]{batchFeat, targetEmb, seqOut}, 1);

        Tensor crossOut = crossForward(dcnIn);          // [B, dcnInDim]
        Tensor dnnOut = parallelDNN.forward(dcnIn);     // [B, dcnHiddenUnits.last]
        Tensor finalIn = TensorHelpers.cat(new Tensor[]{crossOut, dnnOut}, 1);
        return finalMLP.forward(finalIn);               // [B, 1]
    }

    /** FuxiCTR CrossNetV2: X_{i+1} = X_i + X_0 * Linear_i(X_i). */
    private Tensor crossForward(Tensor x) {
        Tensor x0 = x;
        Tensor xl = x;
        for (int i = 0; i < crossLayers.length; i++) {
            Tensor projected = crossLayers[i].forward(xl);
            xl = xl.add(x0.mul(projected));
        }
        return xl;
    }

    /**
     * BCEWithLogits loss.
     *
     * @param label  [B] or [B,1] float 0/1
     * @param logits [B, 1] from {@link #forward}
     */
    public Tensor computeLoss(Tensor label, Tensor logits) {
        Tensor label2D = label.dim() == 1 ? label.reshape(-1L, 1L) : label;
        return bceLoss.forward(logits, label2D);
    }

    /** Print architecture summary (Python-aligned dims). */
    public void summary() {
        System.out.println("=== Transformer_DCN Model Summary (Python-aligned) ===");
        System.out.println("  embDim        : " + embDim);
        System.out.println("  itemInfoDim   : " + itemInfoDim + "  (item_id + tags + emb_d128)");
        System.out.println("  seqInDim      : " + (itemInfoDim * 2));
        System.out.println("  seqOutDim     : " + seqTransformer.outputDim());
        System.out.println("  DCN in dim    : " + dcnInDim);
        System.out.println("  DCN cross     : " + dcnCrossLayers + " layers");
        System.out.print("  Parallel DNN  : [");
        for (int i = 0; i < dcnHiddenUnits.length; i++) {
            if (i > 0) System.out.print(", ");
            System.out.print(dcnHiddenUnits[i]);
        }
        long dnnOut = dcnHiddenUnits.length > 0
                ? dcnHiddenUnits[dcnHiddenUnits.length - 1] : dcnInDim;
        System.out.println("]  out=" + dnnOut);
        System.out.print("  Final MLP     : [");
        for (int i = 0; i < mlpHiddenUnits.length; i++) {
            if (i > 0) System.out.print(", ");
            System.out.print(mlpHiddenUnits[i]);
        }
        System.out.println("] -> 1");
    }

    public long embDim() { return embDim; }
    public long itemInfoDim() { return itemInfoDim; }
    public long dcnInDim() { return dcnInDim; }
    public int dcnCrossLayers() { return dcnCrossLayers; }
    public long[] dcnHiddenUnits() { return dcnHiddenUnits.clone(); }
    public long[] mlpHiddenUnits() { return mlpHiddenUnits.clone(); }
    public TransformerDCNFeatureEmbedding featureEmbedding() { return featEmb; }
    public SequenceTransformer sequenceTransformer() { return seqTransformer; }
}
