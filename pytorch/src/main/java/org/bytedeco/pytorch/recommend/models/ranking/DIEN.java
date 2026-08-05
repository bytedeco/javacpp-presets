/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DIEN.scala
 *
 * Deep Interest Evolution Network (DIEN, AAAI'2019).
 * Reference: https://arxiv.org/pdf/1809.03672
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.GRUImpl;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.nn.options.GRUOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DIEN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<SequenceFeature> sequenceFeatures;
    private final long historyDim;
    private final EmbeddingLayer embedding;
//    private final List<GRUImpl> interestExtractorLayers = new ArrayList<>();
    private final List<RankingAUGRU> interestEvolvingLayers = new ArrayList<>();
    private final ModuleListImpl interestExtractorLayers = new ModuleListImpl();
//    private final ModuleListImpl interestEvolvingLayers = new ModuleListImpl();
    private final MLP mlp;

    public DIEN(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures) {
        this(features, sequenceFeatures, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public DIEN(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures,
                int embedDim, long[] mlpDims, float dropout, String device) {
        super("DIEN");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("DIEN: features cannot be empty");
        }
        if (sequenceFeatures == null || sequenceFeatures.isEmpty()) {
            throw new IllegalArgumentException("DIEN: sequenceFeatures cannot be empty");
        }
        this.sequenceFeatures = new ArrayList<>(sequenceFeatures);

        long sparseDim = Features.calcSparseDim(new ArrayList<>(features));
        long histDim = 0L;
        for (SequenceFeature sf : this.sequenceFeatures) {
            histDim += sf.embedDim();
        }
        this.historyDim = histDim;
        long totalDim = sparseDim + historyDim;

        List<Feature> allFeats = new ArrayList<>();
        allFeats.addAll(features);
        allFeats.addAll(this.sequenceFeatures);
        this.embedding = new EmbeddingLayer(allFeats, embedDim, device);
        register_module("embedding", embedding);

        for (int i = 0; i < this.sequenceFeatures.size(); i++) {
            SequenceFeature fea = this.sequenceFeatures.get(i);
            GRUOptions opts = new GRUOptions(fea.embedDim(), fea.embedDim());
            opts.batch_first().put(true);
            GRUImpl gru = new GRUImpl(opts);
            register_module("interest_extractor_" + i, gru);
            interestExtractorLayers.insert(i,gru);
        }

        for (int i = 0; i < this.sequenceFeatures.size(); i++) {
            RankingAUGRU augru = new RankingAUGRU(this.sequenceFeatures.get(i).embedDim(), device);
            register_module("interest_evolving_" + i, augru);
            interestEvolvingLayers.add(augru);
        }

        // Top MLP — activation fixed to "dice" as in the paper.
        this.mlp = new MLP(totalDim, mlpDims, 1L, "dice", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> seqFeats) {
        Tensor featEmb = embedding.forward(sparseFeats, Collections.emptyMap(), true);
        Tensor seqEmb = embedding.forwardSeqRaw(seqFeats);

        List<Tensor> interestOut = new ArrayList<>();
        int offset = 0;
        for (int i = 0; i < sequenceFeatures.size(); i++) {
            int dim = sequenceFeatures.get(i).embedDim();
            Tensor seq = seqEmb.narrow(2, offset, dim);
            offset += dim;
            Tensor mask = getMask(seqFeats, sequenceFeatures.get(i));

            // GRU returns (output [B,T,H], h_n [num_layers,B,H]).
            // Use get0() = full sequence output for AUGRU evolution (get1 was a Scala/C++ port bug).
            T_TensorTensor_T gruRet = interestExtractorLayers.get(i).forwardT_TensorTensor_T(seq);
            Tensor gruOut = gruRet.get0();

            long seqLen = seq.size(1);
            Tensor targetEmb = (gruOut.dim() == 3L) ? gruOut.select(1, seqLen - 1) : gruOut;

            Tensor[] augruOut =((RankingAUGRU)interestEvolvingLayers.get(i)).run(gruOut, targetEmb, mask);
            interestOut.add(augruOut[1].unsqueeze(1L)); // lastHidden
        }

        TensorVector iVec = new TensorVector();
        for (Tensor t : interestOut) iVec.push_back(t);
        Tensor historyOut = torch.cat(iVec, 1L);

        TensorVector mVec = new TensorVector();
        mVec.push_back(historyOut.view(-1L, historyDim));
        mVec.push_back(featEmb);
        Tensor mlpIn = torch.cat(mVec, 1L);
        return mlp.forward(mlpIn).squeeze(1L);
    }

    private Tensor getMask(Map<String, Tensor> seqFeats, SequenceFeature sf) {
        Tensor raw = seqFeats.get(sf.name());
        return raw.ne(new Scalar(0L)).toType(ScalarType.Float).squeeze(-1L);
    }
}
