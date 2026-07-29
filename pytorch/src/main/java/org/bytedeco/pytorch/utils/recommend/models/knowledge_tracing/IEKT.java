/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/IEKT.scala
 *
 * IEKT: Individual Estimation Knowledge Tracing.
 * Cognitive/acquisition policy → GRU state update → prediction.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.GRUCellImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class IEKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final float gamma;
    private final String device;
    private final Tensor conceptEmb;
    private final EmbeddingImpl qEmbed;
    private final Tensor cogMatrix;
    private final Tensor acqMatrix;
    private final MLP predictor;
    private final MLP cogSelector;
    private final MLP acqSelector;
    private final GRUCellImpl gruCell;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl outputLayer;

    public IEKT(long numConcepts) {
        this(numConcepts, 64, 10, 10, 1, 0.2f, 0.93f, DeviceSupport.backend());
    }

    public IEKT(
            long numConcepts,
            int embedDim,
            int numCogLevels,
            int numAcqLevels,
            int numLayers,
            float dropout,
            float gamma,
            String device) {
        super("IEKT");
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;
        this.gamma = gamma;
        this.device = device;

        Tensor cEmb = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        if (!"cpu".equals(device)) {
            cEmb = cEmb.to(new Device(device), ScalarType.Float);
        }
        this.conceptEmb = cEmb;
        register_parameter("concept_emb", conceptEmb);

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("q_embed", qEmbed);

        Tensor cog = torch.randn(
                new long[]{numCogLevels, embedDim * 2L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            cog = cog.to(new Device(device), ScalarType.Float);
        }
        this.cogMatrix = cog;
        register_parameter("cog_matrix", cogMatrix);

        Tensor acq = torch.randn(
                new long[]{numAcqLevels, embedDim * 2L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            acq = acq.to(new Device(device), ScalarType.Float);
        }
        this.acqMatrix = acq;
        register_parameter("acq_matrix", acqMatrix);

        // predictor input: hQConcat (2*embed) + rt.narrow(.,0,embed) => 3*embed
        this.predictor = new MLP(embedDim * 3L, new long[]{(long) embedDim}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("predictor", predictor);

        this.cogSelector = new MLP(embedDim * 4L, new long[]{(long) embedDim}, numCogLevels, "relu", dropout,
                false, false, true, device);
        register_module("cog_selector", cogSelector);

        this.acqSelector = new MLP(embedDim * 4L, new long[]{(long) embedDim}, numAcqLevels, "relu", dropout,
                false, false, true, device);
        register_module("acq_selector", acqSelector);

        // GRUCell input: q + cogEmb + acqEmb => 5*embedDim (Scala says 5* but cat is q+cog+acq = 1+2+2=5)
        this.gruCell = new GRUCellImpl(embedDim * 5L, embedDim);
        register_module("gru_cell", gruCell);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.outputLayer = new LinearImpl(embedDim, 1);
        register_module("output", outputLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qEmbed.to(dev, false);
            predictor.to(dev, false);
            cogSelector.to(dev, false);
            acqSelector.to(dev, false);
            gruCell.to(dev, false);
            outputLayer.to(dev, false);
        }
    }

    public float gamma() {
        return gamma;
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);
        Device dev = new Device(device);

        Tensor cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)));
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor cIdsLongDev = cIdsLong.to(dev, ScalarType.Long);
        Tensor cEmb = conceptEmb.index_select(0, cIdsLongDev.view(-1L)).view(batchSize, seqLen, embedDim);
        Tensor qEmb = qEmbed.forward(cIdsLongDev);

        Tensor h = torch.zeros(
                new long[]{batchSize, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            h = h.to(dev, ScalarType.Float);
        }

        Tensor rt = torch.zeros(
                new long[]{batchSize, embedDim * 2L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            rt = rt.to(dev, ScalarType.Float);
        }

        List<Tensor> predictions = new ArrayList<>();
        for (int i = 0; i < seqLen; i++) {
            Tensor q = qEmb.select(1, i);
            Tensor hQConcat = torch.cat(new TensorVector(h, q), 1);
            Tensor fullInput = torch.cat(new TensorVector(hQConcat, rt), 1);

            Tensor cogLogits = cogSelector.forward(fullInput);
            Tensor cogAction = cogLogits.argmax(new LongOptional(1L), true);

            Tensor acqLogits = acqSelector.forward(fullInput);
            Tensor acqAction = acqLogits.argmax(new LongOptional(1L), true);

            Tensor cogActionDev = cogAction.to(dev, ScalarType.Long);
            Tensor acqActionDev = acqAction.to(dev, ScalarType.Long);
            Tensor cogEmb = cogMatrix.index_select(0, cogActionDev.view(-1L)).view(batchSize, embedDim * 2L);
            Tensor acqEmb = acqMatrix.index_select(0, acqActionDev.view(-1L)).view(batchSize, embedDim * 2L);

            Tensor predInput = torch.cat(new TensorVector(hQConcat, rt.narrow(1, 0, embedDim)), 1);
            Tensor predLogit = predictor.forward(predInput);
            Tensor prob = torch.sigmoid(predLogit);
            predictions.add(prob.squeeze(1));

            Tensor r = rLong.select(1, i).toType(ScalarType.Float).unsqueeze(1);
            Tensor correctMask = r;
            Tensor incorrectMask = torch.ones_like(r).sub(r);

            Tensor qExpanded = torch.cat(new TensorVector(q, q), 1);
            Tensor rtCorrect = qExpanded.mul(correctMask);
            Tensor rtIncorrect = rt.narrow(1, 0, embedDim * 2L).mul(incorrectMask);
            rt = rtCorrect.add(rtIncorrect);

            Tensor gruInput = torch.cat(new TensorVector(q, cogEmb, acqEmb), 1);
            h = gruCell.forward(gruInput, h);
        }

        return torch.stack(new TensorVector(predictions.toArray(new Tensor[0])), 1);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
