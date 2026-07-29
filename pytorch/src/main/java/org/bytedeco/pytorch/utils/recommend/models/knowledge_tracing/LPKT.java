/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/LPKT.scala
 *
 * LPKT: Learning Persistence Knowledge Tracing.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LPKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final EmbeddingImpl exerciseEmb;
    private final EmbeddingImpl actionEmb;
    private final EmbeddingImpl itemEmb;
    private final LinearImpl fc1;
    private final LinearImpl fc2;
    private final LinearImpl fc3;
    private final LinearImpl fc4;
    private final LinearImpl predictor;

    public LPKT(long numExercises, long numConcepts) {
        this(numExercises, numConcepts, 1, 64, 64, 0.2f, DeviceSupport.backend());
    }

    public LPKT(
            long numExercises,
            long numConcepts,
            int numActionTypes,
            int embedDim,
            int exerciseDim,
            float dropout,
            String device) {
        super("LPKT");
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;

        this.exerciseEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 1, exerciseDim));
        register_module("exercise_emb", exerciseEmb);

        this.actionEmb = new EmbeddingImpl(new EmbeddingOptions(numActionTypes + 10L, embedDim));
        register_module("action_emb", actionEmb);

        this.itemEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 10, embedDim));
        register_module("item_emb", itemEmb);

        this.fc1 = new LinearImpl(embedDim * 3L, embedDim);
        this.fc2 = new LinearImpl(embedDim, embedDim);
        this.fc3 = new LinearImpl(embedDim * 3L, embedDim);
        this.fc4 = new LinearImpl(embedDim * 3L, embedDim);
        this.predictor = new LinearImpl(exerciseDim + embedDim, 1);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc4", fc4);
        register_module("predictor", predictor);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            this.to(dev, false);
        }
    }

    public Tensor forward(Tensor exerciseIds, Tensor actionTypes, Tensor knowledgeStates) {
        int batchSize = (int) exerciseIds.size(0);
        int seqLen = (int) exerciseIds.size(1);

        Tensor eEmbed = exerciseEmb.forward(exerciseIds).contiguous();
        Tensor atEmbed = actionEmb.forward(actionTypes).contiguous();
        Tensor itEmbed = itemEmb.forward(exerciseIds).contiguous();

        Tensor hPre = torch.zeros(
                new long[]{batchSize, numConcepts, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));

        List<Tensor> preds = new ArrayList<>();
        for (int t = 0; t < seqLen; t++) {
            Tensor e_t = eEmbed.select(1, t).contiguous();
            Tensor at_t = atEmbed.select(1, t).contiguous();
            Tensor it_t = itEmbed.select(1, t).contiguous();

            Tensor hPreAvg = hPre.mean(1).contiguous();

            Tensor combined1 = torch.cat(new TensorVector(hPreAvg, e_t, it_t), 1).contiguous();
            Tensor lc0 = torch.tanh(fc1.forward(combined1)).contiguous();

            Tensor combined2 = torch.cat(new TensorVector(lc0, e_t, it_t), 1).contiguous();
            Tensor gammaL = torch.sigmoid(fc3.forward(combined2)).contiguous();
            Tensor LG = gammaL.mul(torch.tanh(fc2.forward(lc0)).add(new Scalar(1.0)).div(new Scalar(2.0))).contiguous();

            Tensor itRepeat = it_t.unsqueeze(1).expand(batchSize, numConcepts, embedDim).clone().contiguous();
            Tensor hPreDet = hPre.detach();
            Tensor hPreFlat = hPreDet.mul(itRepeat).contiguous();
            Tensor lgRepeat = LG.unsqueeze(1).expand(batchSize, numConcepts, embedDim).clone().contiguous();
            Tensor hPreFlat2D = hPreFlat.view(-1L, embedDim);
            Tensor lgRepeat2D = lgRepeat.view(-1L, embedDim);
            Tensor itRepeat2D = itRepeat.view(-1L, embedDim);
            Tensor combined3 = torch.cat(new TensorVector(hPreFlat2D, lgRepeat2D, itRepeat2D), 1).contiguous();
            Tensor gammaF = torch.sigmoid(fc4.forward(combined3)).contiguous();
            Tensor gammaF3D = gammaF.view(batchSize, numConcepts, embedDim);
            Tensor hfTerm = gammaF3D.mul(hPre).contiguous();

            Tensor hTildeNew = LG.unsqueeze(1).mul(itRepeat).contiguous();
            hPre = hfTerm.add(hTildeNew);

            Tensor hFinal = hPre.mean(1).contiguous();
            Tensor predCombined = torch.cat(new TensorVector(e_t, hFinal), 1).contiguous();
            Tensor pred = torch.sigmoid(predictor.forward(predCombined));
            preds.add(pred.squeeze(1));
        }

        if (preds.isEmpty()) {
            return torch.zeros(
                    new long[]{batchSize, seqLen},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }
        return torch.stack(new TensorVector(preds.toArray(new Tensor[0])), 1);
    }

    public Tensor predict(Tensor exerciseIds, Tensor actionTypes, Tensor knowledgeStates) {
        return forward(exerciseIds, actionTypes, knowledgeStates);
    }
}
