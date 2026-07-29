/*
 * PEPNet — Personalized Embedding & Parameter Personalized Network.
 *
 * Reference (industrial, Kuaishou):
 *   Chang et al., "PEPNet: Parameter and Embedding Personalized Network for
 *   Infusing with Personalized Prior Information", KDD 2023 / CIKM industrial.
 *   Widely deployed for multi-scenario / multi-task short-video ranking.
 *
 * Two gates (paper):
 *   EPNet (Embedding Personalized Network):
 *     gate over concatenated embeddings using user/scenario prior → personalize embeddings
 *   PPNet (Parameter Personalized Network):
 *     gate over MLP hidden using prior → personalize network parameters (soft)
 *
 * This implementation:
 *   prior = [user_side_emb ; scenario_emb]
 *   emb'  = GateFusion(emb, prior)           // EPNet
 *   h     = MLP(emb')
 *   h'    = GateFusion(h, prior)             // PPNet
 *   multi-task towers on h'
 */
package org.bytedeco.pytorch.utils.recommend.models.shortvideo;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.GateFusion;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PEPNet extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer featureEmbedding;
    private final EmbeddingImpl scenarioEmbedding;
    private final GateFusion epNet;  // embedding personalization
    private final MLP sharedBottom;
    private final GateFusion ppNet;  // parameter personalization on hidden
    private final List<MLP> taskTowers = new ArrayList<>();
    private final int numTasks;
    private final int priorDim;
    private final int sharedDim;

    public PEPNet(List<? extends Feature> features, int numScenarios, int numTasks) {
        this(features, numScenarios, numTasks, 16, new long[]{256L, 128L},
                new long[]{64L, 32L}, DeviceSupport.backend());
    }

    public PEPNet(List<? extends Feature> features, int numScenarios, int numTasks,
                  int scenarioEmbedDim, long[] sharedHidden, long[] towerHidden, String device) {
        super("PEPNet");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("PEPNet: features cannot be empty");
        }
        if (numTasks < 1) {
            throw new IllegalArgumentException("numTasks must be >= 1");
        }
        this.numTasks = numTasks;

        List<Feature> featList = new ArrayList<>(features);
        int featDim = 0;
        for (Feature f : featList) featDim += f.embedDim();

        this.featureEmbedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("feature_embedding", featureEmbedding);

        EmbeddingOptions sOpts = new EmbeddingOptions(Math.max(numScenarios, 1), scenarioEmbedDim);
        sOpts.padding_idx().put(new LongOptional(0L));
        this.scenarioEmbedding = new EmbeddingImpl(sOpts);
        register_module("scenario_embedding", scenarioEmbedding);

        // prior = scenario emb only by default (user side can be injected via features)
        this.priorDim = scenarioEmbedDim;
        this.epNet = new GateFusion(featDim, priorDim, 64, GateFusion.Mode.MULTIPLICATIVE, device);
        register_module("epnet", epNet);

        this.sharedBottom = new MLP(featDim, sharedHidden, sharedHidden[sharedHidden.length - 1],
                "relu", 0.1f, false, false, true, device);
        register_module("shared_bottom", sharedBottom);
        this.sharedDim = (int) sharedHidden[sharedHidden.length - 1];

        this.ppNet = new GateFusion(sharedDim, priorDim, 64, GateFusion.Mode.RESIDUAL_SCALE, device);
        register_module("ppnet", ppNet);

        for (int t = 0; t < numTasks; t++) {
            MLP tower = new MLP(sharedDim, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
            register_module("task_tower_" + t, tower);
            taskTowers.add(tower);
        }

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            scenarioEmbedding.to(dev, false);
        }
    }

    /**
     * @param features    sparse/dense feature map
     * @param scenarioIds [B] long scenario / domain id
     * @return [B, numTasks] task probabilities (sigmoid)
     */
    public Tensor forward(Map<String, Tensor> features, Tensor scenarioIds) {
        Tensor emb = featureEmbedding.forward(features, Collections.emptyMap(), true);
        Tensor prior = scenarioEmbedding.forward(scenarioIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long));

        Tensor embP = epNet.forward(emb, prior);
        Tensor h = sharedBottom.forward(embP);
        Tensor hP = ppNet.forward(h, prior);

        TensorVector outs = new TensorVector();
        for (MLP tower : taskTowers) {
            outs.push_back(tower.forward(hP).sigmoid());
        }
        return torch.cat(outs, 1L);
    }

    /** Single-task convenience: task 0 probability [B]. */
    public Tensor forwardTask0(Map<String, Tensor> features, Tensor scenarioIds) {
        return forward(features, scenarioIds).select(1L, 0L);
    }

    public int numTasks() {
        return numTasks;
    }
}
