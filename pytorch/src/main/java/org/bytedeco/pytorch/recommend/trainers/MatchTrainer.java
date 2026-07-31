/*
 * Ported from torch-rechub-scala: torchrec/trainers/MatchTrainer.scala
 *
 * Trainer for matching / retrieval models (DSSM, MAMBA, MIND, YoutubeDNN, DIEN, ...).
 * Uses in-batch negative sampling + BPR-style loss. Extends {@link Trainer}.
 */
package org.bytedeco.pytorch.recommend.trainers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.TensorHelpers;
import org.bytedeco.pytorch.recommend.basic.metrics.HitRate;
import org.bytedeco.pytorch.recommend.basic.metrics.MRR;
import org.bytedeco.pytorch.recommend.basic.metrics.NDCG;
import org.bytedeco.pytorch.recommend.data.Batch;
import org.bytedeco.pytorch.recommend.models.matching.DSSM;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MatchTrainer extends Trainer<MatchTrainer> {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Matching mode flag (reserved for future hard-neg / sampled-softmax paths). */
    private int mode;
    private float temperature;
    private int evalTopK;
    private BatchForward userForward;
    private BatchForward itemForward;

    public MatchTrainer(Module model) {
        super(model);
        this.mode = 0;
        this.temperature = 0.07f;
        this.evalTopK = 10;
        this.userForward = ModelForwards.matchUser(model, device);
        this.itemForward = ModelForwards.matchItem(model);
        maximizeMetric(true);
    }

    public MatchTrainer mode(int mode) {
        this.mode = mode;
        return this;
    }

    public MatchTrainer temperature(float temperature) {
        this.temperature = temperature;
        return this;
    }

    public MatchTrainer evalTopK(int k) {
        this.evalTopK = Math.max(1, k);
        return this;
    }

    public MatchTrainer withUserForward(BatchForward userForward) {
        this.userForward = userForward;
        return this;
    }

    public MatchTrainer withItemForward(BatchForward itemForward) {
        this.itemForward = itemForward;
        return this;
    }

    @Override
    public MatchTrainer device(String device) {
        super.device(device);
        // rebuild user forward so positions land on the right device
        this.userForward = ModelForwards.matchUser(model, this.device);
        return this;
    }

    @Override
    protected String primaryMetricName() {
        return "Hit@" + evalTopK;
    }

    @Override
    protected Tensor computeTrainLoss(Batch batch) {
        if (batch == null) {
            return null;
        }
        Map<String, Tensor> userFeats = batch.sparseFeatures;
        Map<String, Tensor> itemFeats = batch.itemFeatures;
        if (userFeats == null || userFeats.isEmpty()) {
            return null;
        }

        Tensor userEmb = userForward.forward(batch);
        if (userEmb == null) {
            return null;
        }

        Tensor itemEmb;
        if (model instanceof DSSM) {
            if (itemFeats == null || itemFeats.isEmpty()) {
                return null;
            }
            itemEmb = itemForward.forward(batch);
        } else {
            // MAMBA / MIND / DIEN path: use first item feature or zeros_like
            if (itemFeats != null && !itemFeats.isEmpty()) {
                itemEmb = itemFeats.values().iterator().next().toType(ScalarType.Float);
            } else {
                itemEmb = torch.zeros_like(userEmb);
            }
        }
        if (itemEmb == null) {
            return null;
        }

        long batchSize = userEmb.size(0);
        if (batchSize <= 1L) {
            // cannot form in-batch negatives
            return null;
        }

        return inBatchBprLoss(userEmb, itemEmb.toType(ScalarType.Float), (int) batchSize);
    }

    /**
     * Vectorized in-batch BPR: score matrix [B,B], positive = diagonal,
     * negative = max off-diagonal, loss = -log(sigmoid(pos - neg_max)).
     */
    private Tensor inBatchBprLoss(Tensor userEmb, Tensor itemEmb, int batchSize) {
        // For DSSM the Scala path used pairwise BPR over all (i,j!=i); vectorized max-neg is faster
        // and matches the MAMBA/MIND path. DSSM can still use the same formulation.
        Tensor allScores = userEmb.matmul(itemEmb.t()); // [B, B]
        if (temperature > 0f && temperature != 1f) {
            allScores = allScores.div(new Scalar(temperature));
        }

        Tensor batchIdx = arangeLong(batchSize, device);
        Tensor posScores = allScores.gather(1, batchIdx.view(batchSize, 1L)).squeeze(); // [B]

        Tensor diagOnes = torch.ones(new long[]{batchSize, batchSize},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        Tensor eyeMask = torch.eye(batchSize,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (device != null && !"cpu".equals(device)) {
            Device d = new Device(device);
            diagOnes = diagOnes.to(d, ScalarType.Float);
            eyeMask = eyeMask.to(d, ScalarType.Float);
        }
        Tensor mask = diagOnes.sub(eyeMask);
        Tensor negScores = allScores.add(mask.mul(new Scalar(-1e9f)));
        Tensor negScoresMax = torch.max(negScores, 1L).get0(); // [B]
        Tensor diff = posScores.sub(negScoresMax);
        return torch.log(torch.sigmoid(diff).add(new Scalar(1e-8f))).neg().mean();
    }

    @Override
    protected Tensor predictBatch(Batch batch) {
        // Matching scores are pairwise; single-batch predict returns user embedding.
        // Prefer evaluate() / inferenceEmbedding() for ranking metrics.
        return userForward.forward(batch);
    }

    @Override
    public Map<String, Float> evaluate(Iterable<Batch> dataLoader) {
        return evaluateFull(dataLoader, evalTopK);
    }

    /** Primary scalar used by early stopping — Hit@K. */
    public float evaluateHit(Iterable<Batch> dataLoader, int topk) {
        Map<String, Float> m = evaluateFull(dataLoader, topk);
        Float v = m.get("Hit@" + topk);
        return v != null ? v : 0.0f;
    }

    /**
     * In-batch evaluation: for each user i rank all items j in the batch;
     * positive item is the diagonal (j == i).
     */
    public Map<String, Float> evaluateFull(Iterable<Batch> dataLoader, int topk) {
        model.eval();
        HitRate hitRate = new HitRate(topk);
        NDCG ndcg = new NDCG(topk);
        MRR mrr = new MRR();

        for (Batch batch : dataLoader) {
            if (batch == null) continue;
            Map<String, Tensor> userFeats = batch.sparseFeatures;
            Map<String, Tensor> itemFeats = batch.itemFeatures;
            if (userFeats == null || userFeats.isEmpty()) continue;

            Tensor userEmb = userForward.forward(batch);
            if (userEmb == null) continue;

            Tensor itemEmb;
            if (model instanceof DSSM && itemFeats != null && !itemFeats.isEmpty()) {
                itemEmb = itemForward.forward(batch);
            } else if (itemFeats != null && !itemFeats.isEmpty()) {
                itemEmb = itemFeats.values().iterator().next().toType(ScalarType.Float);
            } else {
                itemEmb = torch.zeros_like(userEmb);
            }
            if (itemEmb == null) continue;

            int batchSize = (int) userEmb.size(0);
            if (batchSize < 2) continue;

            try {
                Tensor allScores = torch.matmul(userEmb, itemEmb.t());
                Tensor scoresHost = allScores.to(ScalarType.Float).contiguous().cpu();
                float[] scoreArr = TensorHelpers.toFloatArray(scoresHost);

                for (int i = 0; i < batchSize; i++) {
                    int rowOffset = i * batchSize;
                    float[] labelArr = new float[batchSize];
                    labelArr[i] = 1.0f;
                    float[] scoreSlice = new float[batchSize];
                    System.arraycopy(scoreArr, rowOffset, scoreSlice, 0, batchSize);
                    hitRate.update(scoreSlice, labelArr);
                    ndcg.update(scoreSlice, labelArr);
                    mrr.update(scoreSlice, labelArr);
                }
            } catch (Throwable ignored) {
                // skip bad batch
            }
        }

        model.train(true);
        Map<String, Float> out = new LinkedHashMap<>();
        out.put("Hit@" + topk, hitRate.compute());
        out.put("NDCG@" + topk, ndcg.compute());
        out.put("MRR", mrr.compute());
        return out;
    }

    /**
     * Extract user or item embeddings over a loader.
     *
     * @param mode {@code "user"} or {@code "item"}
     */
    public List<Tensor> inferenceEmbedding(Iterable<Batch> dataLoader, String mode) {
        model.eval();
        List<Tensor> embeddings = new ArrayList<>();
        String m = mode != null ? mode.toLowerCase() : "user";
        for (Batch batch : dataLoader) {
            if (batch == null) continue;
            try {
                if ("user".equals(m)) {
                    Tensor emb = userForward.forward(batch);
                    if (emb != null) embeddings.add(emb);
                } else if ("item".equals(m)) {
                    Tensor emb = itemForward.forward(batch);
                    if (emb != null) embeddings.add(emb);
                }
            } catch (Throwable ignored) {
                // skip
            }
        }
        model.train(true);
        return embeddings;
    }

    public Tensor[] inferenceEmbeddingArray(Iterable<Batch> dataLoader, String mode) {
        List<Tensor> list = inferenceEmbedding(dataLoader, mode);
        return list.toArray(new Tensor[0]);
    }
}
