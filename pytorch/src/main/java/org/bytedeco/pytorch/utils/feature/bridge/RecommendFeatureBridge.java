/*
 * Bridge Feature Platform → recommend.basic.features + FeatureStoreSnapshot.
 *
 * One-way adapter: utils.feature.core does NOT depend on recommend types.
 */
package org.bytedeco.pytorch.utils.feature.bridge;

import org.bytedeco.pytorch.utils.feature.core.FeatureDef;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.ValueType;
import org.bytedeco.pytorch.utils.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.utils.feature.serving.FeatureVector;
import org.bytedeco.pytorch.utils.recommend.basic.features.DenseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.modelops.FeatureStoreSnapshot;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Maps platform metadata / vectors into TorchRec-style feature objects. */
public final class RecommendFeatureBridge {

    private RecommendFeatureBridge() {}

    /**
     * Map a registry Field / FeatureDef to a recommend Feature.
     * Tags {@code vocab_size}, {@code embed_dim}, or FeatureDef vocab/embedDim drive dims.
     */
    public static Feature toRecommendFeature(Field field) {
        Objects.requireNonNull(field, "field");
        return toRecommendFeature(FeatureDef.fromField(field));
    }

    public static Feature toRecommendFeature(FeatureDef def) {
        Objects.requireNonNull(def, "def");
        String name = def.name();
        long vocab = def.vocabSize() > 0 ? def.vocabSize() : parseLongTag(def.tags(), "vocab_size", 10000L);
        int embedDim = def.embedDim() > 0 ? def.embedDim() : (int) parseLongTag(def.tags(), "embed_dim", 8L);
        int embDim = def.embeddingDim() > 0 ? def.embeddingDim() : embedDim;

        switch (def.valueType()) {
            case INT64_LIST:
            case INT32_LIST:
            case STRING_LIST:
                return Features.sequence(name, vocab, embedDim, "mean");
            case EMBEDDING:
            case FLOAT32:
            case FLOAT64:
            case FLOAT32_LIST:
            case FLOAT64_LIST:
                return Features.dense(name, Math.max(1, embDim > 0 ? embDim : 1));
            case INT32:
            case INT64:
            case BOOL:
            case STRING:
            case UNIX_TIMESTAMP:
            default:
                // id-like → sparse; pure continuous already handled above
                if (def.valueType().isFloating()) {
                    return Features.dense(name, 1);
                }
                return Features.sparse(name, vocab, embedDim);
        }
    }

    public static List<Feature> toRecommendFeatures(FeatureView view) {
        Objects.requireNonNull(view, "view");
        List<Feature> out = new ArrayList<>();
        for (Field f : view.schema()) {
            out.add(toRecommendFeature(f));
        }
        return out;
    }

    public static List<Feature> toRecommendFeatures(List<FeatureDef> defs) {
        List<Feature> out = new ArrayList<>();
        if (defs == null) return out;
        for (FeatureDef d : defs) out.add(toRecommendFeature(d));
        return out;
    }

    /** Classify FeatureVector entries into Sparse/Dense/Sequence lists by value shape. */
    public static List<Feature> inferRecommendFeatures(FeatureVector vector) {
        Objects.requireNonNull(vector, "vector");
        List<Feature> out = new ArrayList<>();
        for (String k : vector.sparse().keySet()) {
            out.add(Features.sparse(k, 100000L, 8));
        }
        for (Map.Entry<String, Double> e : vector.dense().entrySet()) {
            if (!vector.sparse().containsKey(e.getKey())) {
                out.add(Features.dense(e.getKey(), 1));
            }
        }
        for (String k : vector.sequences().keySet()) {
            out.add(Features.sequence(k, 100000L, 8, "mean"));
        }
        for (Map.Entry<String, float[]> e : vector.embeddings().entrySet()) {
            int dim = e.getValue() != null ? e.getValue().length : 1;
            out.add(Features.dense(e.getKey(), Math.max(1, dim)));
        }
        return out;
    }

    /**
     * Build a FeatureStoreSnapshot from a served FeatureVector for train/serve skew audit.
     */
    public static FeatureStoreSnapshot toSnapshot(FeatureVector vector, String snapshotId, String schemaVersion) {
        Objects.requireNonNull(vector, "vector");
        String sid = snapshotId != null ? snapshotId : "fs-" + System.currentTimeMillis();
        FeatureStoreSnapshot.Builder b = FeatureStoreSnapshot.builder(sid)
                .schemaVersion(schemaVersion != null ? schemaVersion : "v1");

        Object user = vector.entities().get("user_id");
        if (user != null) b.userId(String.valueOf(user));

        String eventTs = vector.meta().get("event_timestamp");
        if (eventTs != null) {
            try {
                b.eventTimeMs(Long.parseLong(eventTs));
            } catch (NumberFormatException ignored) {
            }
        }

        for (Map.Entry<String, Double> e : vector.dense().entrySet()) {
            b.dense(e.getKey(), e.getValue());
        }
        for (Map.Entry<String, Long> e : vector.sparse().entrySet()) {
            b.sparse(e.getKey(), e.getValue());
        }
        for (Map.Entry<String, long[]> e : vector.sequences().entrySet()) {
            b.sequence(e.getKey(), e.getValue());
        }
        for (Map.Entry<String, String> e : vector.meta().entrySet()) {
            b.meta(e.getKey(), e.getValue());
        }
        return b.build();
    }

    public static FeatureStoreSnapshot toSnapshot(FeatureResponse response, int index) {
        Objects.requireNonNull(response, "response");
        FeatureVector v = response.vectors().isEmpty()
                ? FeatureVector.builder().build()
                : response.vectors().get(Math.min(index, response.vectors().size() - 1));
        return toSnapshot(v, response.featureService() + "-" + index, response.meta().getOrDefault("schema", "v1"));
    }

    /**
     * Identity skew check: snapshot built from vector should have empty dense skew vs itself.
     */
    public static Map<String, Double> identitySkew(FeatureVector vector) {
        FeatureStoreSnapshot a = toSnapshot(vector, "a", "v1");
        FeatureStoreSnapshot b = toSnapshot(vector, "b", "v1");
        return a.denseSkewAgainst(b, 1e-9);
    }

    /** Flatten FeatureVector raw map (for DataFrame / logging). */
    public static Map<String, Object> toRawMap(FeatureVector vector) {
        Objects.requireNonNull(vector, "vector");
        Map<String, Object> out = new LinkedHashMap<>();
        out.putAll(vector.entities());
        out.putAll(vector.raw());
        return out;
    }

    public static boolean isSparse(Feature f) {
        return f instanceof SparseFeature;
    }

    public static boolean isDense(Feature f) {
        return f instanceof DenseFeature;
    }

    public static boolean isSequence(Feature f) {
        return f instanceof SequenceFeature;
    }

    private static long parseLongTag(Map<String, String> tags, String key, long dflt) {
        if (tags == null) return dflt;
        String v = tags.get(key);
        if (v == null || v.isEmpty()) return dflt;
        try {
            return Long.parseLong(v);
        } catch (NumberFormatException e) {
            return dflt;
        }
    }
}
