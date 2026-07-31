/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/EmbeddingLayer.scala
 *
 * Embedding layer for sparse and sequence features.
 * Uses EmbeddingImpl tables registered as submodules (ModuleDict-style).
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Embedding layer for sparse and sequence features.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class EmbeddingLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> features;
    private final int embedDim;
    private final String device;
    private final Long paddingIdx;
    private final boolean sparse;

    /** Direct access map (mirrors registered embedding tables). LinkedHashMap preserves insertion order. */
    private final Map<String, EmbeddingImpl> embeddingTables = new LinkedHashMap<>();
    private final Set<String> warnedMissingSparse = new HashSet<>();
    private final Set<String> warnedMissingSeq = new HashSet<>();
    private final Set<String> warnedInvalidIndices = new HashSet<>();

    public EmbeddingLayer(List<? extends Feature> features) {
        this(features, 8, DeviceSupport.backend(), null, false);
    }

    public EmbeddingLayer(List<? extends Feature> features, int embedDim) {
        this(features, embedDim, DeviceSupport.backend(), null, false);
    }

    public EmbeddingLayer(List<? extends Feature> features, int embedDim, String device) {
        this(features, embedDim, device, null, false);
    }

    public EmbeddingLayer(List<? extends Feature> features, int embedDim, String device,
                          Long paddingIdx, boolean sparse) {
        super("EmbeddingLayer");
        this.features = new ArrayList<>(features);
        this.embedDim = embedDim;
        this.device = device != null ? device : DeviceSupport.backend();
        this.paddingIdx = paddingIdx;
        this.sparse = sparse;

        for (Feature f : this.features) {
            if (f instanceof SparseFeature) {
                SparseFeature sf = (SparseFeature) f;
                String key = "embed_" + baseName(sf.name());
                boolean sharedExists = sf.sharedWith() != null && embeddingTables.containsKey(sf.sharedWith());
                if (!sharedExists && !embeddingTables.containsKey(key)) {
                    EmbeddingOptions options = new EmbeddingOptions(sf.vocabSize(), sf.embedDim());
                    if (sf.paddingIdx() != null) {
                        options.padding_idx().put(new LongOptional(sf.paddingIdx()));
                    }
                    EmbeddingImpl embedding = new EmbeddingImpl(options);
                    if (!"cpu".equals(this.device)) {
                        embedding.to(new Device(this.device), false);
                    }
                    register_module(sanitizeModuleName(key), embedding);
                    embeddingTables.put(key, embedding);
                }
            } else if (f instanceof SequenceFeature) {
                SequenceFeature seqf = (SequenceFeature) f;
                String key = "embed_seq_" + baseName(seqf.name());
                boolean sharedExists = seqf.sharedWith() != null && embeddingTables.containsKey(seqf.sharedWith());
                if (!sharedExists && !embeddingTables.containsKey(key)) {
                    EmbeddingOptions options = new EmbeddingOptions(seqf.vocabSize(), seqf.embedDim());
                    if (seqf.paddingIdx() != 0) {
                        options.padding_idx().put(new LongOptional(seqf.paddingIdx()));
                    }
                    EmbeddingImpl embedding = new EmbeddingImpl(options);
                    if (!"cpu".equals(this.device)) {
                        embedding.to(new Device(this.device), false);
                    }
                    register_module(sanitizeModuleName(key), embedding);
                    embeddingTables.put(key, embedding);
                }
            }
            // DenseFeature — no embedding table
        }
    }

    public List<Feature> features() {
        return Collections.unmodifiableList(features);
    }

    public int embedDim() {
        return embedDim;
    }

    public String deviceName() {
        return device;
    }

    public Long paddingIdx() {
        return paddingIdx;
    }

    public boolean sparse() {
        return sparse;
    }

    public Map<String, EmbeddingImpl> embeddingTables() {
        return Collections.unmodifiableMap(embeddingTables);
    }

    private static String sanitizeModuleName(String name) {
        String cleaned = name.replaceAll("[^A-Za-z0-9_]+", "_");
        while (cleaned.startsWith("_")) {
            cleaned = cleaned.substring(1);
        }
        while (cleaned.endsWith("_")) {
            cleaned = cleaned.substring(0, cleaned.length() - 1);
        }
        return cleaned;
    }

    /**
     * Canonical base name by stripping common prefixes like seq_, embed_, feat_
     * to avoid double-prefixing when building table keys.
     */
    private static String baseName(String name) {
        if (name == null) {
            return "";
        }
        String withoutPrefix = name.replaceAll("^(seq_|embed_|feat_)+", "");
        return sanitizeModuleName(withoutPrefix);
    }

    private static Device safeDevice(Tensor t) {
        try {
            return t.device();
        } catch (Throwable e) {
            return new Device("cpu");
        }
    }

    private Tensor clampIndices(Tensor idxOnDev, EmbeddingImpl embed, String embedKey) {
        try {
            long numEmb = embed.weight().size(0);
            long maxIdx = numEmb - 1;
            double anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble();
            double anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble();
            if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
                System.err.println("[WARNING] EmbeddingLayer: indices for '" + embedKey
                        + "' contain out-of-range values. Clamping to [0," + maxIdx + "].");
                warnedInvalidIndices.add(embedKey);
            }
            return idxOnDev.clamp(
                    new ScalarOptional(new Scalar(0L)),
                    new ScalarOptional(new Scalar(maxIdx)));
        } catch (Throwable t) {
            return idxOnDev;
        }
    }

    /**
     * Forward: sparse (+ optional sequence) → flattened 2D (batch, total_embed_dim).
     */
    public Tensor forward(Map<String, Tensor> sparseFeats,
                          Map<String, Tensor> sequenceFeats,
                          boolean squeeze) {
        if (sequenceFeats == null) {
            sequenceFeats = Collections.emptyMap();
        }
        List<Tensor> embeddingList = new ArrayList<>();

        // Filter sparse
        Map<String, Tensor> filteredSparse = new LinkedHashMap<>();
        List<String> missingSparse = new ArrayList<>();
        for (Map.Entry<String, Tensor> e : sparseFeats.entrySet()) {
            String key = "embed_" + baseName(e.getKey());
            if (embeddingTables.containsKey(key)) {
                filteredSparse.put(e.getKey(), e.getValue());
            } else {
                missingSparse.add(e.getKey());
            }
        }
        if (!missingSparse.isEmpty()) {
            String key = String.join(",", missingSparse);
            if (!warnedMissingSparse.contains(key)) {
                System.err.println("[WARNING] Ignoring unknown sparse features: "
                        + String.join(", ", missingSparse)
                        + ". Available tables: " + String.join(", ", embeddingTables.keySet()));
                warnedMissingSparse.add(key);
            }
        }

        for (Map.Entry<String, Tensor> e : filteredSparse.entrySet()) {
            String name = e.getKey();
            Tensor indices = e.getValue();
            String embedKey = "embed_" + baseName(name);
            EmbeddingImpl embed = embeddingTables.get(embedKey);
            if (embed == null) {
                continue;
            }
            try {
                Device embedDev = embed.weight().device();
                Tensor idx1d = (indices.dim() == 2L && indices.size(1) == 1L)
                        ? indices.squeeze(1L) : indices;
                Device idxDev = safeDevice(idx1d);
                Tensor idxOnDev;
                if (idxDev.equals(embedDev)) {
                    idxOnDev = idx1d.toType(ScalarType.Long);
                } else {
                    idxOnDev = idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
                }
                idxOnDev = clampIndices(idxOnDev, embed, embedKey);
                Tensor emb = embed.forward(idxOnDev);
                Tensor emb3d = emb.dim() == 2L ? emb.unsqueeze(1L) : emb;
                embeddingList.add(emb3d);
            } catch (Exception ex) {
                System.err.println("[WARNING] Failed to embed feature '" + name + "': " + ex.getMessage());
            }
        }

        // Filter sequence
        Map<String, Tensor> filteredSeq = new LinkedHashMap<>();
        List<String> missingSeq = new ArrayList<>();
        for (Map.Entry<String, Tensor> e : sequenceFeats.entrySet()) {
            String key = "embed_seq_" + baseName(e.getKey());
            if (embeddingTables.containsKey(key)) {
                filteredSeq.put(e.getKey(), e.getValue());
            } else {
                missingSeq.add(e.getKey());
            }
        }
        if (!missingSeq.isEmpty()) {
            String key = String.join(",", missingSeq);
            if (!warnedMissingSeq.contains(key)) {
                System.err.println("[WARNING] Ignoring unknown sequence features: "
                        + String.join(", ", missingSeq)
                        + ". Available tables: " + String.join(", ", embeddingTables.keySet()));
                warnedMissingSeq.add(key);
            }
        }

        for (Map.Entry<String, Tensor> e : filteredSeq.entrySet()) {
            String name = e.getKey();
            Tensor indices = e.getValue();
            String embedKey = "embed_seq_" + baseName(name);
            EmbeddingImpl embed = embeddingTables.get(embedKey);
            if (embed == null) {
                continue;
            }
            try {
                Device embedDev = embed.weight().device();
                Device idxDev = safeDevice(indices);
                Tensor idxOnDev;
                if (idxDev.equals(embedDev)) {
                    idxOnDev = indices.toType(ScalarType.Long);
                } else {
                    idxOnDev = indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
                }
                idxOnDev = clampIndices(idxOnDev, embed, embedKey);
                Tensor emb = embed.forward(idxOnDev);
                Tensor pooledEmb = poolSequence(emb, idxOnDev, "mean");
                Tensor emb3d = pooledEmb.dim() == 2L ? pooledEmb.unsqueeze(1L) : pooledEmb;
                embeddingList.add(emb3d);
            } catch (Exception ex) {
                System.err.println("[WARNING] Failed to embed sequence feature '" + name + "': " + ex.getMessage());
            }
        }

        if (embeddingList.isEmpty()) {
            String availableTables = String.join(", ", embeddingTables.keySet());
            StringBuilder inputFeats = new StringBuilder();
            for (String k : sparseFeats.keySet()) {
                if (inputFeats.length() > 0) inputFeats.append(", ");
                inputFeats.append(k);
            }
            for (String k : sequenceFeats.keySet()) {
                if (inputFeats.length() > 0) inputFeats.append(", ");
                inputFeats.append(k);
            }
            throw new IllegalArgumentException(
                    "No embeddings found for given features. Input features: [" + inputFeats
                            + "], Available embedding tables: [" + availableTables + "]");
        }

        int batchSize = (int) embeddingList.get(0).size(0);
        int totalDim = 0;
        for (Tensor e : embeddingList) {
            totalDim += (e.dim() == 3L) ? (int) e.size(2) : (int) e.size(1);
        }

        Device targetDev = safeDevice(embeddingList.get(0));
        TensorVector vec = new TensorVector();
        for (Tensor t : embeddingList) {
            Tensor onDev = safeDevice(t).equals(targetDev) ? t : t.to(targetDev, t.dtype());
            vec.push_back(onDev);
        }
        Tensor concatenated = torch.cat(vec, 1L);

        long actualNumel = concatenated.numel();
        long expectedNumel = (long) batchSize * totalDim;
        int finalTotalDim = (actualNumel % batchSize == 0L)
                ? (int) (actualNumel / batchSize) : totalDim;
        if (actualNumel != expectedNumel) {
            System.err.println("[EmbeddingLayer DEBUG] concatenated.numel()=" + actualNumel
                    + " expected=" + expectedNumel + "; falling back to per-batch dim=" + finalTotalDim);
        }

        Tensor flattened;
        try {
            flattened = concatenated.contiguous().view(batchSize, finalTotalDim);
        } catch (Throwable e) {
            StringBuilder shapes = new StringBuilder();
            for (Tensor emb : embeddingList) {
                if (shapes.length() > 0) shapes.append(",");
                try {
                    shapes.append(emb == null ? "null" : emb.sizes().toString());
                } catch (Throwable t) {
                    shapes.append("<unknown>");
                }
            }
            System.err.println("[EmbeddingLayer ERROR] Failed to view concatenated into (batch="
                    + batchSize + ", dim=" + finalTotalDim + "). concatenated.sizes="
                    + concatenated.sizes() + " embeddingShapes=[" + shapes + "]");
            throw e;
        }

        if (squeeze && embeddingList.size() == 1) {
            return flattened.squeeze(1L);
        }
        return flattened;
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap(), true);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> sequenceFeats) {
        return forward(sparseFeats, sequenceFeats, true);
    }

    /**
     * Forward pass returning 3D tensor (batch, num_fields, embed_dim).
     * Needed for models like AFM, AutoInt that require field dimension.
     */
    public Tensor forward3D(Map<String, Tensor> sparseFeats, Map<String, Tensor> sequenceFeats) {
        if (sequenceFeats == null) {
            sequenceFeats = Collections.emptyMap();
        }
        List<Tensor> embeddingList = new ArrayList<>();

        for (Map.Entry<String, Tensor> e : sparseFeats.entrySet()) {
            String name = e.getKey();
            Tensor indices = e.getValue();
            String embedKey = "embed_" + baseName(name);
            EmbeddingImpl embed = embeddingTables.get(embedKey);
            if (embed == null) {
                continue;
            }
            Device embedDev = embed.weight().device();
            Tensor idx1d = (indices.dim() == 2L && indices.size(1) == 1L)
                    ? indices.squeeze(1L) : indices;
            Device idxDev = safeDevice(idx1d);
            Tensor idxOnDev;
            if (idxDev.equals(embedDev)) {
                idxOnDev = idx1d.toType(ScalarType.Long);
            } else {
                idxOnDev = idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
            }
            try {
                long numEmb = embed.weight().size(0);
                long maxIdx = numEmb - 1;
                double anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble();
                double anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble();
                if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
                    System.err.println("[WARNING] EmbeddingLayer.forward3D: indices for '" + embedKey
                            + "' contain out-of-range values. Clamping to [0," + maxIdx + "].");
                    warnedInvalidIndices.add(embedKey);
                }
                idxOnDev = idxOnDev.clamp(
                        new ScalarOptional(new Scalar(0L)),
                        new ScalarOptional(new Scalar(maxIdx)));
            } catch (Throwable ignored) {
            }
            Tensor emb = embed.forward(idxOnDev);
            Tensor emb3d = emb.dim() == 2L ? emb.unsqueeze(1L) : emb;
            embeddingList.add(emb3d);
        }

        for (Map.Entry<String, Tensor> e : sequenceFeats.entrySet()) {
            String name = e.getKey();
            Tensor indices = e.getValue();
            String embedKey = "embed_seq_" + baseName(name);
            EmbeddingImpl embed = embeddingTables.get(embedKey);
            if (embed == null) {
                continue;
            }
            Device embedDev = embed.weight().device();
            Device idxDev = safeDevice(indices);
            Tensor idxOnDev;
            if (idxDev.equals(embedDev)) {
                idxOnDev = indices.toType(ScalarType.Long);
            } else {
                idxOnDev = indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
            }
            try {
                long numEmb = embed.weight().size(0);
                long maxIdx = numEmb - 1;
                double anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble();
                double anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble();
                if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
                    System.err.println("[WARNING] EmbeddingLayer.forward3D: sequence indices for '"
                            + embedKey + "' contain out-of-range values. Clamping to [0," + maxIdx + "].");
                    warnedInvalidIndices.add(embedKey);
                }
                idxOnDev = idxOnDev.clamp(
                        new ScalarOptional(new Scalar(0L)),
                        new ScalarOptional(new Scalar(maxIdx)));
            } catch (Throwable ignored) {
            }
            Tensor emb = embed.forward(idxOnDev);
            Tensor pooledEmb = poolSequence(emb, idxOnDev, "mean");
            Tensor emb3d = pooledEmb.dim() == 2L ? pooledEmb.unsqueeze(1L) : pooledEmb;
            embeddingList.add(emb3d);
        }

        if (embeddingList.isEmpty()) {
            throw new IllegalArgumentException("No embeddings found for given features");
        }

        Device targetDev = embeddingList.get(0).device();
        TensorVector vec = new TensorVector();
        for (Tensor t : embeddingList) {
            Tensor onDev = t.device().equals(targetDev) ? t : t.to(targetDev, t.dtype());
            vec.push_back(onDev);
        }
        return torch.cat(vec, 1L);
    }

    public Tensor forward3D(Map<String, Tensor> sparseFeats) {
        return forward3D(sparseFeats, Collections.emptyMap());
    }

    private Tensor poolSequence(Tensor emb, Tensor indices, String pooling) {
        long padIdx = paddingIdx != null ? paddingIdx : 0L;
        if (indices.dim() == 1L) {
            return emb;
        }
        switch (pooling) {
            case "mean": {
                Tensor padTensor = torch.full(new long[]{1L}, new Scalar(padIdx))
                        .to(indices.device(), ScalarType.Long);
                Tensor mask = indices.ne(padTensor).toType(ScalarType.Float);
                Tensor sum = emb.mul(mask.unsqueeze(2L)).sum(1L);
                Tensor count = mask.sum(1L).unsqueeze(1L);
                return sum.div(count);
            }
            case "sum":
                return emb.sum(1L);
            case "max": {
                var maxPair = torch.max(emb, 1L);
                return maxPair.get0();
            }
            case "last":
                return emb.mean(1L);
            default:
                return emb.mean(1L);
        }
    }

    /** Get embedding for a single sparse feature. */
    public Tensor getEmbedding(String name, Tensor indices) {
        String embedKey = "embed_" + baseName(name);
        EmbeddingImpl embed = embeddingTables.get(embedKey);
        if (embed == null) {
            throw new IllegalArgumentException("No embedding table for: " + name);
        }
        Tensor idx1d;
        if (indices.dim() == 2L && indices.size(1) == 1L) {
            idx1d = indices.squeeze(1L);
        } else {
            idx1d = indices;
        }
        Device embedDev = embed.weight().device();
        Tensor idxOnDev;
        if (idx1d.device().equals(embedDev)) {
            idxOnDev = idx1d.toType(ScalarType.Long);
        } else {
            idxOnDev = idx1d.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
        }
        try {
            long numEmb = embed.weight().size(0);
            long maxIdx = numEmb - 1;
            double anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble();
            double anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble();
            if ((anyLow != 0.0 || anyHigh != 0.0) && !warnedInvalidIndices.contains(embedKey)) {
                System.err.println("[WARNING] EmbeddingLayer.getEmbedding: indices for '" + embedKey
                        + "' contain out-of-range values. Clamping to [0," + maxIdx + "].");
                warnedInvalidIndices.add(embedKey);
            }
            idxOnDev = idxOnDev.clamp(
                    new ScalarOptional(new Scalar(0L)),
                    new ScalarOptional(new Scalar(maxIdx)));
        } catch (Throwable ignored) {
        }
        return embed.forward(idxOnDev);
    }

    /** Get embedding for a single sequence feature. */
    public Tensor getSequenceEmbedding(String name, Tensor indices) {
        String embedKey = "embed_seq_" + baseName(name);
        EmbeddingImpl embed = embeddingTables.get(embedKey);
        if (embed == null) {
            throw new IllegalArgumentException("No sequence embedding table for: " + name);
        }
        Device embedDev = embed.weight().device();
        Tensor idxOnDev;
        if (indices.device().equals(embedDev)) {
            idxOnDev = indices.toType(ScalarType.Long);
        } else {
            idxOnDev = indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
        }
        long numEmb = embed.weight().size(0);
        long maxIdx = numEmb - 1;
        double anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble();
        double anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble();
        if (anyLow != 0.0 || anyHigh != 0.0) {
            if (!warnedInvalidIndices.contains(embedKey)) {
                System.err.println("[WARNING] EmbeddingLayer.getSequenceEmbedding: indices for '"
                        + embedKey + "' contain out-of-range values. Clamping to [0," + maxIdx + "].");
                warnedInvalidIndices.add(embedKey);
            }
            idxOnDev = idxOnDev.clamp(
                    new ScalarOptional(new Scalar(0L)),
                    new ScalarOptional(new Scalar(maxIdx)));
        }
        return embed.forward(idxOnDev);
    }

    /**
     * Forward for sequence features only, without pooling.
     * Returns 3D tensor (batch, seqLen, embedDim) for single feature,
     * or concatenated along last dim for multiple.
     * Used by BST that needs raw sequence embeddings for self-attention.
     */
    public Tensor forwardSeqRaw(Map<String, Tensor> sequenceFeats) {
        List<Tensor> embeddingList = new ArrayList<>();

        for (Map.Entry<String, Tensor> e : sequenceFeats.entrySet()) {
            String name = e.getKey();
            Tensor indices = e.getValue();
            String embedKey = "embed_seq_" + baseName(name);
            EmbeddingImpl embed = embeddingTables.get(embedKey);
            if (embed == null) {
                continue;
            }
            Device embedDev = embed.weight().device();
            Tensor idxOnDev;
            if (indices.device().equals(embedDev)) {
                idxOnDev = indices.toType(ScalarType.Long);
            } else {
                idxOnDev = indices.toType(ScalarType.Long).to(embedDev, ScalarType.Long);
            }
            long numEmb = embed.weight().size(0);
            long maxIdx = numEmb - 1;
            double anyLow = idxOnDev.lt(new Scalar(0L)).any().item().toDouble();
            double anyHigh = idxOnDev.gt(new Scalar(maxIdx)).any().item().toDouble();
            if (anyLow != 0.0 || anyHigh != 0.0) {
                if (!warnedInvalidIndices.contains(embedKey)) {
                    System.err.println("[WARNING] EmbeddingLayer.forwardSeqRaw: sequence indices for '"
                            + embedKey + "' contain out-of-range values. Clamping to [0," + maxIdx + "].");
                    warnedInvalidIndices.add(embedKey);
                }
                idxOnDev = idxOnDev.clamp(
                        new ScalarOptional(new Scalar(0L)),
                        new ScalarOptional(new Scalar(maxIdx)));
            }
            embeddingList.add(embed.forward(idxOnDev));
        }

        if (embeddingList.isEmpty()) {
            throw new IllegalArgumentException("No sequence embeddings found for given features");
        }

        if (embeddingList.size() == 1) {
            return embeddingList.get(0);
        }

        Device targetDev = embeddingList.get(0).device();
        TensorVector vec = new TensorVector();
        for (Tensor t : embeddingList) {
            Tensor onDev = t.device().equals(targetDev) ? t : t.to(targetDev, t.dtype());
            vec.push_back(onDev);
        }
        return torch.cat(vec, 2);
    }

    /** Scala {@code def to(device: String): this.type} was a no-op; kept for API parity. */
    public EmbeddingLayer toDevice(String device) {
        return this;
    }
}
