/*
 * Configurable batch collate for recommend datasets.
 *
 * Modes for packing a mini-batch {@link Batch} (or List&lt;Batch&gt; rows) into
 * native {@link org.bytedeco.pytorch.data.Example} / stacked tensors:
 *
 * <ul>
 *   <li>{@link Mode#FLAT_SCALARS} — 1-D concat of ordered sparse+dense scalars (default)</li>
 *   <li>{@link Mode#STACKED_FEATURES} — keep named maps; stack dim0 only (no Example pack)</li>
 *   <li>{@link Mode#MULTI_HOT} — per-field multi-hot [B, vocab] from id lists / scalars</li>
 *   <li>{@link Mode#PADDED_SEQUENCE} — pad variable-length sequences to maxLen with mask</li>
 *   <li>{@link Mode#HYBRID} — scalars flat + padded sequences side-by-side in Example data</li>
 * </ul>
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class BatchCollator {

    public enum Mode {
        FLAT_SCALARS,
        STACKED_FEATURES,
        MULTI_HOT,
        PADDED_SEQUENCE,
        HYBRID
    }

    /** Per-field multi-hot config. */
    public static final class MultiHotSpec {
        public final String field;
        public final int vocabSize;
        /** true → read from sequenceFeatures (id list); false → sparse scalar id. */
        public final boolean fromSequence;

        public MultiHotSpec(String field, int vocabSize) {
            this(field, vocabSize, false);
        }

        public MultiHotSpec(String field, int vocabSize, boolean fromSequence) {
            this.field = Objects.requireNonNull(field);
            this.vocabSize = vocabSize;
            this.fromSequence = fromSequence;
        }
    }

    /** Per-field sequence pad config. */
    public static final class SequenceSpec {
        public final String field;
        public final int maxLen;
        public final long padValue;
        /** left-pad if true, else right-pad. */
        public final boolean leftPad;
        public final boolean emitMask;

        public SequenceSpec(String field, int maxLen) {
            this(field, maxLen, 0L, false, true);
        }

        public SequenceSpec(String field, int maxLen, long padValue, boolean leftPad, boolean emitMask) {
            this.field = Objects.requireNonNull(field);
            this.maxLen = maxLen;
            this.padValue = padValue;
            this.leftPad = leftPad;
            this.emitMask = emitMask;
        }
    }

    public static final class Options {
        public Mode mode = Mode.FLAT_SCALARS;
        public List<String> sparseOrder = null;   // null → sorted keys of first row
        public List<String> denseOrder = null;
        public List<MultiHotSpec> multiHot = new ArrayList<>();
        public List<SequenceSpec> sequences = new ArrayList<>();
        public boolean includeLabel = true;
        public ScalarType indexDtype = ScalarType.Long;
        public ScalarType valueDtype = ScalarType.Float;

        public Options mode(Mode m) { this.mode = m; return this; }
        public Options sparseOrder(List<String> o) { this.sparseOrder = o; return this; }
        public Options denseOrder(List<String> o) { this.denseOrder = o; return this; }
        public Options multiHot(MultiHotSpec... specs) {
            Collections.addAll(this.multiHot, specs);
            return this;
        }
        public Options sequences(SequenceSpec... specs) {
            Collections.addAll(this.sequences, specs);
            return this;
        }
        public Options includeLabel(boolean v) { this.includeLabel = v; return this; }

        public static Options defaults() { return new Options(); }
    }

    /** Result of collating a mini-batch. */
    public static final class Collated {
        /** Named stacked batch (always filled for STACKED / useful for models). */
        public final Batch batch;
        /** Packed Example for native DataLoader path (may be null in STACKED_FEATURES). */
        public final Example example;
        /** Optional per-sequence masks: field -> [B, maxLen] float 1=valid. */
        public final Map<String, Tensor> sequenceMasks;
        /** Optional multi-hot tensors: field -> [B, vocab]. */
        public final Map<String, Tensor> multiHotFeatures;

        public Collated(Batch batch, Example example,
                        Map<String, Tensor> sequenceMasks,
                        Map<String, Tensor> multiHotFeatures) {
            this.batch = batch;
            this.example = example;
            this.sequenceMasks = sequenceMasks != null ? sequenceMasks : Collections.emptyMap();
            this.multiHotFeatures = multiHotFeatures != null ? multiHotFeatures : Collections.emptyMap();
        }
    }

    private final Options opts;

    public BatchCollator() {
        this(Options.defaults());
    }

    public BatchCollator(Options opts) {
        this.opts = opts != null ? opts : Options.defaults();
    }

    public Options options() { return opts; }

    // ---- public API ---------------------------------------------------------

    /** Collate a list of single-row Batches. */
    public Collated collate(List<Batch> rows) {
        if (rows == null || rows.isEmpty()) {
            throw new IllegalArgumentException("collate: empty rows");
        }
        Batch stacked = stackNamed(rows);
        Map<String, Tensor> masks = new LinkedHashMap<>();
        Map<String, Tensor> multiHots = new LinkedHashMap<>();

        switch (opts.mode) {
            case STACKED_FEATURES:
                return new Collated(stacked, null, masks, multiHots);

            case MULTI_HOT: {
                multiHots.putAll(buildMultiHots(rows));
                Example ex = packMultiHotExample(multiHots, stacked);
                // also expose multi-hots on batch via denseFeatures merge? keep separate map
                return new Collated(stacked, ex, masks, multiHots);
            }

            case PADDED_SEQUENCE: {
                PadResult pad = padSequences(rows);
                masks.putAll(pad.masks);
                Batch withSeq = mergeSequences(stacked, pad.padded);
                Example ex = packFlat(withSeq);
                return new Collated(withSeq, ex, masks, multiHots);
            }

            case HYBRID: {
                multiHots.putAll(buildMultiHots(rows));
                PadResult pad = padSequences(rows);
                masks.putAll(pad.masks);
                Batch withSeq = mergeSequences(stacked, pad.padded);
                Example ex = packHybrid(withSeq, multiHots, pad);
                return new Collated(withSeq, ex, masks, multiHots);
            }

            case FLAT_SCALARS:
            default: {
                Example ex = packFlat(stacked);
                return new Collated(stacked, ex, masks, multiHots);
            }
        }
    }

    /** Collate already-stacked mini-batch (single Batch with B>1) into Example. */
    public Example toExample(Batch batch) {
        return packFlat(batch);
    }

    // ---- stacking named features --------------------------------------------

    public static Batch stackNamed(List<Batch> rows) {
        return DataLoader.stackBatches(rows, "cpu");
    }

    // ---- multi-hot ----------------------------------------------------------

    private Map<String, Tensor> buildMultiHots(List<Batch> rows) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        if (opts.multiHot == null || opts.multiHot.isEmpty()) return out;
        int B = rows.size();
        for (MultiHotSpec spec : opts.multiHot) {
            float[] flat = new float[B * spec.vocabSize];
            for (int b = 0; b < B; b++) {
                Batch row = rows.get(b);
                if (spec.fromSequence) {
                    Tensor seq = row.sequenceFeatures.get(spec.field);
                    if (seq == null) seq = row.tokens;
                    if (seq != null) {
                        long[] ids = toLongIds(seq);
                        for (long id : ids) {
                            if (id > 0 && id < spec.vocabSize) {
                                flat[b * spec.vocabSize + (int) id] = 1f;
                            }
                        }
                    }
                } else {
                    Tensor t = row.sparseFeatures.get(spec.field);
                    if (t == null) t = row.denseFeatures.get(spec.field);
                    if (t != null) {
                        long[] ids = toLongIds(t);
                        if (ids.length > 0) {
                            long id = ids[0];
                            if (id >= 0 && id < spec.vocabSize) {
                                flat[b * spec.vocabSize + (int) id] = 1f;
                            }
                        }
                    }
                }
            }
            out.put(spec.field, TensorHelpers.tensor(flat, B, spec.vocabSize));
        }
        return out;
    }

    private Example packMultiHotExample(Map<String, Tensor> multiHots, Batch stacked) {
        if (multiHots.isEmpty()) return packFlat(stacked);
        TensorVector vec = new TensorVector();
        for (Tensor t : multiHots.values()) {
            // [B, V] -> keep and cat on last dim later; for Example use flat per-row via view
            vec.push_back(t);
        }
        Tensor cat = torch.cat(vec, 1); // [B, sumV]
        return new Example(cat, labelTensor(stacked));
    }

    // ---- padded sequences ---------------------------------------------------

    private static final class PadResult {
        final Map<String, Tensor> padded = new LinkedHashMap<>();
        final Map<String, Tensor> masks = new LinkedHashMap<>();
    }

    private PadResult padSequences(List<Batch> rows) {
        PadResult res = new PadResult();
        if (opts.sequences == null || opts.sequences.isEmpty()) {
            // auto: pad every sequence feature to max length observed
            Map<String, Integer> maxLens = new LinkedHashMap<>();
            for (Batch row : rows) {
                for (Map.Entry<String, Tensor> e : row.sequenceFeatures.entrySet()) {
                    int len = sequenceLength(e.getValue());
                    maxLens.merge(e.getKey(), len, Math::max);
                }
                if (row.tokens != null) {
                    int len = sequenceLength(row.tokens);
                    maxLens.merge("__tokens__", len, Math::max);
                }
            }
            for (Map.Entry<String, Integer> e : maxLens.entrySet()) {
                SequenceSpec spec = new SequenceSpec(
                        e.getKey(), Math.max(1, e.getValue()), 0L, false, true);
                padOneField(rows, spec, res);
            }
            return res;
        }
        for (SequenceSpec spec : opts.sequences) {
            padOneField(rows, spec, res);
        }
        return res;
    }

    private void padOneField(List<Batch> rows, SequenceSpec spec, PadResult res) {
        int B = rows.size();
        int L = Math.max(1, spec.maxLen);
        long[] data = new long[B * L];
        float[] mask = new float[B * L];
        // fill pad
        for (int i = 0; i < data.length; i++) data[i] = spec.padValue;

        for (int b = 0; b < B; b++) {
            Batch row = rows.get(b);
            Tensor src;
            if ("__tokens__".equals(spec.field)) {
                src = row.tokens;
            } else {
                src = row.sequenceFeatures.get(spec.field);
                if (src == null && "item_seq".equals(spec.field)) src = row.tokens;
            }
            long[] ids = src != null ? toLongIds(src) : new long[0];
            int n = Math.min(ids.length, L);
            if (spec.leftPad) {
                int start = L - n;
                for (int t = 0; t < n; t++) {
                    data[b * L + start + t] = ids[t];
                    mask[b * L + start + t] = 1f;
                }
            } else {
                for (int t = 0; t < n; t++) {
                    data[b * L + t] = ids[t];
                    mask[b * L + t] = 1f;
                }
            }
        }
        Tensor padded = TensorHelpers.longTensorDirect(data).reshape(B, L);
        res.padded.put(spec.field, padded);
        if (spec.emitMask) {
            res.masks.put(spec.field, TensorHelpers.tensor(mask, B, L));
        }
    }

    private static Batch mergeSequences(Batch base, Map<String, Tensor> padded) {
        if (padded.isEmpty()) return base;
        Map<String, Tensor> seq = new LinkedHashMap<>(base.sequenceFeatures);
        seq.putAll(padded);
        Tensor tokens = base.tokens;
        if (padded.containsKey("__tokens__")) tokens = padded.get("__tokens__");
        return new Batch(
                base.sparseFeatures, base.denseFeatures, seq, base.labels,
                tokens, base.positions, base.timeDiffs, base.targets,
                base.itemFeatures, base.negItemFeatures, base.taskLabels);
    }

    // ---- packing ------------------------------------------------------------

    private Example packFlat(Batch batch) {
        List<String> sparseOrder = resolveSparseOrder(batch);
        List<String> denseOrder = resolveDenseOrder(batch);
        // For batched [B,1] features, pack per-row into [B, F]
        int B = (int) batch.numSamples();
        if (B <= 0) B = 1;
        int F = sparseOrder.size() + denseOrder.size();
        if (F == 0 && batch.tokens != null) {
            // fallback: use tokens flattened
            Tensor data = batch.tokens.toType(ScalarType.Float).reshape(B, -1).contiguous().clone();
            return new Example(data, labelTensor(batch));
        }
        if (F == 0) {
            return new Example(torch.zeros(new long[]{B, 1L}), labelTensor(batch));
        }
        float[] flat = new float[B * F];
        int col = 0;
        for (String name : sparseOrder) {
            fillColumn(flat, B, F, col++, batch.sparseFeatures.get(name));
        }
        for (String name : denseOrder) {
            fillColumn(flat, B, F, col++, batch.denseFeatures.get(name));
        }
        Tensor data = TensorHelpers.tensor(flat, B, F);
        return new Example(data, labelTensor(batch));
    }

    private Example packHybrid(Batch batch, Map<String, Tensor> multiHots, PadResult pad) {
        // data = [flat scalars | multi-hot cat | flattened padded seqs]
        Example flat = packFlat(batch);
        List<Tensor> parts = new ArrayList<>();
        parts.add(flat.data());
        if (!multiHots.isEmpty()) {
            TensorVector v = new TensorVector();
            for (Tensor t : multiHots.values()) v.push_back(t);
            parts.add(torch.cat(v, 1));
        }
        for (Tensor seq : pad.padded.values()) {
            // [B, L] -> [B, L] as float
            parts.add(seq.toType(ScalarType.Float));
        }
        if (parts.size() == 1) return flat;
        TensorVector vec = new TensorVector(parts.size());
        for (int i = 0; i < parts.size(); i++) vec.put(i, parts.get(i));
        Tensor data = torch.cat(vec, 1);
        return new Example(data, flat.target());
    }

    private static void fillColumn(float[] flat, int B, int F, int col, Tensor t) {
        if (t == null) return;
        float[] vals = TensorHelpers.toFloatArray(t.toType(ScalarType.Float));
        for (int b = 0; b < B; b++) {
            float v = b < vals.length ? vals[b] : (vals.length > 0 ? vals[0] : 0f);
            flat[b * F + col] = v;
        }
    }

    private Tensor labelTensor(Batch batch) {
        if (!opts.includeLabel || batch.labels == null) {
            long B = Math.max(1, batch.numSamples());
            return torch.zeros(new long[]{B});
        }
        Tensor y = batch.labels.toType(ScalarType.Float).reshape(-1L).contiguous().clone();
        return y;
    }

    private List<String> resolveSparseOrder(Batch batch) {
        if (opts.sparseOrder != null && !opts.sparseOrder.isEmpty()) return opts.sparseOrder;
        List<String> keys = new ArrayList<>(batch.sparseFeatures.keySet());
        Collections.sort(keys);
        return keys;
    }

    private List<String> resolveDenseOrder(Batch batch) {
        if (opts.denseOrder != null && !opts.denseOrder.isEmpty()) return opts.denseOrder;
        List<String> keys = new ArrayList<>(batch.denseFeatures.keySet());
        Collections.sort(keys);
        return keys;
    }

    private static int sequenceLength(Tensor t) {
        if (t == null) return 0;
        if (t.dim() == 0) return 1;
        if (t.dim() == 1) return (int) t.size(0);
        // [1, L] or [L]
        return (int) t.size(t.dim() - 1);
    }

    private static long[] toLongIds(Tensor t) {
        Tensor c = t.contiguous();
        // Always intern() before comparing ScalarType (JavaCPP non-canonical proxies).
        ScalarType st = c.scalar_type().intern();
        if (st != ScalarType.Long) {
            c = c.toType(ScalarType.Long);
        }
        return TensorHelpers.toLongArray(c);
    }
}
