package org.bytedeco.pytorch.dataframe.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;

/**
 * Batch-embed one or more DataFrame columns with pluggable {@link EmbeddingModel}s.
 *
 * <p>Daft-aligned multi-model multimodal embedding:
 * <pre>
 *   DataFrame out = BatchEmbedder.create()
 *       .model("clip-vit-base-patch32")
 *       .textColumn("caption", "caption_emb")
 *       .imageColumn("image", "image_emb")
 *       .audioColumn("audio", "audio_emb")
 *       .videoColumn("video", "video_emb")
 *       .batchSize(64)
 *       .parallel(true)
 *       .transform(df);
 * </pre>
 */
public final class BatchEmbedder {
    private String defaultModelId = "clip-vit-base-patch32";
    private final List<Job> jobs = new ArrayList<>();
    private int batchSize = 32;
    private boolean parallel = true;
    private boolean keepInput = true;

    private BatchEmbedder() {}

    public static BatchEmbedder create() { return new BatchEmbedder(); }

    public BatchEmbedder model(String modelId) {
        this.defaultModelId = modelId;
        return this;
    }

    public BatchEmbedder batchSize(int n) {
        this.batchSize = Math.max(1, n);
        return this;
    }

    public BatchEmbedder parallel(boolean v) {
        this.parallel = v;
        return this;
    }

    /** Embed {@code inputCol} as TEXT → {@code outputCol} using default / override model. */
    public BatchEmbedder textColumn(String inputCol, String outputCol) {
        return textColumn(inputCol, outputCol, null);
    }

    public BatchEmbedder textColumn(String inputCol, String outputCol, String modelId) {
        jobs.add(new Job(inputCol, outputCol, Modality.TEXT, modelId));
        return this;
    }

    public BatchEmbedder imageColumn(String inputCol, String outputCol) {
        return imageColumn(inputCol, outputCol, null);
    }

    public BatchEmbedder imageColumn(String inputCol, String outputCol, String modelId) {
        jobs.add(new Job(inputCol, outputCol, Modality.IMAGE, modelId));
        return this;
    }

    public BatchEmbedder audioColumn(String inputCol, String outputCol) {
        return audioColumn(inputCol, outputCol, null);
    }

    public BatchEmbedder audioColumn(String inputCol, String outputCol, String modelId) {
        jobs.add(new Job(inputCol, outputCol, Modality.AUDIO, modelId));
        return this;
    }

    public BatchEmbedder videoColumn(String inputCol, String outputCol) {
        return videoColumn(inputCol, outputCol, null);
    }

    public BatchEmbedder videoColumn(String inputCol, String outputCol, String modelId) {
        jobs.add(new Job(inputCol, outputCol, Modality.VIDEO, modelId));
        return this;
    }

    /** Generic column with explicit modality. */
    public BatchEmbedder column(String inputCol, String outputCol, Modality modality, String modelId) {
        jobs.add(new Job(inputCol, outputCol, modality, modelId));
        return this;
    }

    /**
     * Run all configured embedding jobs on {@code df}.
     * Each job appends/replaces an EMBEDDING column.
     */
    public DataFrame transform(DataFrame df) {
        Objects.requireNonNull(df, "df");
        if (jobs.isEmpty()) return df.copy();

        DataFrame out = df;
        for (Job job : jobs) {
            out = embedOne(out, job);
        }
        return out;
    }

    private DataFrame embedOne(DataFrame df, Job job) {
        if (!df.hasColumn(job.inputCol)) {
            throw new IllegalArgumentException("No such column: " + job.inputCol);
        }
        String modelId = job.modelId != null ? job.modelId : defaultModelId;
        EmbeddingModel model = EmbeddingRegistry.get(modelId);
        model.warmup();

        Column src = df.column(job.inputCol);
        int n = df.rowCount();
        List<Object> inputs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) inputs.add(src.get(i));

        float[][] vectors = embedInBatches(model, inputs, job.modality);
        List<Object> embCol = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = i < vectors.length ? vectors[i] : null;
            embCol.add(v == null ? null : model.toEmbeddingData(v));
        }
        return df.withColumn(job.outputCol, embCol);
    }

    private float[][] embedInBatches(EmbeddingModel model, List<Object> inputs, Modality modality) {
        int n = inputs.size();
        float[][] out = new float[n][];
        if (n == 0) return out;

        List<int[]> ranges = new ArrayList<>();
        for (int start = 0; start < n; start += batchSize) {
            ranges.add(new int[]{start, Math.min(n, start + batchSize)});
        }

        if (parallel && ranges.size() > 1) {
            List<Future<BatchResult>> futures = new ArrayList<>(ranges.size());
            for (int[] r : ranges) {
                final int s = r[0], e = r[1];
                futures.add(ForkJoinPool.commonPool().submit(() -> {
                    List<Object> slice = inputs.subList(s, e);
                    float[][] vecs = model.embedBatch(slice, modality);
                    return new BatchResult(s, vecs);
                }));
            }
            try {
                for (Future<BatchResult> f : futures) {
                    BatchResult br = f.get();
                    for (int i = 0; i < br.vecs.length; i++) {
                        out[br.start + i] = br.vecs[i];
                    }
                }
            } catch (Exception e) {
                // fallback sequential
                return model.embedBatch(inputs, modality);
            }
        } else {
            for (int[] r : ranges) {
                List<Object> slice = inputs.subList(r[0], r[1]);
                float[][] vecs = model.embedBatch(slice, modality);
                for (int i = 0; i < vecs.length; i++) out[r[0] + i] = vecs[i];
            }
        }
        return out;
    }

    /**
     * Auto-detect modality columns and embed all of them.
     * Detects IMAGE / AUDIO / VIDEO / STRING columns.
     */
    public static DataFrame embedAll(DataFrame df, String modelId) {
        BatchEmbedder b = create().model(modelId);
        for (Column c : df.columns()) {
            String name = c.name();
            String out = name + "_emb";
            switch (c.dtype()) {
                case IMAGE -> b.imageColumn(name, out);
                case AUDIO -> b.audioColumn(name, out);
                case VIDEO -> b.videoColumn(name, out);
                case STRING -> b.textColumn(name, out);
                case EMBEDDING, VECTOR, TENSOR -> { /* already embedded */ }
                default -> {
                    // peek first non-null
                    Modality m = peekModality(c);
                    if (m != null) b.column(name, out, m, null);
                }
            }
        }
        if (b.jobs.isEmpty()) return df.copy();
        return b.transform(df);
    }

    private static Modality peekModality(Column c) {
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) continue;
            if (v instanceof ImageData) return Modality.IMAGE;
            if (v instanceof AudioData) return Modality.AUDIO;
            if (v instanceof VideoData) return Modality.VIDEO;
            if (v instanceof CharSequence) return Modality.TEXT;
            if (v instanceof EmbeddingData) return null;
            break;
        }
        return null;
    }

    private record Job(String inputCol, String outputCol, Modality modality, String modelId) {}
    private record BatchResult(int start, float[][] vecs) {}
}
