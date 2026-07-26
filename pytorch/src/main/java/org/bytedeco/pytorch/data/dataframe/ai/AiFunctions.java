package org.bytedeco.pytorch.data.dataframe.ai;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.Expression;
import org.bytedeco.pytorch.data.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.data.dataframe.dtype.ImageData;

/**
 * Daft-style {@code daft.functions.ai.*} entry points for multimodal AI ops.
 *
 * <pre>
 *   import static org.bytedeco.pytorch.data.dataframe.ai.AiFunctions.*;
 *
 *   df = df.withColumn("emb", embedText("caption", "bge-small-zh"));
 *   df = df.withColumn("img_emb", embedImage("image", "clip-vit-base-patch32"));
 *   df = classifyImage(df, "image", List.of("cat", "dog"), "clip-vit-base-patch32");
 * </pre>
 */
public final class AiFunctions {
    private AiFunctions() {}

    // ── embed expressions ──────────────────────────────────────────────

    /** Embed a text column with the given model → EMBEDDING expression. */
    public static Expression embedText(String column, String modelId) {
        return new EmbedExpr(Expression.col(column), modelId, Modality.TEXT);
    }

    public static Expression embedText(Expression expr, String modelId) {
        return new EmbedExpr(expr, modelId, Modality.TEXT);
    }

    public static Expression embedImage(String column, String modelId) {
        return new EmbedExpr(Expression.col(column), modelId, Modality.IMAGE);
    }

    public static Expression embedImage(Expression expr, String modelId) {
        return new EmbedExpr(expr, modelId, Modality.IMAGE);
    }

    public static Expression embedAudio(String column, String modelId) {
        return new EmbedExpr(Expression.col(column), modelId, Modality.AUDIO);
    }

    public static Expression embedAudio(Expression expr, String modelId) {
        return new EmbedExpr(expr, modelId, Modality.AUDIO);
    }

    public static Expression embedVideo(String column, String modelId) {
        return new EmbedExpr(Expression.col(column), modelId, Modality.VIDEO);
    }

    public static Expression embedVideo(Expression expr, String modelId) {
        return new EmbedExpr(expr, modelId, Modality.VIDEO);
    }

    /** Auto-detect modality from cell values. */
    public static Expression embed(String column, String modelId) {
        return new EmbedExpr(Expression.col(column), modelId, null);
    }

    public static Expression embed(Expression expr, String modelId) {
        return new EmbedExpr(expr, modelId, null);
    }

    // ── DataFrame-level batch helpers ──────────────────────────────────

    public static DataFrame embedTextColumn(DataFrame df, String inputCol, String outputCol, String modelId) {
        return BatchEmbedder.create().model(modelId).textColumn(inputCol, outputCol).transform(df);
    }

    public static DataFrame embedImageColumn(DataFrame df, String inputCol, String outputCol, String modelId) {
        return BatchEmbedder.create().model(modelId).imageColumn(inputCol, outputCol).transform(df);
    }

    public static DataFrame embedAudioColumn(DataFrame df, String inputCol, String outputCol, String modelId) {
        return BatchEmbedder.create().model(modelId).audioColumn(inputCol, outputCol).transform(df);
    }

    public static DataFrame embedVideoColumn(DataFrame df, String inputCol, String outputCol, String modelId) {
        return BatchEmbedder.create().model(modelId).videoColumn(inputCol, outputCol).transform(df);
    }

    /** Embed all detectable multimodal columns with one model (CLIP dual-encoder recommended). */
    public static DataFrame embedAll(DataFrame df, String modelId) {
        return BatchEmbedder.embedAll(df, modelId);
    }

    /**
     * Zero-shot image classification via CLIP-style cosine similarity to label text embeddings.
     * Adds {@code outputCol} (predicted label) and {@code outputCol + "_score"} (confidence).
     */
    public static DataFrame classifyImage(DataFrame df, String imageCol, List<String> labels,
                                          String modelId, String outputCol) {
        Objects.requireNonNull(labels, "labels");
        if (labels.isEmpty()) throw new IllegalArgumentException("labels empty");
        String out = outputCol == null ? "label" : outputCol;
        String scoreCol = out + "_score";

        EmbeddingModel model = EmbeddingRegistry.get(modelId == null ? "clip-vit-base-patch32" : modelId);
        model.warmup();

        // embed label prompts once
        float[][] labelVecs = new float[labels.size()][];
        for (int i = 0; i < labels.size(); i++) {
            String prompt = "a photo of a " + labels.get(i);
            labelVecs[i] = model.embed(prompt, Modality.TEXT);
        }

        Column imgs = df.column(imageCol);
        List<Object> predLabels = new ArrayList<>(df.rowCount());
        List<Object> predScores = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = imgs.get(r);
            if (cell == null) {
                predLabels.add(null);
                predScores.add(null);
                continue;
            }
            float[] imgVec = model.embed(cell, Modality.IMAGE);
            int best = 0;
            double bestScore = -Double.MAX_VALUE;
            for (int i = 0; i < labelVecs.length; i++) {
                double s = EmbeddingMath.cosine(imgVec, labelVecs[i]);
                if (s > bestScore) { bestScore = s; best = i; }
            }
            predLabels.add(labels.get(best));
            predScores.add(bestScore);
        }
        return df.withColumn(out, predLabels).withColumn(scoreCol, predScores);
    }

    public static DataFrame classifyImage(DataFrame df, String imageCol, List<String> labels, String modelId) {
        return classifyImage(df, imageCol, labels, modelId, "label");
    }

    /**
     * Zero-shot text classification (same cosine-to-label-embedding pattern).
     */
    public static DataFrame classifyText(DataFrame df, String textCol, List<String> labels,
                                         String modelId, String outputCol) {
        Objects.requireNonNull(labels, "labels");
        String out = outputCol == null ? "label" : outputCol;
        String scoreCol = out + "_score";
        EmbeddingModel model = EmbeddingRegistry.get(modelId == null ? "bge-small-zh" : modelId);
        model.warmup();

        float[][] labelVecs = new float[labels.size()][];
        for (int i = 0; i < labels.size(); i++) {
            labelVecs[i] = model.embed(labels.get(i), Modality.TEXT);
        }

        Column texts = df.column(textCol);
        List<Object> predLabels = new ArrayList<>(df.rowCount());
        List<Object> predScores = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = texts.get(r);
            if (cell == null) { predLabels.add(null); predScores.add(null); continue; }
            float[] v = model.embed(cell, Modality.TEXT);
            int best = 0; double bestScore = -Double.MAX_VALUE;
            for (int i = 0; i < labelVecs.length; i++) {
                double s = EmbeddingMath.cosine(v, labelVecs[i]);
                if (s > bestScore) { bestScore = s; best = i; }
            }
            predLabels.add(labels.get(best));
            predScores.add(bestScore);
        }
        return df.withColumn(out, predLabels).withColumn(scoreCol, predScores);
    }

    /**
     * Visual QA stand-in: embed question+image jointly and return nearest caption template.
     * Without a generative LLM this returns a structured answer map with similarity scores
     * against provided candidate answers (or a default set).
     */
    public static DataFrame visualQa(DataFrame df, String imageCol, String questionCol,
                                     List<String> candidateAnswers, String modelId, String outputCol) {
        String out = outputCol == null ? "answer" : outputCol;
        EmbeddingModel model = EmbeddingRegistry.get(modelId == null ? "clip-vit-base-patch32" : modelId);
        model.warmup();

        List<String> cands = candidateAnswers;
        if (cands == null || cands.isEmpty()) {
            cands = List.of("yes", "no", "unknown", "a person", "an object", "an animal", "a scene");
        }
        float[][] candVecs = new float[cands.size()][];
        for (int i = 0; i < cands.size(); i++) {
            candVecs[i] = model.embed(cands.get(i), Modality.TEXT);
        }

        Column imgs = df.column(imageCol);
        Column qs = df.hasColumn(questionCol) ? df.column(questionCol) : null;
        List<Object> answers = new ArrayList<>(df.rowCount());
        List<Object> scores = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            Object img = imgs.get(r);
            Object q = qs == null ? null : qs.get(r);
            if (img == null) { answers.add(null); scores.add(null); continue; }
            float[] imgVec = model.embed(img, Modality.IMAGE);
            float[] qVec = q == null ? null : model.embed(q, Modality.TEXT);
            // fuse image + question in shared space
            float[] query = qVec == null ? imgVec : mix(imgVec, qVec, 0.6f);
            int best = 0; double bestScore = -Double.MAX_VALUE;
            for (int i = 0; i < candVecs.length; i++) {
                double s = EmbeddingMath.cosine(query, candVecs[i]);
                if (s > bestScore) { bestScore = s; best = i; }
            }
            answers.add(cands.get(best));
            scores.add(bestScore);
        }
        return df.withColumn(out, answers).withColumn(out + "_score", scores);
    }

    /**
     * Simple object-detect stand-in: returns list of {label, score, box} via
     * sliding-window CLIP classification on image tiles (no external detector weights).
     */
    public static DataFrame objectDetect(DataFrame df, String imageCol, List<String> labels,
                                         String modelId, String outputCol) {
        String out = outputCol == null ? "detections" : outputCol;
        EmbeddingModel model = EmbeddingRegistry.get(modelId == null ? "clip-vit-base-patch32" : modelId);
        model.warmup();
        List<String> labs = (labels == null || labels.isEmpty())
            ? List.of("person", "car", "dog", "cat", "object") : labels;

        float[][] labVecs = new float[labs.size()][];
        for (int i = 0; i < labs.size(); i++) {
            labVecs[i] = model.embed("a photo of a " + labs.get(i), Modality.TEXT);
        }

        Column imgs = df.column(imageCol);
        List<Object> allDets = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = imgs.get(r);
            if (!(cell instanceof ImageData img) || img.getImage() == null) {
                allDets.add(List.of());
                continue;
            }
            int w = img.getWidth(), h = img.getHeight();
            // 2x2 grid tiles
            List<Map<String, Object>> dets = new ArrayList<>();
            int gw = Math.max(1, w / 2), gh = Math.max(1, h / 2);
            for (int ty = 0; ty < 2; ty++) {
                for (int tx = 0; tx < 2; tx++) {
                    int x1 = tx * gw, y1 = ty * gh;
                    int tw = Math.min(gw, w - x1), th = Math.min(gh, h - y1);
                    if (tw <= 0 || th <= 0) continue;
                    try {
                        ImageData tile = img.crop(x1, y1, tw, th);
                        float[] vec = model.embed(tile, Modality.IMAGE);
                        int best = 0; double bestScore = -Double.MAX_VALUE;
                        for (int i = 0; i < labVecs.length; i++) {
                            double s = EmbeddingMath.cosine(vec, labVecs[i]);
                            if (s > bestScore) { bestScore = s; best = i; }
                        }
                        if (bestScore > 0.05) { // loose threshold for hash backend
                            Map<String, Object> det = new LinkedHashMap<>();
                            det.put("label", labs.get(best));
                            det.put("score", bestScore);
                            det.put("x1", x1); det.put("y1", y1);
                            det.put("x2", x1 + tw); det.put("y2", y1 + th);
                            dets.add(det);
                        }
                    } catch (Exception ignored) {}
                }
            }
            allDets.add(dets);
        }
        return df.withColumn(out, allDets);
    }

    /**
     * Image caption stand-in: pick nearest caption template by CLIP similarity.
     */
    public static DataFrame caption(DataFrame df, String imageCol, List<String> templates,
                                    String modelId, String outputCol) {
        String out = outputCol == null ? "caption" : outputCol;
        List<String> tpls = (templates == null || templates.isEmpty())
            ? List.of("a photo", "a red object", "a blue object", "a green object",
                      "a person", "an animal", "a landscape", "an indoor scene")
            : templates;
        return visualQa(df, imageCol, null, tpls, modelId, out);
    }

    /**
     * Sentiment via embedding similarity to polarity prototypes.
     */
    public static DataFrame sentiment(DataFrame df, String textCol, String modelId, String outputCol) {
        return classifyText(df, textCol,
            List.of("positive", "negative", "neutral"),
            modelId == null ? "bge-base-en" : modelId,
            outputCol == null ? "sentiment" : outputCol);
    }

    /**
     * Pairwise cosine between two embedding columns → float score column.
     */
    public static DataFrame cosineSimilarity(DataFrame df, String embColA, String embColB, String outputCol) {
        String out = outputCol == null ? "cosine" : outputCol;
        Column a = df.column(embColA);
        Column b = df.column(embColB);
        List<Object> scores = new ArrayList<>(df.rowCount());
        for (int i = 0; i < df.rowCount(); i++) {
            float[] va = toFloat(a.get(i));
            float[] vb = toFloat(b.get(i));
            scores.add(va == null || vb == null ? null : EmbeddingMath.cosine(va, vb));
        }
        return df.withColumn(out, scores);
    }

    private static float[] toFloat(Object v) {
        if (v == null) return null;
        if (v instanceof EmbeddingData ed) return ed.getVector();
        if (v instanceof float[] f) return f;
        return null;
    }

    private static float[] mix(float[] a, float[] b, float alphaA) {
        if (a == null) return b;
        if (b == null) return a;
        int n = Math.min(a.length, b.length);
        float[] out = new float[Math.max(a.length, b.length)];
        float alphaB = 1f - alphaA;
        for (int i = 0; i < n; i++) out[i] = alphaA * a[i] + alphaB * b[i];
        return EmbeddingMath.l2Normalize(out);
    }

    // ── Expression node ────────────────────────────────────────────────

    /**
     * Lazy embedding expression evaluated row-wise (use {@link BatchEmbedder}
     * for large batches — this path is convenient for {@code withColumn}).
     */
    public static final class EmbedExpr extends Expression {
        private final Expression child;
        private final String modelId;
        private final Modality modality;
        private transient EmbeddingModel cachedModel;

        EmbedExpr(Expression child, String modelId, Modality modality) {
            this.child = child;
            this.modelId = modelId == null ? "clip-vit-base-patch32" : modelId;
            this.modality = modality;
        }

        private EmbeddingModel model() {
            if (cachedModel == null) {
                cachedModel = EmbeddingRegistry.get(modelId);
                cachedModel.warmup();
            }
            return cachedModel;
        }

        @Override
        public Object eval(int row, DataFrame df) {
            Object v = child.eval(row, df);
            if (v == null) return null;
            Modality m = modality;
            if (m == null) m = ClipStyleEmbeddingModel.detect(v);
            float[] vec = model().embed(v, m);
            return model().toEmbeddingData(vec);
        }

        @Override
        public Column evaluate(DataFrame df) {
            // batch path for efficiency
            int n = df.rowCount();
            List<Object> inputs = new ArrayList<>(n);
            for (int i = 0; i < n; i++) inputs.add(child.eval(i, df));
            Modality m = modality;
            if (m == null && !inputs.isEmpty()) {
                for (Object o : inputs) {
                    if (o != null) { m = ClipStyleEmbeddingModel.detect(o); break; }
                }
            }
            if (m == null) m = Modality.TEXT;
            float[][] vecs = model().embedBatch(inputs, m);
            List<Object> data = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                float[] v = i < vecs.length ? vecs[i] : null;
                data.add(v == null ? null : model().toEmbeddingData(v));
            }
            return new Column(suggestedName(), Column.DType.EMBEDDING, data);
        }

        @Override public String suggestedName() {
            return "embed_" + (modality == null ? "auto" : modality.name().toLowerCase())
                + "(" + child.suggestedName() + ")";
        }

        @Override public java.util.Set<String> referencedColumns() {
            return child.referencedColumns();
        }
    }
}
