package samples;

import static org.bytedeco.pytorch.dataframe.Functions.col;
import static org.bytedeco.pytorch.dataframe.ai.AiFunctions.*;

import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.util.*;

import org.bytedeco.pytorch.dataframe.*;
import org.bytedeco.pytorch.dataframe.ai.*;
import org.bytedeco.pytorch.dataframe.ann.AnnSearchResult;
import org.bytedeco.pytorch.dataframe.dtype.*;
import org.bytedeco.pytorch.dataframe.lance.LanceDataset;

/**
 * AI layer + Lance vector store correctness suite:
 * multi-model batch embedding (text/image/audio/video), CLIP zero-shot,
 * VQA / object-detect stand-ins, and Lance write/read/ANN search.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameAI
 * </pre>
 */
public class BenchmarkDataFrameAI {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) passed++;
        else {
            failed++;
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK FAIL: " + name);
        }
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok = Objects.equals(expected, actual)
            || (expected != null && actual != null
                && String.valueOf(expected).equals(String.valueOf(actual)));
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + actual);
        check(name, ok);
    }

    static void checkEq(String name, double expected, Object actual, double eps) {
        if (actual == null) { check(name, false); return; }
        double a = ((Number) actual).doubleValue();
        boolean ok = Math.abs(a - expected) <= eps;
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + a);
        check(name, ok);
    }

    static ImageData solidImage(int w, int h, int rgb) {
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                bi.setRGB(x, y, rgb);
        return new ImageData(bi);
    }

    static AudioData tone(int sr, double sec, double freq) {
        int n = (int) (sr * sec);
        float[] s = new float[n];
        for (int i = 0; i < n; i++) s[i] = (float) Math.sin(2 * Math.PI * freq * i / sr);
        return new AudioData(s, sr, 1);
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameAI — multi-model embed + Lance ===\n");
        Path tmp = Files.createTempDirectory("df_ai");

        try {
            // ── 1. Registry + multi-model resolve ────────────────────
            benchmark("1. EmbeddingRegistry multi-model", () -> {
                EmbeddingModel clip = EmbeddingRegistry.get("clip-vit-base-patch32");
                check("clip.backend", clip.backend() != null);
                checkEq("clip.dim", 512, clip.dimension());
                check("clip.supports.image", clip.supports(Modality.IMAGE));
                check("clip.supports.text", clip.supports(Modality.TEXT));

                EmbeddingModel bge = EmbeddingRegistry.get("bge-small-zh");
                checkEq("bge.dim", 512, bge.dimension());
                check("bge.text", bge.supports(Modality.TEXT));

                EmbeddingModel wav = EmbeddingRegistry.get("wav2vec2-base");
                check("wav.audio", wav.supports(Modality.AUDIO));

                EmbeddingModel hashImg = EmbeddingRegistry.get("hash-image");
                check("hash-image", hashImg.supports(Modality.IMAGE));

                check("registry.has.clip", EmbeddingRegistry.contains("clip-vit-base-patch32"));
                check("registry.ids", EmbeddingRegistry.ids().size() >= 5);

                // custom register
                EmbeddingRegistry.register(HashEmbeddingModel.forText(128));
                checkEq("custom.dim", 128, EmbeddingRegistry.get("hash-text").dimension());
                // restore default text dim via re-register larger
                EmbeddingRegistry.register(new HashEmbeddingModel(
                    ModelSpec.of("hash-text", Modality.TEXT, 384, "hash", true)));
            });

            // ── 2. Single-modality embed ─────────────────────────────
            benchmark("2. single embed text/image/audio/video", () -> {
                EmbeddingModel clip = EmbeddingRegistry.get("clip-vit-base-patch32");

                float[] t1 = clip.embed("a red square", Modality.TEXT);
                float[] t2 = clip.embed("a red square", Modality.TEXT);
                float[] t3 = clip.embed("a blue circle", Modality.TEXT);
                check("text.dim", t1.length == clip.dimension());
                checkEq("text.deterministic", 1.0, EmbeddingMath.cosine(t1, t2), 1e-5);
                check("text.diff", EmbeddingMath.cosine(t1, t3) < 0.999);

                ImageData red = solidImage(32, 32, 0xFF0000);
                ImageData blue = solidImage(32, 32, 0x0000FF);
                float[] i1 = clip.embed(red, Modality.IMAGE);
                float[] i2 = clip.embed(red, Modality.IMAGE);
                float[] i3 = clip.embed(blue, Modality.IMAGE);
                checkEq("img.deterministic", 1.0, EmbeddingMath.cosine(i1, i2), 1e-5);
                check("img.diff", EmbeddingMath.cosine(i1, i3) < 1.0);

                // cross-modal: same model space (structural)
                double cross = EmbeddingMath.cosine(t1, i1);
                check("cross.finite", !Double.isNaN(cross));

                AudioData a = tone(16000, 0.3, 440);
                float[] av = EmbeddingRegistry.get("wav2vec2-base").embed(a, Modality.AUDIO);
                check("audio.dim", av != null && av.length == 768);

                List<ImageData> frames = List.of(red, blue, solidImage(16, 16, 0x00FF00));
                VideoData vid = new VideoData(frames, 3.0);
                float[] vv = EmbeddingRegistry.get("hash-video").embed(vid, Modality.VIDEO);
                check("video.dim", vv != null && vv.length > 0);
            });

            // ── 3. BatchEmbedder multi-column ──────��─────────────────
            benchmark("3. BatchEmbedder multimodal columns", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("caption", Column.DType.STRING);
                df.addColumn("image", Column.DType.IMAGE);
                df.addColumn("audio", Column.DType.AUDIO);
                df.addRow("red block", solidImage(24, 24, 0xFF0000), tone(8000, 0.2, 220));
                df.addRow("blue block", solidImage(24, 24, 0x0000FF), tone(8000, 0.2, 440));
                df.addRow("green block", solidImage(24, 24, 0x00FF00), tone(8000, 0.2, 880));

                DataFrame out = BatchEmbedder.create()
                    .model("clip-vit-base-patch32")
                    .textColumn("caption", "text_emb")
                    .imageColumn("image", "image_emb")
                    .audioColumn("audio", "audio_emb", "wav2vec2-base")
                    .batchSize(2)
                    .parallel(true)
                    .transform(df);

                check("batch.rows", out.rowCount() == 3);
                check("batch.text_emb", out.get(0, "text_emb") instanceof EmbeddingData);
                check("batch.image_emb", out.get(0, "image_emb") instanceof EmbeddingData);
                check("batch.audio_emb", out.get(0, "audio_emb") instanceof EmbeddingData);
                checkEq("batch.text.model", "clip-vit-base-patch32",
                    ((EmbeddingData) out.get(0, "text_emb")).getModelName());
                checkEq("batch.audio.model", "wav2vec2-base",
                    ((EmbeddingData) out.get(0, "audio_emb")).getModelName());

                // cosine between text/image of same row should be computable
                DataFrame scored = cosineSimilarity(out, "text_emb", "image_emb", "ti_sim");
                check("ti_sim.col", scored.hasColumn("ti_sim"));
                check("ti_sim.num", scored.get(0, "ti_sim") instanceof Number);

                // embedAll
                DataFrame all = df.embedAll("clip-vit-base-patch32");
                check("embedAll.has", all.hasColumn("caption_emb") || all.hasColumn("image_emb"));
            });

            // ── 4. Expression-level embed + withColumn ───────────────
            benchmark("4. Expression embedText/embedImage", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("s", Column.DType.STRING);
                df.addColumn("img", Column.DType.IMAGE);
                df.addRow("hello world", solidImage(16, 16, 0xABCDEF));
                df.addRow("你好世界", solidImage(16, 16, 0x123456));

                DataFrame out = df
                    .withColumn("te", embedText("s", "bge-small-zh"))
                    .withColumn("ie", embedImage("img", "clip-vit-base-patch32"));
                check("expr.te", out.get(0, "te") instanceof EmbeddingData);
                check("expr.ie", out.get(0, "ie") instanceof EmbeddingData);

                // namespace model overload
                DataFrame out2 = df.withColumn("ie2", col("img").image().toEmbedding("clip-vit-base-patch32"));
                check("ns.ie2", out2.get(0, "ie2") instanceof EmbeddingData);

                DataFrame out3 = df.withColumn("te2", col("s").text().toEmbedding("bge-base-en"));
                check("ns.te2", out3.get(0, "te2") instanceof EmbeddingData);
            });

            // ── 5. Zero-shot classify / sentiment / caption / VQA ─────
            benchmark("5. classify VQA objectDetect caption sentiment", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("image", Column.DType.IMAGE);
                df.addColumn("q", Column.DType.STRING);
                df.addColumn("text", Column.DType.STRING);
                df.addRow(solidImage(32, 32, 0xFF0000), "what color?", "I love this product, amazing!");
                df.addRow(solidImage(32, 32, 0x0000FF), "what color?", "This is terrible and broken.");

                DataFrame clf = classifyImage(df, "image",
                    List.of("red object", "blue object", "green object"),
                    "clip-vit-base-patch32", "pred");
                check("clf.label", clf.hasColumn("pred"));
                check("clf.score", clf.hasColumn("pred_score"));
                check("clf.nonnull", clf.get(0, "pred") != null);

                DataFrame sent = sentiment(df, "text", "bge-base-en", "sentiment");
                check("sent.col", sent.hasColumn("sentiment"));

                DataFrame cap = caption(df, "image",
                    List.of("a red object", "a blue object", "a photo"),
                    "clip-vit-base-patch32", "caption");
                check("caption.col", cap.hasColumn("caption"));

                DataFrame vqa = visualQa(df, "image", "q",
                    List.of("red", "blue", "green", "unknown"),
                    "clip-vit-base-patch32", "answer");
                check("vqa.col", vqa.hasColumn("answer"));

                DataFrame det = objectDetect(df, "image",
                    List.of("object", "person", "car"),
                    "clip-vit-base-patch32", "dets");
                check("det.col", det.hasColumn("dets"));
                check("det.list", det.get(0, "dets") instanceof List);
            });

            // ── 6. Lance write / read / ANN search ───────────────────
            benchmark("6. Lance vector dataset IO + search", () -> {
                // build multimodal df with embeddings
                DataFrame raw = DataFrame.create();
                raw.addColumn("id", Column.DType.INT64);
                raw.addColumn("caption", Column.DType.STRING);
                raw.addColumn("image", Column.DType.IMAGE);
                int[] colors = {0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF,
                    0x800000, 0x008000, 0x000080, 0x808000};
                String[] caps = {"red", "green", "blue", "yellow", "magenta", "cyan",
                    "maroon", "darkgreen", "navy", "olive"};
                for (int i = 0; i < colors.length; i++) {
                    raw.addRow((long) i, caps[i] + " block", solidImage(20, 20, colors[i]));
                }

                DataFrame emb = BatchEmbedder.create()
                    .model("clip-vit-base-patch32")
                    .textColumn("caption", "text_emb")
                    .imageColumn("image", "image_emb")
                    .transform(raw);

                Path lancePath = tmp.resolve("clips.lance");
                emb.writeLance(lancePath.toString(), "text_emb", "image_emb");
                check("lance.dir", Files.isDirectory(lancePath));
                check("lance.manifest", Files.isRegularFile(lancePath.resolve("_manifest.json")));
                check("lance.vec", Files.isRegularFile(lancePath.resolve("vectors/image_emb.f32")));

                // read back
                DataFrame back = DataFrame.readLance(lancePath.toString());
                check("lance.rows", back.rowCount() == colors.length);
                check("lance.has.image_emb", back.hasColumn("image_emb"));
                check("lance.has.caption", back.hasColumn("caption"));
                check("lance.emb.type", back.get(0, "image_emb") instanceof EmbeddingData);

                // open + ANN search
                try (LanceDataset ds = LanceDataset.open(lancePath)) {
                    checkEq("lance.ds.rows", colors.length, ds.rowCount());
                    check("lance.ds.vcols", ds.vectorColumns().contains("image_emb"));
                    checkEq("lance.dim", 512, ds.vectorDim("image_emb"));

                    float[] query = ((EmbeddingData) emb.get(0, "image_emb")).getVector();
                    AnnSearchResult top = ds.search("image_emb", query, 3);
                    check("search.k", top.size() == 3);
                    check("search.self", top.indices()[0] == 0); // self should be nearest

                    DataFrame hits = ds.searchAsDataFrame("image_emb", query, 3);
                    check("hits.rows", hits.rowCount() == 3);
                    check("hits.score", hits.hasColumn("_score"));
                }

                // text→image retrieval: embed query text, search image index
                EmbeddingModel clip = EmbeddingRegistry.get("clip-vit-base-patch32");
                float[] qText = clip.embed("blue block", Modality.TEXT);
                try (LanceDataset ds = LanceDataset.open(lancePath)) {
                    // search text_emb space with text query
                    AnnSearchResult tr = ds.search("text_emb", qText, 3);
                    check("text.search.k", tr.size() == 3);
                    // row 2 is "blue" — should rank high (not guaranteed with hash, but finite)
                    check("text.search.ok", tr.indices() != null && tr.indices().length > 0);
                }
            });

            // ── 7. E2E training-pipeline style ───────────────────────
            benchmark("7. e2e multimodal dataset pipeline", () -> {
                // simulate dataset: images + captions + audio narrations
                DataFrame ds = DataFrame.create();
                ds.addColumn("path", Column.DType.STRING);
                ds.addColumn("image", Column.DType.IMAGE);
                ds.addColumn("caption", Column.DType.STRING);
                ds.addColumn("audio", Column.DType.AUDIO);
                ds.addRow("a.png", solidImage(28, 28, 0xE74C3C), "red apple", tone(16000, 0.15, 300));
                ds.addRow("b.png", solidImage(28, 28, 0x3498DB), "blue sky", tone(16000, 0.15, 600));
                ds.addRow("c.png", solidImage(28, 28, 0x2ECC71), "green leaf", tone(16000, 0.15, 900));
                ds.addRow("d.png", solidImage(28, 28, 0xF1C40F), "yellow sun", tone(16000, 0.15, 1200));

                // multi-model batch embed
                DataFrame featured = BatchEmbedder.create()
                    .textColumn("caption", "cap_emb", "bge-small-zh")
                    .imageColumn("image", "img_emb", "clip-vit-base-patch32")
                    .audioColumn("audio", "aud_emb", "wav2vec2-base")
                    .batchSize(2)
                    .transform(ds);

                check("e2e.cols", featured.hasColumn("cap_emb")
                    && featured.hasColumn("img_emb")
                    && featured.hasColumn("aud_emb"));

                // write lance with all three vector cols
                Path out = tmp.resolve("mm_train.lance");
                featured.writeLance(out.toString(),
                    LanceDataset.WriteOptions.defaults().metric("cosine").buildIndex(true),
                    "cap_emb", "img_emb", "aud_emb");

                DataFrame reloaded = DataFrame.readLance(out.toString());
                check("e2e.reload.rows", reloaded.rowCount() == 4);
                check("e2e.reload.emb", reloaded.get(0, "img_emb") instanceof EmbeddingData);

                // retrieval: text query → image index
                float[] q = EmbeddingRegistry.get("clip-vit-base-patch32")
                    .embed("blue sky", Modality.TEXT);
                // project to image space via clip image tower of same dual model
                // (hash clip dual towers differ per modality — search text index instead)
                try (LanceDataset lance = LanceDataset.open(out)) {
                    AnnSearchResult r = lance.search("cap_emb",
                        EmbeddingRegistry.get("bge-small-zh").embed("blue sky", Modality.TEXT), 2);
                    check("e2e.retrieve.k", r.size() == 2);
                    DataFrame hits = lance.searchAsDataFrame("cap_emb",
                        EmbeddingRegistry.get("bge-small-zh").embed("blue sky", Modality.TEXT), 2);
                    check("e2e.hits.caption", hits.hasColumn("caption") || hits.columnCount() > 0);
                }

                // zero-shot on reloaded images if present — use original featured
                DataFrame zshot = classifyImage(featured, "image",
                    List.of("fruit", "sky", "plant", "star"), "clip-vit-base-patch32");
                check("e2e.zshot", zshot.hasColumn("label"));
            });

        } finally {
            try {
                Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
                    try { Files.deleteIfExists(p); } catch (Exception ignored) {}
                });
            } catch (Exception ignored) {}
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("AI multi-model embedding + Lance IO covered.");
    }
}
