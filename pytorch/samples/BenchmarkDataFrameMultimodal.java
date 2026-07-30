package samples;

import static org.bytedeco.pytorch.dataframe.Functions.*;

import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.util.*;

import org.bytedeco.pytorch.dataframe.*;
import org.bytedeco.pytorch.dataframe.dtype.*;

/**
 * Correctness suite for Daft-aligned multimodal DataFrame operators:
 * image / audio / video / tensor / text namespaces, plus multimodal IO helpers.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameMultimodal
 * </pre>
 */
public class BenchmarkDataFrameMultimodal {
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

    static AudioData tone(int sr, double seconds, double freq) {
        int n = (int) (sr * seconds);
        float[] samples = new float[n];
        for (int i = 0; i < n; i++) {
            samples[i] = (float) Math.sin(2 * Math.PI * freq * i / sr);
        }
        return new AudioData(samples, sr, 1);
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameMultimodal — Daft-aligned ops ===\n");
        Path tmp = Files.createTempDirectory("df_mm");

        try {
            // ── Daft base gaps: negate/expm1/coalesce/ifNull/replace/astype ──
            benchmark("0. Daft scalar gaps", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("x", Column.DType.FLOAT64);
                df.addColumn("y", Column.DType.FLOAT64);
                df.addRow(1.0, null);
                df.addRow(null, 2.0);
                df.addRow(3.0, 4.0);

                Column c = col("x").negate().evaluate(df);
                checkEq("negate[0]", -1.0, c.get(0), 1e-9);

                c = col("x").expm1().evaluate(df);
                checkEq("expm1[0]", Math.expm1(1.0), c.get(0), 1e-9);

                c = col("x").coalesce(col("y")).evaluate(df);
                checkEq("coalesce[0]", 1.0, c.get(0), 1e-9);
                checkEq("coalesce[1]", 2.0, c.get(1), 1e-9);

                c = col("y").ifNull(0.0).evaluate(df);
                checkEq("ifNull[0]", 0.0, c.get(0), 1e-9);

                c = col("x").replace(3.0, 99.0).evaluate(df);
                checkEq("replace[2]", 99.0, c.get(2), 1e-9);

                c = col("x").astype(Column.DType.INT64).evaluate(df);
                checkEq("astype[0]", 1L, c.get(0));

                DataFrame lim = df.limit(2);
                check("limit", lim.rowCount() == 2);
                DataFrame wh = df.where(col("x").isNotNull());
                check("where", wh.rowCount() == 2);
            });

            // ── Image namespace ──────────────────────────────────────
            benchmark("1. image namespace", () -> {
                ImageData img = solidImage(32, 24, 0xFF0000); // red
                DataFrame df = DataFrame.create();
                df.addColumn("img", Column.DType.IMAGE);
                df.addRow(img);
                df.addRow(solidImage(16, 16, 0x00FF00));

                // resize
                Column c = col("img").image().resize(8, 8).evaluate(df);
                check("resize.type", c.get(0) instanceof ImageData);
                checkEq("resize.w", 8, ((ImageData) c.get(0)).getWidth());
                checkEq("resize.h", 8, ((ImageData) c.get(0)).getHeight());

                // grayscale
                c = col("img").image().toGrayscale().evaluate(df);
                check("gray.type", c.get(0) instanceof ImageData);

                // flip / rotate / blur / sharpen
                check("flip", col("img").image().flipHorizontal().evaluate(df).get(0) instanceof ImageData);
                check("rotate", col("img").image().rotate(90).evaluate(df).get(0) instanceof ImageData);
                check("blur", col("img").image().blur().evaluate(df).get(0) instanceof ImageData);
                check("sharpen", col("img").image().sharpen().evaluate(df).get(0) instanceof ImageData);
                check("equalize", col("img").image().equalizeHist().evaluate(df).get(0) instanceof ImageData);

                // pad
                c = col("img").image().pad(2, 2, 2, 2).evaluate(df);
                ImageData padded = (ImageData) c.get(0);
                checkEq("pad.w", 36, padded.getWidth()); // 32+4
                checkEq("pad.h", 28, padded.getHeight());

                // crop
                c = col("img").image().crop(0, 0, 10, 10).evaluate(df);
                checkEq("crop.w", 10, ((ImageData) c.get(0)).getWidth());

                // to_array
                c = col("img").image().toArray().evaluate(df);
                check("to_array", c.get(0) instanceof float[]);
                check("to_array.len", ((float[]) c.get(0)).length == 32 * 24 * 3);

                // info
                c = col("img").image().info().evaluate(df);
                check("info.map", c.get(0) instanceof Map);
                @SuppressWarnings("unchecked")
                Map<String, Object> info = (Map<String, Object>) c.get(0);
                checkEq("info.w", 32, info.get("width"));

                // encode
                c = col("img").image().encode("PNG", 90).evaluate(df);
                check("encode", c.get(0) instanceof BinaryData);

                // phash
                c = col("img").image().phash().evaluate(df);
                check("phash", c.get(0) instanceof String);

                // embedding
                c = col("img").image().toEmbedding(32).evaluate(df);
                check("embed", c.get(0) instanceof EmbeddingData);
                checkEq("embed.dim", 32, ((EmbeddingData) c.get(0)).getDimension());

                // normalize → float array
                c = col("img").image().normalize(
                    new float[]{0.5f, 0.5f, 0.5f},
                    new float[]{0.5f, 0.5f, 0.5f}).evaluate(df);
                check("normalize", c.get(0) instanceof float[]);
            });

            // ── Audio namespace ──────────────────────────────────────
            benchmark("2. audio namespace", () -> {
                AudioData a = tone(16000, 0.5, 440);
                DataFrame df = DataFrame.create();
                df.addColumn("aud", Column.DType.AUDIO);
                df.addRow(a);

                Column c = col("aud").audio().metadata().evaluate(df);
                check("meta.map", c.get(0) instanceof Map);
                @SuppressWarnings("unchecked")
                Map<String, Object> m = (Map<String, Object>) c.get(0);
                checkEq("meta.sr", 16000, m.get("sample_rate"));

                c = col("aud").audio().normalize().evaluate(df);
                check("normalize", c.get(0) instanceof AudioData);

                c = col("aud").audio().toMono().evaluate(df);
                check("mono", c.get(0) instanceof AudioData);
                checkEq("mono.ch", 1, ((AudioData) c.get(0)).getChannels());

                c = col("aud").audio().toStereo().evaluate(df);
                check("stereo", c.get(0) instanceof AudioData);
                checkEq("stereo.ch", 2, ((AudioData) c.get(0)).getChannels());

                c = col("aud").audio().resample(8000).evaluate(df);
                check("resample", c.get(0) instanceof AudioData);
                checkEq("resample.sr", 8000, ((AudioData) c.get(0)).getSampleRate());

                c = col("aud").audio().trim(0.0f, 0.1f).evaluate(df);
                check("trim", c.get(0) instanceof AudioData);

                c = col("aud").audio().mfcc(13).evaluate(df);
                check("mfcc", c.get(0) instanceof float[][]);

                c = col("aud").audio().spectrogram().evaluate(df);
                check("spec", c.get(0) instanceof float[][]);

                c = col("aud").audio().toEmbedding().evaluate(df);
                check("aud.embed", c.get(0) instanceof EmbeddingData);

                c = col("aud").audio().denoise().evaluate(df);
                check("denoise", c.get(0) instanceof AudioData);
            });

            // ── Video namespace ──────────────────────────────────────
            benchmark("3. video namespace", () -> {
                List<ImageData> frames = new ArrayList<>();
                for (int i = 0; i < 10; i++) frames.add(solidImage(16, 16, 0x101010 * i));
                VideoData vid = new VideoData(frames, 10.0);
                vid.setWidth(16); vid.setHeight(16); vid.setDuration(1.0);

                DataFrame df = DataFrame.create();
                df.addColumn("vid", Column.DType.VIDEO);
                df.addRow(vid);

                Column c = col("vid").video().metadata().evaluate(df);
                check("vmeta", c.get(0) instanceof Map);

                c = col("vid").video().frameAt(0.5).evaluate(df);
                check("frameAt", c.get(0) instanceof ImageData);

                c = col("vid").video().extractFrames(2.0).evaluate(df);
                check("frames", c.get(0) instanceof List);
                @SuppressWarnings("unchecked")
                List<ImageData> extracted = (List<ImageData>) c.get(0);
                check("frames.count", extracted.size() >= 2);

                c = col("vid").video().resize(8, 8).evaluate(df);
                check("vresize", c.get(0) instanceof VideoData);

                c = col("vid").video().trim(0.0, 0.5).evaluate(df);
                check("vtrim", c.get(0) instanceof VideoData);
            });

            // ── Tensor / embedding namespace ─────────────────────────
            benchmark("4. tensor namespace", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("emb", Column.DType.EMBEDDING);
                df.addColumn("emb2", Column.DType.EMBEDDING);
                df.addRow(new EmbeddingData(new float[]{3f, 4f, 0f}, "t"),
                          new EmbeddingData(new float[]{3f, 4f, 0f}, "t"));
                df.addRow(new EmbeddingData(new float[]{1f, 0f, 0f}, "t"),
                          new EmbeddingData(new float[]{0f, 1f, 0f}, "t"));

                // l2_norm: [3,4,0] → [0.6, 0.8, 0]
                Column c = col("emb").tensor().l2Norm().evaluate(df);
                check("l2.type", c.get(0) instanceof EmbeddingData || c.get(0) instanceof float[]);
                float[] n0;
                if (c.get(0) instanceof EmbeddingData ed) n0 = ed.getVector();
                else n0 = (float[]) c.get(0);
                checkEq("l2[0]", 0.6, n0[0], 1e-5);
                checkEq("l2[1]", 0.8, n0[1], 1e-5);

                // dot
                c = col("emb").tensor().dot(col("emb2")).evaluate(df);
                checkEq("dot[0]", 25.0, c.get(0), 1e-9); // 9+16
                checkEq("dot[1]", 0.0, c.get(1), 1e-9);

                // cosine
                c = col("emb").tensor().cosineSim(col("emb2")).evaluate(df);
                checkEq("cos[0]", 1.0, c.get(0), 1e-5);
                checkEq("cos[1]", 0.0, c.get(1), 1e-5);

                // flatten / reshape / mean / sum
                DataFrame td = DataFrame.create();
                td.addColumn("t", Column.DType.TENSOR);
                td.addRow(new TensorData(new float[]{1, 2, 3, 4}, new int[]{2, 2}));

                c = col("t").tensor().flatten().evaluate(td);
                check("flatten", c.get(0) instanceof TensorData);
                checkEq("flatten.size", 4, ((TensorData) c.get(0)).size());

                c = col("t").tensor().reshape(4).evaluate(td);
                check("reshape", c.get(0) instanceof TensorData);

                c = col("t").tensor().mean().evaluate(td);
                checkEq("t.mean", 2.5, c.get(0), 1e-9);
                c = col("t").tensor().sum().evaluate(td);
                checkEq("t.sum", 10.0, c.get(0), 1e-9);
                c = col("t").tensor().max().evaluate(td);
                checkEq("t.max", 4.0, c.get(0), 1e-9);
                c = col("t").tensor().min().evaluate(td);
                checkEq("t.min", 1.0, c.get(0), 1e-9);

                c = col("t").tensor().slice(1, 3).evaluate(td);
                check("slice", c.get(0) instanceof float[]);
                checkEq("slice.len", 2, ((float[]) c.get(0)).length);

                c = col("t").tensor().transpose().evaluate(td);
                check("transpose", c.get(0) instanceof TensorData);

                c = col("emb").tensor().concat(col("emb2")).evaluate(df);
                check("concat", c.get(0) instanceof float[]);
                checkEq("concat.len", 6, ((float[]) c.get(0)).length);
            });

            // ── Text namespace ───────────────────────────────────────
            benchmark("5. text namespace", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("s", Column.DType.STRING);
                df.addRow("  Hello, World!!!  Contact me@x.com or 13800138000 ");
                df.addRow("The quick brown fox is running.");
                df.addRow("这是一个测试的句子。还有第二句！");

                Column c = col("s").text().clean().evaluate(df);
                check("clean", c.get(0) instanceof String);
                check("clean.trim", !c.get(0).toString().startsWith(" "));

                c = col("s").text().tokenize().evaluate(df);
                check("tokenize", c.get(0) instanceof List);

                c = col("s").text().sentenceSplit().evaluate(df);
                check("sents", c.get(0) instanceof List);

                c = col("s").text().removeStopwords("en").evaluate(df);
                String noStop = String.valueOf(c.get(1));
                check("stopwords", !noStop.toLowerCase().contains(" the ")
                    || !noStop.toLowerCase().startsWith("the "));

                c = col("s").text().piiMask().evaluate(df);
                String masked = String.valueOf(c.get(0));
                check("pii.email", masked.contains("[EMAIL]"));
                check("pii.phone", masked.contains("[PHONE]"));

                c = col("s").text().summarize().evaluate(df);
                check("summarize", c.get(0) instanceof String);

                c = col("s").text().toEmbedding(16).evaluate(df);
                check("text.embed", c.get(0) instanceof EmbeddingData);
                checkEq("text.embed.dim", 16, ((EmbeddingData) c.get(0)).getDimension());

                c = col("s").text().lemmatize().evaluate(df);
                check("lemma", c.get(0) instanceof String);
            });

            // ── Multimodal IO ────────────────────────────────────────
            benchmark("6. multimodal IO", () -> {
                // write a small PNG and read back
                ImageData img = solidImage(4, 4, 0x0000FF);
                Path imgPath = tmp.resolve("blue.png");
                img.save(imgPath.toString());

                DataFrame idf = DataFrame.readImages(imgPath.toString());
                check("readImages.rows", idf.rowCount() == 1);
                check("readImages.cols", idf.hasColumn("image") && idf.hasColumn("path"));
                check("readImages.cell", idf.get(0, "image") instanceof ImageData);

                // directory load
                Path imgDir = tmp.resolve("imgs");
                Files.createDirectories(imgDir);
                solidImage(2, 2, 0xFF0000).save(imgDir.resolve("a.png").toString());
                solidImage(2, 2, 0x00FF00).save(imgDir.resolve("b.png").toString());
                DataFrame dirDf = DataFrame.readImages(imgDir.toString());
                check("readImages.dir", dirDf.rowCount() == 2);

                // fromImages / fromEmbeddings
                DataFrame fi = DataFrame.fromImages("img", List.of(solidImage(3, 3, 1), solidImage(3, 3, 2)));
                check("fromImages", fi.rowCount() == 2);
                DataFrame fe = DataFrame.fromEmbeddings("e",
                    new float[][]{{1, 0}, {0, 1}}, "test");
                check("fromEmbeddings", fe.rowCount() == 2);
                check("fromEmbeddings.type", fe.get(0, "e") instanceof EmbeddingData);

                // download local file as binary
                DataFrame urls = DataFrame.create();
                urls.addColumn("url", Column.DType.STRING);
                urls.addRow(imgPath.toString());
                DataFrame dl = urls.download("url", "bin");
                check("download", dl.get(0, "bin") instanceof BinaryData);
            });

            // ── E2E multimodal pipeline ──────────────────────────────
            benchmark("7. e2e multimodal pipeline", () -> {
                // image → embed → l2 → cosine self-sim
                ImageData a = solidImage(32, 32, 0xFF0000);
                ImageData b = solidImage(32, 32, 0xFF0000);
                ImageData cimg = solidImage(32, 32, 0x0000FF);

                DataFrame df = DataFrame.create();
                df.addColumn("img", Column.DType.IMAGE);
                df.addColumn("label", Column.DType.STRING);
                df.addRow(a, "red1");
                df.addRow(b, "red2");
                df.addRow(cimg, "blue");

                DataFrame out = df
                    .withColumn("emb", col("img").image().toEmbedding(64))
                    .withColumn("emb_n", col("emb").tensor().l2Norm())
                    .withColumn("gray", col("img").image().toGrayscale())
                    .withColumn("thumb", col("img").image().resize(8, 8));

                check("e2e.rows", out.rowCount() == 3);
                check("e2e.emb", out.get(0, "emb") instanceof EmbeddingData);
                check("e2e.thumb", out.get(0, "thumb") instanceof ImageData);

                // text + image fusion table
                DataFrame text = DataFrame.create();
                text.addColumn("caption", Column.DType.STRING);
                text.addRow("a red square");
                text.addRow("another red");
                text.addRow("blue block");
                text = text.withColumn("text_emb", col("caption").text().toEmbedding(64));

                DataFrame fused = DataFrame.hstack(out.select("label", "emb_n"), text.select("caption", "text_emb"));
                check("e2e.fused.cols", fused.columnCount() == 4);
                check("e2e.fused.rows", fused.rowCount() == 3);

                // cosine between first two image embeddings (same color → high sim)
                float[] e0 = ((EmbeddingData) out.get(0, "emb")).getVector();
                float[] e1 = ((EmbeddingData) out.get(1, "emb")).getVector();
                float[] e2 = ((EmbeddingData) out.get(2, "emb")).getVector();
                double sim01 = cosine(e0, e1);
                double sim02 = cosine(e0, e2);
                check("e2e.sim.same", sim01 > 0.99);
                check("e2e.sim.diff", sim02 < sim01 + 1e-6); // blue vs red should not be higher
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
        System.out.println("Daft multimodal operator groups covered.");
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double d = Math.sqrt(na) * Math.sqrt(nb);
        return d == 0 ? 0 : dot / d;
    }
}
