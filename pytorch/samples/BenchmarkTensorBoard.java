package samples;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.utils.tensorboard.SummaryWriter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * TensorBoard benchmark that mirrors <b>real training usage</b> of
 * {@code torch.utils.tensorboard.SummaryWriter}.
 *
 * <p>Public API exercised is Tensor-first / PyTorch-shaped only:
 * {@code add_scalar, add_scalars, add_histogram, add_image, add_images,
 * add_audio, add_text, add_pr_curve, add_hparams, add_embedding, add_mesh,
 * add_custom_scalars_*}. No raw byte[] / HWC plumbing in the training path.
 *
 * <pre>
 *   java samples.BenchmarkTensorBoard [logdir]
 *   tensorboard --logdir runs/tb_bench
 * </pre>
 */
public class BenchmarkTensorBoard {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();

    private static Scalar sc(double v) { return new Scalar(v); }
    private static Scalar sc(float v) { return new Scalar(v); }

    private static TensorOptions longOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
    }

    private static TensorOptions floatOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
    }

    // =========================================================================
    // Models
    // =========================================================================

    static final class MlpClassifier extends Module {
        final LinearImpl fc1, fc2, fc3;

        MlpClassifier(long inFeatures, long hidden, long nClass) {
            super("MlpClassifier");
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hidden));
            fc2 = register_module("fc2", new LinearImpl(hidden, hidden / 2));
            fc3 = register_module("fc3", new LinearImpl(hidden / 2, nClass));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = relu(fc1.forward(x));
            h = dropout(h, 0.1, is_training());
            h = relu(fc2.forward(h));
            return log_softmax(fc3.forward(h), 1);
        }
    }

    static final class Regressor extends Module {
        final LinearImpl fc1, fc2;

        Regressor(long inFeatures, long hidden) {
            super("Regressor");
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hidden));
            fc2 = register_module("fc2", new LinearImpl(hidden, 1));
        }

        @Override
        public Tensor forward(Tensor x) {
            return fc2.forward(relu(fc1.forward(x)));
        }
    }

    static final class BinaryNet extends Module {
        final LinearImpl fc1, fc2;

        BinaryNet(long inFeatures, long hidden) {
            super("BinaryNet");
            fc1 = register_module("fc1", new LinearImpl(inFeatures, hidden));
            fc2 = register_module("fc2", new LinearImpl(hidden, 1));
        }

        /** Probabilities in (0,1), shape [N]. */
        @Override
        public Tensor forward(Tensor x) {
            return sigmoid(fc2.forward(relu(fc1.forward(x)))).squeeze(1);
        }
    }

    // =========================================================================
    // main
    // =========================================================================

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        System.out.println("=== TensorBoard training benchmark (PyTorch-style API) ===\n");

        Path logRoot = args.length > 0
                ? Path.of(args[0])
                : Path.of("runs", "tb_bench_" + System.currentTimeMillis());
        Files.createDirectories(logRoot);
        System.out.println("logdir: " + logRoot.toAbsolutePath());
        System.out.println("open with:  tensorboard --logdir " + logRoot.toAbsolutePath());
        System.out.println();

        try (PointerScope scope = new PointerScope()) {
            benchmark("train MLP on synthetic digits → real-looking images", () ->
                    trainMlpClassifier(logRoot.resolve("mlp_cls")));

            benchmark("train regressor → scalars/hist/hparams", () ->
                    trainRegressor(logRoot.resolve("regressor")));

            benchmark("train binary net → pr_curve/embedding + structured audio", () ->
                    trainBinaryNet(logRoot.resolve("binary")));

            benchmark("multimodal showcase: digits/geometry/colorbar/audio/video/mesh", () ->
                    demoMultimodalShowcase(logRoot.resolve("multimodal")));

            benchmark("heatmap + embedding projector (with sprite)", () ->
                    demoHeatmapAndEmbedding(logRoot.resolve("heatmap_embed")));

            benchmark("mesh + custom_scalars layout demo", () ->
                    demoMeshAndLayout(logRoot.resolve("viz")));
        }

        benchmark("Python EventAccumulator can read training logs", () ->
                verifyWithPython(logRoot));

        System.out.println("\n=== Results ===");
        System.out.println("Passed checks: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("\nAll training runs logged. Start TensorBoard:");
        System.out.println("  tensorboard --logdir " + logRoot.toAbsolutePath());
    }

    // =========================================================================
    // 1) MLP classification
    // =========================================================================

    static void trainMlpClassifier(Path logDir) throws Exception {
        Files.createDirectories(logDir);
        // 28x28 digit-like patterns (readable in TB Images tab), not pure noise
        final long N = 64;
        final long C = 1, H = 28, W = 28;
        final long feat = C * H * W;
        final long nClass = 10;
        final int epochs = 4;
        final int stepsPerEpoch = 15;

        MlpClassifier net = new MlpClassifier(feat, 128, nClass);
        net.train(true);
        SGD opt = new SGD(net.parameters(), new SGDOptions(0.08));

        try (SummaryWriter w = new SummaryWriter(logDir.toString())) {
            w.add_text("run/config",
                    "model=MlpClassifier on synthetic MNIST-style digits " +
                            H + "x" + W + " classes=" + nClass + " lr=0.08 SGD", 0);
            w.add_custom_scalars_multilinechart(
                    List.of("train/loss", "train/acc"), "Train", "loss_vs_acc");

            // Reference sheet: one clean glyph per class 0..9
            w.add_images("refs/digit_sheet", makeDigitBatch(new int[]{0,1,2,3,4,5,6,7,8,9}, (int) H), 0L);
            w.add_image("refs/digit_7", makeDigitImage(7, (int) H), 0L);
            w.add_image("refs/checker", makeCheckerboard(32, 4), 0L);
            w.add_image("refs/colorbar", makeColorBar(64, 16), 0L);

            long globalStep = 0;
            for (int epoch = 1; epoch <= epochs; epoch++) {
                double epochLoss = 0;
                double epochAcc = 0;
                for (int step = 0; step < stepsPerEpoch; step++, globalStep++) {
                    int[] labels = new int[(int) N];
                    for (int i = 0; i < N; i++) labels[i] = (int) ((globalStep * 7 + i * 3) % nClass);
                    Tensor images = makeDigitBatch(labels, (int) H); // NCHW, with mild noise
                    // add slight noise so training is non-trivial but glyphs stay visible
                    images = images.add(randn(new long[]{N, C, H, W}).mul(sc(0.05)));
                    images = images.maximum(tensor(0f)).minimum(tensor(1f));
                    Tensor x = images.reshape(N, feat);
                    Tensor y = tensor(toLongLabels(labels));

                    opt.zero_grad();
                    Tensor logp = net.forward(x);
                    Tensor loss = nll_loss(logp, y);
                    loss.backward();
                    opt.step();

                    Tensor pred = logp.argmax(new LongOptional(1L), false);
                    Tensor correct = pred.eq(y).to(ScalarType.Float).mean();

                    w.add_scalar("train/loss", loss, globalStep);
                    w.add_scalar("train/acc", correct, globalStep);

                    Map<String, Object> group = new LinkedHashMap<>();
                    group.put("loss", loss.item_float());
                    group.put("acc", correct.item_float());
                    w.add_scalars("train/grouped", group, globalStep);

                    if (globalStep % 5 == 0) {
                        logParameterHistograms(w, net, globalStep);
                        // show first 8 batch images — should look like digits 0-9
                        w.add_images("train/batch_digits", images.narrow(0, 0, 8), globalStep);
                        w.add_image("train/one_digit", images.select(0, 0), globalStep);
                        // also log a clean prediction gallery for classes 0..7
                        int[] show = new int[8];
                        for (int i = 0; i < 8; i++) show[i] = i;
                        w.add_images("train/clean_glyphs", makeDigitBatch(show, (int) H), globalStep);
                    }

                    epochLoss += loss.item_float();
                    epochAcc += correct.item_float();
                }
                epochLoss /= stepsPerEpoch;
                epochAcc /= stepsPerEpoch;
                w.add_scalar("epoch/loss", epochLoss, epoch);
                w.add_scalar("epoch/acc", epochAcc, epoch);
                w.add_text("epoch/summary",
                        String.format("epoch=%d loss=%.4f acc=%.4f (digits should be visible in Images)", epoch, epochLoss, epochAcc),
                        epoch);
                w.flush();
                System.out.printf("  [mlp] epoch %d  loss=%.4f  acc=%.4f%n", epoch, epochLoss, epochAcc);
            }
            check("mlp event file exists",
                    Files.list(logDir).anyMatch(p -> p.getFileName().toString().contains("tfevents")));
        }
    }

    // =========================================================================
    // 2) Regressor + hparams
    // =========================================================================

    static void trainRegressor(Path logDir) throws Exception {
        Files.createDirectories(logDir);
        double[] lrs = {0.01, 0.05};
        int[] hiddens = {32, 64};

        for (double lr : lrs) {
            for (int hidden : hiddens) {
                String runName = String.format("lr%.3f_h%d", lr, hidden);
                Path runDir = logDir.resolve(runName);
                Files.createDirectories(runDir);

                Regressor net = new Regressor(16, hidden);
                net.train(true);
                Adam opt = new Adam(net.parameters(), new AdamOptions(lr));

                float finalLoss;
                try (SummaryWriter w = new SummaryWriter(runDir.toString())) {
                    long step = 0;
                    float last = 0;
                    for (int i = 0; i < 40; i++, step++) {
                        Tensor x = randn(new long[]{64, 16});
                        // y ≈ 0.1 * sum(x) + noise
                        Tensor y = x.sum(new long[]{1}, true, new ScalarTypeOptional())
                                .mul(sc(0.1))
                                .add(randn(new long[]{64, 1}).mul(sc(0.01)));

                        opt.zero_grad();
                        Tensor pred = net.forward(x);
                        Tensor loss = mse_loss(pred, y);
                        loss.backward();
                        opt.step();

                        w.add_scalar("train/mse", loss, step);
                        if (step % 10 == 0) {
                            w.add_histogram("fc1.weight", net.fc1.weight(), step);
                            w.add_histogram("fc2.weight", net.fc2.weight(), step);
                        }
                        last = loss.item_float();
                    }
                    finalLoss = last;

                    Map<String, Object> hp = new LinkedHashMap<>();
                    hp.put("lr", lr);
                    hp.put("hidden", (double) hidden);
                    hp.put("opt", "adam");
                    Map<String, Number> metrics = new LinkedHashMap<>();
                    metrics.put("hparam/mse", finalLoss);
                    w.add_hparams(hp, metrics, "hp_" + runName, 40L);
                    w.flush();
                }
                System.out.printf("  [reg] %s  final mse=%.6f%n", runName, finalLoss);
            }
        }
        check("regressor runs written", Files.list(logDir).findAny().isPresent());
    }

    // =========================================================================
    // 3) Binary classifier → PR / embedding / audio
    // =========================================================================

    static void trainBinaryNet(Path logDir) throws Exception {
        Files.createDirectories(logDir);
        final long N = 128;
        final long D = 16;
        BinaryNet net = new BinaryNet(D, 32);
        net.train(true);
        SGD opt = new SGD(net.parameters(), new SGDOptions(0.1));

        try (SummaryWriter w = new SummaryWriter(logDir.toString())) {
            long step = 0;
            Tensor probeX = null;
            Tensor probeY = null;

            for (int i = 0; i < 50; i++, step++) {
                // labels in {0,1} as float for BCE
                Tensor yLong = randint(2, new long[]{N}, longOpts());
                Tensor y = yLong.to(ScalarType.Float);
                Tensor noise = randn(new long[]{N, D});
                Tensor mean = y.unsqueeze(1).mul(sc(1.5f));
                Tensor x = noise.add(mean);

                opt.zero_grad();
                Tensor probs = net.forward(x);
                Tensor loss = binary_cross_entropy(probs, y);
                loss.backward();
                opt.step();

                w.add_scalar("train/bce", loss, step);

                if (step % 10 == 0) {
                    w.add_pr_curve("train/pr", y, probs, step);
                    w.add_histogram("fc1.weight", net.fc1.weight(), step);
                }
                if (step == 49) {
                    probeX = x;
                    probeY = y;
                }
            }

            if (probeX != null) {
                Tensor feats = relu(net.fc1.forward(probeX)); // [N,32]
                List<String> meta = new ArrayList<>();
                Tensor yCpu = probeY.contiguous().cpu();
                for (int i = 0; i < (int) N; i++) {
                    meta.add(yCpu.select(0, i).item_float() > 0.5f ? "pos" : "neg");
                }
                w.add_embedding(feats, meta, 50L, "hidden");
            }

            // Meaningful audio: C-major scale + minor chord + short melody (not a bare beep)
            int sr = 16000;
            w.add_audio("audio/c_major_scale", makeScaleAudio(sr, /*bpm*/120), 50L, sr);
            w.add_audio("audio/a_minor_chord", makeChordAudio(sr, new double[]{220.0, 261.63, 329.63}, 1.2), 50L, sr);
            w.add_audio("audio/twinkle_phrase", makeMelodyAudio(sr, new double[]{
                    261.63, 261.63, 392.00, 392.00, 440.00, 440.00, 392.00, // C C G G A A G
                    349.23, 349.23, 329.63, 329.63, 293.66, 293.66, 261.63  // F F E E D D C
            }, 0.35), 50L, sr);

            w.add_text("run/note",
                    "binary classifier + pr_curve + embedding + structured audio (scale/chord/melody)", 50);
            w.flush();
            System.out.println("  [binary] done (pr_curve + embedding + audio)");
            check("binary events",
                    Files.list(logDir).anyMatch(p -> p.getFileName().toString().contains("tfevents")));
        }
    }

    // =========================================================================
    // 4) Multimodal showcase — content you can *see* and *hear* in TB
    // =========================================================================

    static void demoMultimodalShowcase(Path logDir) throws Exception {
        Files.createDirectories(logDir);
        try (SummaryWriter w = new SummaryWriter(logDir.toString())) {
            w.add_text("readme",
                    "Multimodal showcase: synthetic-but-recognizable digits, geometry, " +
                    "RGB color bars, musical audio (scale/chord/melody), video frames, mesh.", 0);

            // ---- Images: digits 0-9, geometry, color science ----
            w.add_images("vision/digits_0_9", makeDigitBatch(new int[]{0,1,2,3,4,5,6,7,8,9}, 56), 0L);
            w.add_image("vision/digit_big_3", makeDigitImage(3, 64), 0L);
            w.add_image("vision/digit_big_8", makeDigitImage(8, 64), 0L);
            w.add_image("vision/checker_8x8", makeCheckerboard(64, 8), 0L);
            w.add_image("vision/stripes", makeStripes(64, 64, true), 0L);
            w.add_image("vision/circles", makeConcentricCircles(64), 0L);
            w.add_image("vision/colorbar_hsv", makeColorBar(128, 32), 0L);
            w.add_image("vision/rgb_primaries", makeRgbPrimaries(48), 0L);
            w.add_images("vision/geometry_set", makeGeometrySet(48), 0L);

            // step progression: digit morphing 0→9 across steps (like training viz)
            for (int s = 0; s <= 9; s++) {
                w.add_image("vision/digit_over_steps", makeDigitImage(s, 48), s);
            }

            // ---- Audio: structured musical content @ 16 kHz ----
            int sr = 16000;
            w.add_audio("sound/c_major_scale", makeScaleAudio(sr, 100), 0L, sr);
            w.add_audio("sound/a_minor_chord", makeChordAudio(sr, new double[]{220.0, 261.63, 329.63}, 1.5), 0L, sr);
            w.add_audio("sound/twinkle", makeMelodyAudio(sr, new double[]{
                    261.63, 261.63, 392.00, 392.00, 440.00, 440.00, 392.00,
                    349.23, 349.23, 329.63, 329.63, 293.66, 293.66, 261.63
            }, 0.30), 0L, sr);
            w.add_audio("sound/dtmf_911", makeDtmfSequence(sr, new int[]{9, 1, 1}, 0.25, 0.08), 0L, sr);
            w.add_audio("sound/sweep_chirp", makeChirp(sr, 200, 4000, 1.5), 0L, sr);

            // ---- Video-like frame sequence (rotating bar) ----
            w.add_video("video/rotating_bar", makeRotatingBarFrames(16, 48), 0L, 8);

            // ---- Mesh: structured sphere-ish point cloud (not pure noise) ----
            Tensor sphere = makeSphereCloud(400);
            Tensor scolors = sphereToColors(sphere);
            w.add_mesh("mesh/sphere", sphere, scolors, null, 0L);

            // ---- Embedding of digit features (2D layout by class) ----
            float[][] emb = new float[50][2];
            List<String> meta = new ArrayList<>();
            for (int i = 0; i < 50; i++) {
                int cls = i % 10;
                double ang = cls * (2 * Math.PI / 10) + (i / 10) * 0.15;
                double r = 0.6 + 0.05 * (i % 5);
                emb[i][0] = (float) (r * Math.cos(ang));
                emb[i][1] = (float) (r * Math.sin(ang));
                meta.add("digit_" + cls);
            }
            Tensor embT = tensor(flatten(emb)).reshape(50, 2);
            w.add_embedding(embT, meta, 0L, "digits_2d");

            w.add_scalar("showcase/ready", 1.0, 0);
            w.flush();
            check("multimodal events",
                    Files.list(logDir).anyMatch(p -> p.getFileName().toString().contains("tfevents")));
            System.out.println("  [multimodal] digits/geometry/colorbar/audio/video/mesh/embedding written");
        }
    }

    // =========================================================================
    // 5) mesh + layout (kept small)
    // =========================================================================


    // =========================================================================
    // Heatmap + Embedding projector (sprite) — TB Images + Projector tabs
    // =========================================================================

    static void demoHeatmapAndEmbedding(Path logDir) throws Exception {
        Files.createDirectories(logDir);
        final int nClass = 10;
        final int perClass = 20;          // 200 points — enough for projector
        final int n = nClass * perClass;  // 200
        final int imgSize = 28;
        final int featDim = 16;

        try (SummaryWriter w = new SummaryWriter(logDir.toString())) {
            w.add_text("readme",
                    "Heatmaps (confusion / attention / activation) via add_heatmap + " +
                    "Embedding Projector via add_embedding(mat, metadata, label_img, step, tag).", 0);

            // ---- 1) Confusion-matrix style heatmap (mostly diagonal) ----
            float[][] conf = new float[nClass][nClass];
            for (int i = 0; i < nClass; i++) {
                for (int j = 0; j < nClass; j++) {
                    conf[i][j] = (i == j) ? 18f + (i % 3) : (Math.abs(i - j) == 1 ? 2.5f : 0.4f);
                }
            }
            // row-normalize-ish peak on diagonal is visible under viridis/jet
            Tensor confT = tensor(flatten(conf)).reshape(nClass, nClass);
            w.add_heatmap("heatmap/confusion_viridis", confT, 0L, "viridis");
            w.add_heatmap("heatmap/confusion_jet", confT, 0L, "jet");
            w.add_heatmap("heatmap/confusion_hot", confT, 0L, "hot");
            w.add_heatmap("heatmap/confusion_gray", confT, 0L, "gray");

            // ---- 2) Attention-like map (Gaussian blob drifting over steps) ----
            for (int step = 0; step < 8; step++) {
                Tensor attn = makeGaussianHeatmap(32, 32,
                        0.25 + 0.5 * step / 7.0,
                        0.3 + 0.4 * ((step % 4) / 3.0),
                        0.18);
                w.add_heatmap("heatmap/attention_over_steps", attn, step, "viridis");
            }

            // ---- 3) Activation map from a digit (energy of glyph as heatmap) ----
            Tensor digit = makeDigitImage(5, 56); // CHW 1x56x56
            Tensor act = digit.squeeze(0);       // HW
            w.add_heatmap("heatmap/digit5_activation", act, 0L, "hot");
            w.add_image("heatmap/digit5_raw", digit, 0L);

            // ---- 4) Embedding with metadata + sprite label images ----
            // Features: class-conditional clusters in 16-D (first 2 dims form a ring)
            float[][] mat = new float[n][featDim];
            int[] labels = new int[n];
            List<String> meta = new ArrayList<>();
            List<List<String>> metaRows = new ArrayList<>();
            float[] spritesFlat = new float[n * imgSize * imgSize]; // N*1*H*W grayscale planar later

            java.util.Random rng = new java.util.Random(0);
            for (int i = 0; i < n; i++) {
                int cls = i % nClass;
                labels[i] = cls;
                double ang = cls * (2 * Math.PI / nClass) + 0.05 * rng.nextGaussian();
                double rad = 1.2 + 0.08 * rng.nextGaussian();
                mat[i][0] = (float) (rad * Math.cos(ang));
                mat[i][1] = (float) (rad * Math.sin(ang));
                for (int d = 2; d < featDim; d++) {
                    mat[i][d] = 0.15f * (float) rng.nextGaussian() + 0.02f * cls;
                }
                meta.add("digit_" + cls);
                metaRows.add(List.of("digit_" + cls, Integer.toString(cls), i < n / 2 ? "train" : "val"));

                // sprite glyph
                Tensor g = makeDigitImage(cls, imgSize);
                float[] gf = toJavaFloat(g);
                System.arraycopy(gf, 0, spritesFlat, i * imgSize * imgSize, gf.length);
            }

            Tensor emb = tensor(flatten(mat)).reshape(n, featDim);
            Tensor labelImg = tensor(spritesFlat).reshape(n, 1, imgSize, imgSize); // NCHW

            // multi-column metadata header
            w.add_embedding(emb, metaRows, labelImg, 0L, "digits_sprite",
                    List.of("label", "class_id", "split"));
            // also a plain single-column embedding (no sprite) for comparison
            w.add_embedding(emb, meta, 1L, "digits_plain");

            // scalar so the run shows up under Scalars too
            w.add_scalar("embed/n_points", n, 0);
            w.add_scalar("heatmap/confusion_trace", confT.diagonal().sum(), 0);
            w.flush();

            // basic on-disk checks for projector assets
            Path pbtxt = logDir.resolve("projector_config.pbtxt");
            Path sprite = logDir.resolve("00000").resolve("digits_sprite").resolve("sprite.png");
            Path tsv = logDir.resolve("00000").resolve("digits_sprite").resolve("tensors.tsv");
            Path metaPath = logDir.resolve("00000").resolve("digits_sprite").resolve("metadata.tsv");
            check("projector_config.pbtxt", Files.exists(pbtxt));
            check("sprite.png", Files.exists(sprite) && Files.size(sprite) > 100);
            check("tensors.tsv", Files.exists(tsv) && Files.size(tsv) > 100);
            check("metadata.tsv", Files.exists(metaPath));
            String pbtxtText = Files.readString(pbtxt);
            check("pbtxt has sprite", pbtxtText.contains("sprite") && pbtxtText.contains("single_image_dim"));
            check("pbtxt has tensor_path", pbtxtText.contains("tensors.tsv"));
            System.out.println("  [heatmap_embed] confusion/attention heatmaps + projector sprite written");
            System.out.println("  sprite=" + sprite.toAbsolutePath() + " (" + Files.size(sprite) + " bytes)");
        }
    }

    /** Isotropic Gaussian blob heatmap on [0,1] grid, peak at (cx,cy) in relative coords. */
    static Tensor makeGaussianHeatmap(int h, int w, double cx, double cy, double sigma) {
        float[] data = new float[h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double u = x / (double) (w - 1);
                double v = y / (double) (h - 1);
                double d = (u - cx) * (u - cx) + (v - cy) * (v - cy);
                data[y * w + x] = (float) Math.exp(-d / (2 * sigma * sigma));
            }
        }
        return tensor(data).reshape(h, w);
    }


    static void demoMeshAndLayout(Path logDir) throws Exception {
        Files.createDirectories(logDir);
        try (SummaryWriter w = new SummaryWriter(logDir.toString())) {
            Tensor sphere = makeSphereCloud(300);
            w.add_mesh("cloud", sphere, sphereToColors(sphere), null, 0L);
            w.add_custom_scalars_multilinechart(
                    List.of("train/loss", "train/bce", "train/mse"),
                    "AllRuns", "losses");
            w.flush();
            check("viz events",
                    Files.list(logDir).anyMatch(p -> p.getFileName().toString().contains("tfevents")));
            System.out.println("  [viz] mesh + custom_scalars layout");
        }
    }

    // =========================================================================
    // Content generators (recognizable vision / audio)
    // =========================================================================

    /** 5x7 stroke fonts for digits 0-9 (1 = ink). */
    private static final int[][][] DIGIT_FONT = {
        {{1,1,1},{1,0,1},{1,0,1},{1,0,1},{1,1,1}}, // 0
        {{0,1,0},{1,1,0},{0,1,0},{0,1,0},{1,1,1}}, // 1
        {{1,1,1},{0,0,1},{1,1,1},{1,0,0},{1,1,1}}, // 2
        {{1,1,1},{0,0,1},{1,1,1},{0,0,1},{1,1,1}}, // 3
        {{1,0,1},{1,0,1},{1,1,1},{0,0,1},{0,0,1}}, // 4
        {{1,1,1},{1,0,0},{1,1,1},{0,0,1},{1,1,1}}, // 5
        {{1,1,1},{1,0,0},{1,1,1},{1,0,1},{1,1,1}}, // 6
        {{1,1,1},{0,0,1},{0,1,0},{0,1,0},{0,1,0}}, // 7
        {{1,1,1},{1,0,1},{1,1,1},{1,0,1},{1,1,1}}, // 8
        {{1,1,1},{1,0,1},{1,1,1},{0,0,1},{1,1,1}}, // 9
    };

    /** CHW float [0,1] grayscale digit, size x size. */
    static Tensor makeDigitImage(int digit, int size) {
        digit = ((digit % 10) + 10) % 10;
        float[] hw = new float[size * size];
        int[][] g = DIGIT_FONT[digit];
        int rows = g.length, cols = g[0].length;
        int margin = size / 8;
        int gh = size - 2 * margin, gw = size - 2 * margin;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                float v = 0.08f; // dark bg
                if (y >= margin && y < size - margin && x >= margin && x < size - margin) {
                    int gy = (y - margin) * rows / gh;
                    int gx = (x - margin) * cols / gw;
                    if (gy >= rows) gy = rows - 1;
                    if (gx >= cols) gx = cols - 1;
                    if (g[gy][gx] == 1) v = 1.0f;
                }
                hw[y * size + x] = v;
            }
        }
        return tensor(hw).reshape(1, size, size); // CHW
    }

    /** NCHW batch of digits (optionally already clean). */
    static Tensor makeDigitBatch(int[] labels, int size) {
        int n = labels.length;
        float[] all = new float[n * size * size];
        for (int i = 0; i < n; i++) {
            Tensor one = makeDigitImage(labels[i], size);
            float[] flat = toJavaFloat(one);
            System.arraycopy(flat, 0, all, i * size * size, flat.length);
        }
        return tensor(all).reshape(n, 1, size, size);
    }

    static Tensor makeCheckerboard(int size, int cell) {
        float[] rgb = new float[3 * size * size];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                boolean on = ((x / cell) + (y / cell)) % 2 == 0;
                float v = on ? 0.95f : 0.15f;
                int i = y * size + x;
                rgb[0 * size * size + i] = v;
                rgb[1 * size * size + i] = v;
                rgb[2 * size * size + i] = v * 0.85f;
            }
        }
        return tensor(rgb).reshape(3, size, size);
    }

    static Tensor makeStripes(int h, int w, boolean vertical) {
        float[] rgb = new float[3 * h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int t = vertical ? x : y;
                float phase = (t % 8) / 8.0f;
                int i = y * w + x;
                rgb[0 * h * w + i] = phase;
                rgb[1 * h * w + i] = 1.0f - phase;
                rgb[2 * h * w + i] = 0.4f;
            }
        }
        return tensor(rgb).reshape(3, h, w);
    }

    static Tensor makeConcentricCircles(int size) {
        float[] rgb = new float[3 * size * size];
        float cx = (size - 1) / 2.0f, cy = (size - 1) / 2.0f;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double r = Math.hypot(x - cx, y - cy) / (size / 2.0);
                double band = (Math.sin(r * 12) + 1) * 0.5;
                int i = y * size + x;
                rgb[0 * size * size + i] = (float) band;
                rgb[1 * size * size + i] = (float) (1.0 - band);
                rgb[2 * size * size + i] = (float) (0.3 + 0.4 * r);
            }
        }
        return tensor(rgb).reshape(3, size, size);
    }

    /** Horizontal HSV-ish rainbow color bar, CHW RGB. */
    static Tensor makeColorBar(int width, int height) {
        float[] rgb = new float[3 * height * width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float hue = x / (float) (width - 1); // 0..1
                float[] c = hsvToRgb(hue, 1.0f, 1.0f);
                int i = y * width + x;
                rgb[0 * height * width + i] = c[0];
                rgb[1 * height * width + i] = c[1];
                rgb[2 * height * width + i] = c[2];
            }
        }
        return tensor(rgb).reshape(3, height, width);
    }

    static Tensor makeRgbPrimaries(int size) {
        // 3 side-by-side blocks: R, G, B
        int w = size * 3;
        float[] rgb = new float[3 * size * w];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < w; x++) {
                int block = x / size;
                int i = y * w + x;
                rgb[0 * size * w + i] = block == 0 ? 1f : 0f;
                rgb[1 * size * w + i] = block == 1 ? 1f : 0f;
                rgb[2 * size * w + i] = block == 2 ? 1f : 0f;
            }
        }
        return tensor(rgb).reshape(3, size, w);
    }

    static Tensor makeGeometrySet(int size) {
        // N=4: checker, stripes, circles, colorbar-as-square crop
        Tensor a = makeCheckerboard(size, Math.max(2, size / 8)).unsqueeze(0);
        Tensor b = makeStripes(size, size, true).unsqueeze(0);
        Tensor c = makeConcentricCircles(size).unsqueeze(0);
        Tensor d = makeColorBar(size, size).unsqueeze(0);
        return cat(new org.bytedeco.pytorch.TensorVector(a, b, c, d), 0);
    }

    static Tensor makeRotatingBarFrames(int frames, int size) {
        // TCHW
        float[] all = new float[frames * 3 * size * size];
        float cx = (size - 1) / 2.0f, cy = (size - 1) / 2.0f;
        for (int t = 0; t < frames; t++) {
            double ang = t * Math.PI / frames;
            double ca = Math.cos(ang), sa = Math.sin(ang);
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    double dx = x - cx, dy = y - cy;
                    double proj = Math.abs(dx * sa - dy * ca);
                    float v = proj < size / 10.0 ? 1f : 0.1f;
                    int i = t * 3 * size * size + y * size + x;
                    all[i] = v;
                    all[i + size * size] = v * 0.6f;
                    all[i + 2 * size * size] = 1f - v * 0.5f;
                }
            }
        }
        return tensor(all).reshape(frames, 3, size, size);
    }

    // ---- audio generators ---------------------------------------------------

    static Tensor makeScaleAudio(int sr, int bpm) {
        // C4..C5 major scale
        double[] freqs = {261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25};
        double beat = 60.0 / bpm;
        return makeMelodyAudio(sr, freqs, beat * 0.85);
    }

    static Tensor makeMelodyAudio(int sr, double[] freqsHz, double noteSec) {
        int noteSamples = Math.max(1, (int) (sr * noteSec));
        int gap = (int) (sr * 0.02);
        int total = freqsHz.length * (noteSamples + gap);
        float[] pcm = new float[total];
        int off = 0;
        for (double f : freqsHz) {
            for (int i = 0; i < noteSamples; i++) {
                double t = i / (double) sr;
                double env = attackRelease(i, noteSamples, 0.02, 0.08);
                // soft square-ish: fundamental + odd partials (more "instrument-like")
                double s = Math.sin(2 * Math.PI * f * t)
                        + 0.35 * Math.sin(2 * Math.PI * 2 * f * t)
                        + 0.18 * Math.sin(2 * Math.PI * 3 * f * t);
                pcm[off + i] = (float) (0.25 * env * s);
            }
            off += noteSamples + gap;
        }
        return tensor(pcm);
    }

    static Tensor makeChordAudio(int sr, double[] freqsHz, double seconds) {
        int n = Math.max(1, (int) (sr * seconds));
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) {
            double t = i / (double) sr;
            double env = attackRelease(i, n, 0.05, 0.3);
            double s = 0;
            for (double f : freqsHz) s += Math.sin(2 * Math.PI * f * t);
            pcm[i] = (float) (0.2 * env * s / freqsHz.length);
        }
        return tensor(pcm);
    }

    /** DTMF dial tones (phone keypad) — instantly recognizable. */
    static Tensor makeDtmfSequence(int sr, int[] digits, double toneSec, double gapSec) {
        // DTMF freq pairs
        double[] low = {697, 697, 697, 770, 770, 770, 852, 852, 852, 941};
        double[] high = {1209, 1336, 1477, 1209, 1336, 1477, 1209, 1336, 1477, 1336};
        // map digit 0..9 → index; 0 is last in standard table
        int toneN = Math.max(1, (int) (sr * toneSec));
        int gapN = Math.max(0, (int) (sr * gapSec));
        int total = digits.length * (toneN + gapN);
        float[] pcm = new float[total];
        int off = 0;
        for (int d : digits) {
            int idx = d == 0 ? 9 : d - 1;
            if (idx < 0 || idx > 9) idx = 0;
            double f1 = low[idx], f2 = high[idx];
            for (int i = 0; i < toneN; i++) {
                double t = i / (double) sr;
                double env = attackRelease(i, toneN, 0.01, 0.05);
                pcm[off + i] = (float) (0.25 * env * (Math.sin(2 * Math.PI * f1 * t) + Math.sin(2 * Math.PI * f2 * t)));
            }
            off += toneN + gapN;
        }
        return tensor(pcm);
    }

    static Tensor makeChirp(int sr, double f0, double f1, double seconds) {
        int n = Math.max(1, (int) (sr * seconds));
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) {
            double t = i / (double) sr;
            double frac = t / seconds;
            double f = f0 + (f1 - f0) * frac;
            // phase integration approx
            double phase = 2 * Math.PI * (f0 * t + 0.5 * (f1 - f0) * frac * t);
            double env = attackRelease(i, n, 0.02, 0.1);
            pcm[i] = (float) (0.3 * env * Math.sin(phase));
        }
        return tensor(pcm);
    }

    private static double attackRelease(int i, int n, double atkSec, double relSec) {
        // approximate with sample counts later — callers pass seconds relative to sr externally;
        // here treat atkSec/relSec as fractions of n when < 1, else ignore.
        int atk = Math.max(1, (int) (n * Math.min(atkSec, 0.5)));
        int rel = Math.max(1, (int) (n * Math.min(relSec, 0.5)));
        double e = 1.0;
        if (i < atk) e = i / (double) atk;
        int back = n - 1 - i;
        if (back < rel) e = Math.min(e, back / (double) rel);
        return e;
    }

    // ---- mesh / embedding helpers ------------------------------------------

    static Tensor makeSphereCloud(int n) {
        float[] v = new float[n * 3];
        for (int i = 0; i < n; i++) {
            // fibonacci sphere
            double y = 1.0 - (i / (double) (n - 1)) * 2.0;
            double r = Math.sqrt(Math.max(0, 1 - y * y));
            double theta = Math.PI * (3.0 - Math.sqrt(5.0)) * i;
            v[i * 3] = (float) (Math.cos(theta) * r);
            v[i * 3 + 1] = (float) y;
            v[i * 3 + 2] = (float) (Math.sin(theta) * r);
        }
        return tensor(v).reshape(1, n, 3);
    }

    static Tensor sphereToColors(Tensor sphereBN3) {
        // map xyz in [-1,1] → RGB 0..255
        return sphereBN3.add(sc(1)).mul(sc(127.5));
    }

    // ---- small utils --------------------------------------------------------

    static long[] toLongLabels(int[] labels) {
        long[] y = new long[labels.length];
        for (int i = 0; i < labels.length; i++) y[i] = labels[i];
        return y;
    }

    static float[] flatten(float[][] rows) {
        int r = rows.length, c = rows[0].length;
        float[] flat = new float[r * c];
        for (int i = 0; i < r; i++) System.arraycopy(rows[i], 0, flat, i * c, c);
        return flat;
    }

    static float[] toJavaFloat(Tensor t) {
        Tensor c = t.contiguous().cpu().to(ScalarType.Float).flatten();
        long n = c.numel();
        float[] out = new float[(int) n];
        org.bytedeco.javacpp.FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < out.length; i++) out[i] = p.get(i);
        return out;
    }

    static float[] hsvToRgb(float h, float s, float v) {
        float c = v * s;
        float x = c * (1 - Math.abs((h * 6) % 2 - 1));
        float m = v - c;
        float r, g, b;
        int sector = (int) (h * 6);
        switch (sector) {
            case 0 -> { r = c; g = x; b = 0; }
            case 1 -> { r = x; g = c; b = 0; }
            case 2 -> { r = 0; g = c; b = x; }
            case 3 -> { r = 0; g = x; b = c; }
            case 4 -> { r = x; g = 0; b = c; }
            default -> { r = c; g = 0; b = x; }
        }
        return new float[]{r + m, g + m, b + m};
    }

    static void logParameterHistograms(SummaryWriter w, Module net, long step) throws Exception {
        StringTensorDict named = net.named_parameters();
        long n = named.size();
        for (long i = 0; i < n; i++) {
            StringTensorDictItem item = named.get(i);
            String key = item.key().getString();
            w.add_histogram("params/" + key, item.value(), step);
        }
    }

    static void verifyWithPython(Path logRoot) throws Exception {
        String py =
                "from tensorboard.backend.event_processing import event_accumulator\n" +
                "import os\n" +
                "root = r'" + logRoot.toAbsolutePath() + "'\n" +
                "ok = 0\n" +
                "for dirpath, dirs, files in os.walk(root):\n" +
                "  if any('tfevents' in f for f in files):\n" +
                "    ea = event_accumulator.EventAccumulator(dirpath)\n" +
                "    ea.Reload()\n" +
                "    tags = ea.Tags()\n" +
                "    useful = {k: (len(v) if isinstance(v, list) else v) for k, v in tags.items() if v}\n" +
                "    if useful:\n" +
                "      ok += 1\n" +
                "      print('OK', dirpath, useful)\n" +
                "print('RUNS', ok)\n" +
                "assert ok >= 3, 'expected >=3 readable runs'\n" +
                "print('PASS')\n";
        Path script = logRoot.resolve("_verify.py");
        Files.writeString(script, py);
        ProcessBuilder pb = new ProcessBuilder("python3", script.toString());
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes());
        int code = p.waitFor();
        System.out.println(out.trim());
        check("python EA PASS", code == 0 && out.contains("PASS"));
    }

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        try {
            r.run();
            System.out.println("  ✓ " + name);
            passed++;
        } catch (Throwable t) {
            failed++;
            report.append("  FAIL [").append(name).append("]: ").append(t.getMessage()).append("\n");
            System.out.println("  ✗ " + name + " — " + t);
            t.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean condition) {
        if (condition) passed++;
        else {
            failed++;
            report.append("  CHECK FAILED: ").append(name).append("\n");
            throw new AssertionError(name);
        }
    }
}
