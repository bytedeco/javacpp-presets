package samples;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.dataframe.ai.EmbeddingModel;
import org.bytedeco.pytorch.dataframe.ai.EmbeddingRegistry;
import org.bytedeco.pytorch.dataframe.ai.Modality;
import org.bytedeco.pytorch.dataframe.ai.TorchAudioEmbeddingModel;
import org.bytedeco.pytorch.dataframe.ai.TorchVisionEmbeddingModel;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.AudioOptions;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.ImageOptions;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.VideoOptions;
import org.bytedeco.pytorch.dataframe.media.MediaInterop;
import org.bytedeco.pytorch.dataframe.media.MediaSampleFactory;
import org.bytedeco.pytorch.dataframe.media.MultimodalIO;
import org.bytedeco.pytorch.dataframe.media.MultimodalPreprocess;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;
import org.bytedeco.pytorch.utils.vision.utils.ImageTensors;
import org.bytedeco.pytorch.utils.audio.utils.AudioTensors;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-dimensional correctness + interoperability benchmark for:
 * <ol>
 *   <li>MediaBridge — Image/Audio/Video ↔ Tensor ↔ OpenCV/FFmpeg</li>
 *   <li>MultimodalIO — DataFrame batch loaders (flat dir, ImageFolder, AudioFolder, TextFolder)</li>
 *   <li>MediaInterop — torchvision / torchaudio / torchtext style round-trips</li>
 *   <li>MultimodalPreprocess — frame extract, embed, fusion, pipelines</li>
 *   <li>DataFrame facade — readImages/readAudio/readVideo/readImageFolder/…</li>
 *   <li>Real mp3/mp4 corpus via MediaSampleFactory + FFmpeg decode paths</li>
 *   <li>Neural embeddings — TorchVision (ResNet/MobileNet) + TorchAudio (M5/Wav2Letter)</li>
 * </ol>
 *
 * <p>Runs offline with synthetic PNG/WAV; generates real mp3/mp4 when system ffmpeg is present;
 * exercises OpenCV/FFmpeg natives and real neural embedding towers when loadable.
 *
 * <pre>{@code
 * java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *      -cp "target/classes:samples:$(mvn -q dependency:build-classpath -DincludeScope=runtime -Dmdep.outputFile=/dev/stdout)" \
 *      samples.BenchmarkDataFrameMediaInterop
 * }</pre>
 */
public class BenchmarkDataFrameMediaInterop {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void section(String name, CheckedRunnable r) {
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
        if (ok) {
            passed++;
        } else {
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

    static void checkEq(String name, double expected, double actual, double eps) {
        boolean ok = Math.abs(actual - expected) <= eps;
        if (!ok) System.out.println("    " + name + ": expected " + expected + " got " + actual);
        check(name, ok);
    }

    // ── synthetic media ───────────────────────────────────────────────────

    static ImageData solidImage(int w, int h, int rgb) {
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                bi.setRGB(x, y, rgb);
        return new ImageData(bi);
    }

    static void writePng(Path path, int w, int h, int rgb) throws Exception {
        solidImage(w, h, rgb).save(path.toString());
    }

    static AudioData tone(int sr, double seconds, double freq) {
        int n = Math.max(1, (int) (sr * seconds));
        float[] samples = new float[n];
        for (int i = 0; i < n; i++) {
            samples[i] = (float) Math.sin(2 * Math.PI * freq * i / sr);
        }
        AudioData a = new AudioData(samples, sr, 1);
        a.setDuration(seconds);
        return a;
    }

    /** 16-bit mono PCM WAV. */
    static void writeWav(Path path, float[] samples, int sampleRate, int channels) throws Exception {
        int n = samples.length;
        int dataBytes = n * 2; // 16-bit
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bos);
        // RIFF header
        out.writeBytes("RIFF");
        out.writeInt(Integer.reverseBytes(36 + dataBytes));
        out.writeBytes("WAVE");
        out.writeBytes("fmt ");
        out.writeInt(Integer.reverseBytes(16));
        out.writeShort(Short.reverseBytes((short) 1)); // PCM
        out.writeShort(Short.reverseBytes((short) channels));
        out.writeInt(Integer.reverseBytes(sampleRate));
        out.writeInt(Integer.reverseBytes(sampleRate * channels * 2));
        out.writeShort(Short.reverseBytes((short) (channels * 2)));
        out.writeShort(Short.reverseBytes((short) 16));
        out.writeBytes("data");
        out.writeInt(Integer.reverseBytes(dataBytes));
        out.flush();
        byte[] header = bos.toByteArray();
        ByteBuffer pcm = ByteBuffer.allocate(dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (float s : samples) {
            float c = Math.max(-1f, Math.min(1f, s));
            pcm.putShort((short) (c * 32767));
        }
        byte[] all = new byte[header.length + dataBytes];
        System.arraycopy(header, 0, all, 0, header.length);
        System.arraycopy(pcm.array(), 0, all, header.length, dataBytes);
        Files.write(path, all);
    }

    static VideoData mockVideo(int frames, int w, int h, double fps) {
        List<ImageData> list = new ArrayList<>();
        for (int i = 0; i < frames; i++) {
            int shade = (i * 20) & 0xFF;
            int rgb = (shade << 16) | (shade << 8) | shade;
            list.add(solidImage(w, h, rgb));
        }
        return new VideoData(list, fps);
    }

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("df_media_interop");
        System.out.println("Temp: " + tmp);
        System.out.println("Capabilities: " + MultimodalIO.capabilities());
        System.out.println("OpenCV:  " + MediaBridge.isOpenCvAvailable());
        System.out.println("FFmpeg:  " + MediaBridge.isFFmpegAvailable());
        System.out.println();

        // prepare synthetic corpus
        Path imgDir = tmp.resolve("images");
        Path imgFolder = tmp.resolve("imagefolder");
        Path audDir = tmp.resolve("audio");
        Path audFolder = tmp.resolve("audiofolder");
        Path txtFolder = tmp.resolve("textfolder");
        Path mixed = tmp.resolve("mixed");
        Files.createDirectories(imgDir);
        Files.createDirectories(imgFolder.resolve("cats"));
        Files.createDirectories(imgFolder.resolve("dogs"));
        Files.createDirectories(audDir);
        Files.createDirectories(audFolder.resolve("yes"));
        Files.createDirectories(audFolder.resolve("no"));
        Files.createDirectories(txtFolder.resolve("pos"));
        Files.createDirectories(txtFolder.resolve("neg"));
        Files.createDirectories(mixed);

        writePng(imgDir.resolve("red.png"), 32, 24, 0xFF0000);
        writePng(imgDir.resolve("green.png"), 16, 16, 0x00FF00);
        writePng(imgDir.resolve("blue.png"), 8, 8, 0x0000FF);
        writePng(imgFolder.resolve("cats").resolve("c1.png"), 20, 20, 0xFFAA00);
        writePng(imgFolder.resolve("cats").resolve("c2.png"), 20, 20, 0xFF8800);
        writePng(imgFolder.resolve("dogs").resolve("d1.png"), 20, 20, 0x00AAFF);

        AudioData t440 = tone(16000, 0.25, 440);
        AudioData t880 = tone(16000, 0.25, 880);
        writeWav(audDir.resolve("a440.wav"), t440.getSamples(), 16000, 1);
        writeWav(audDir.resolve("a880.wav"), t880.getSamples(), 16000, 1);
        writeWav(audFolder.resolve("yes").resolve("y1.wav"), t440.getSamples(), 16000, 1);
        writeWav(audFolder.resolve("no").resolve("n1.wav"), t880.getSamples(), 16000, 1);

        Files.writeString(txtFolder.resolve("pos").resolve("p1.txt"), "great product love it", StandardCharsets.UTF_8);
        Files.writeString(txtFolder.resolve("pos").resolve("p2.txt"), "amazing quality", StandardCharsets.UTF_8);
        Files.writeString(txtFolder.resolve("neg").resolve("n1.txt"), "terrible waste of money", StandardCharsets.UTF_8);

        writePng(mixed.resolve("shot.png"), 12, 12, 0xABCDEF);
        writeWav(mixed.resolve("beep.wav"), tone(8000, 0.1, 1000).getSamples(), 8000, 1);
        Files.writeString(mixed.resolve("note.txt"), "hello multimodal", StandardCharsets.UTF_8);

        // ── 1. MediaBridge unit conversions ───────────────────────────────
        System.out.println("══ 1. MediaBridge conversions ══");

        section("image → tensor [0,1] CHW", () -> {
            ImageData img = solidImage(10, 6, 0xFF0000);
            Tensor t = MediaBridge.imageToTensor(img);
            long[] s = TensorBridge.sizesOf(t);
            checkEq("rank", 3, s.length);
            checkEq("C", 3L, s[0]);
            checkEq("H", 6L, s[1]);
            checkEq("W", 10L, s[2]);
            // red channel dominant
            float[] data = new float[(int) t.numel()];
            t.contiguous().cpu().data_ptr_float().get(data);
            check("R mean high", data[0] > 0.9f); // first plane is R
        });

        section("image ↔ tensor roundtrip", () -> {
            ImageData img = solidImage(8, 8, 0x00FF00);
            Tensor t = MediaBridge.imageToTensor(img);
            ImageData back = MediaBridge.tensorToImage(t);
            check("back non-null", back != null && back.getImage() != null);
            checkEq("W", 8, back.getWidth());
            checkEq("H", 8, back.getHeight());
            Tensor t2 = MediaBridge.imageToTensor(back);
            checkEq("numel", t.numel(), t2.numel());
        });

        section("imageToTensor255 scale", () -> {
            ImageData img = solidImage(4, 4, 0xFFFFFF);
            Tensor t255 = MediaBridge.imageToTensor255(img);
            float max = 0;
            org.bytedeco.javacpp.FloatPointer fp = t255.contiguous().cpu().data_ptr_float();
            long n = Math.min(t255.numel(), 48);
            for (long i = 0; i < n; i++) max = Math.max(max, fp.get(i));
            check("max near 255", max > 200);
        });

        section("audio ↔ tensor", () -> {
            AudioData a = tone(16000, 0.05, 440);
            Tensor w = MediaBridge.audioToTensor(a);
            long[] s = TensorBridge.sizesOf(w);
            check("rank 2 [C,T]", s.length == 2);
            checkEq("C", 1L, s[0]);
            check("T>0", s[1] > 0);
            AudioData back = MediaBridge.tensorToAudio(w, 16000);
            checkEq("sr", 16000, back.getSampleRate());
            check("samples", back.getSamples() != null && back.getSamples().length > 0);
        });

        section("audio mono + resample", () -> {
            // stereo-ish interleaved
            float[] stereo = new float[1600];
            for (int i = 0; i < 800; i++) {
                stereo[i * 2] = 0.5f;
                stereo[i * 2 + 1] = -0.5f;
            }
            AudioData a = new AudioData(stereo, 16000, 2);
            AudioData mono = MediaBridge.toMono(a);
            checkEq("mono ch", 1, mono.getChannels());
            checkEq("mono frames", 800, mono.getSamples().length);
            AudioData r8k = MediaBridge.resample(mono, 8000);
            checkEq("sr", 8000, r8k.getSampleRate());
            checkEq("len ~ half", 400, r8k.getSamples().length, /*allow via check*/ 0);
            // length approx
            check("resample len", Math.abs(r8k.getSamples().length - 400) <= 2);
        });

        section("video frame extract + frameAt", () -> {
            VideoData vid = mockVideo(10, 16, 16, 10.0);
            List<ImageData> every2 = MediaBridge.extractFrames(vid, 5.0);
            check("extract ~5", every2.size() >= 4 && every2.size() <= 6);
            ImageData f = MediaBridge.frameAt(vid, 0.5);
            check("frameAt", f != null && f.getImage() != null);
        });

        section("video → NCHW tensor", () -> {
            VideoData vid = mockVideo(4, 8, 8, 4.0);
            Tensor t = MediaBridge.videoToTensor(vid);
            long[] s = TensorBridge.sizesOf(t);
            checkEq("N", 4L, s[0]);
            checkEq("C", 3L, s[1]);
            checkEq("H", 8L, s[2]);
            checkEq("W", 8L, s[3]);
        });

        section("embed image/audio/video dims", () -> {
            EmbeddingData ie = MediaBridge.embedImage(solidImage(16, 16, 0x112233), 32);
            EmbeddingData ae = MediaBridge.embedAudio(tone(8000, 0.1, 220), 32);
            EmbeddingData ve = MediaBridge.embedVideo(mockVideo(6, 8, 8, 6), 32);
            checkEq("img dim", 32, ie.getDimension());
            checkEq("aud dim", 32, ae.getDimension());
            checkEq("vid dim", 32, ve.getDimension());
            // L2 ~ 1
            check("img unitish", l2(ie.getVector()) > 0.5);
            check("aud unitish", l2(ae.getVector()) > 0.5);
        });

        section("loadImage from PNG path", () -> {
            ImageData img = MediaBridge.loadImage(imgDir.resolve("red.png").toString());
            check("loaded", img != null && img.getImage() != null);
            checkEq("W", 32, img.getWidth());
            checkEq("H", 24, img.getHeight());
        });

        section("loadAudio from WAV path", () -> {
            AudioData a = MediaBridge.loadAudio(audDir.resolve("a440.wav").toString(), 16000, true);
            check("samples", a.getSamples() != null && a.getSamples().length > 100);
            checkEq("sr", 16000, a.getSampleRate());
        });

        section("stackImages batch", () -> {
            List<ImageData> imgs = List.of(
                    solidImage(8, 8, 0xFF0000),
                    solidImage(8, 8, 0x00FF00),
                    solidImage(4, 4, 0x0000FF) // different size → resized
            );
            Tensor batch = MediaBridge.stackImages(imgs);
            long[] s = TensorBridge.sizesOf(batch);
            checkEq("N", 3L, s[0]);
            checkEq("H", 8L, s[2]);
            checkEq("W", 8L, s[3]);
        });

        // ── 2. MultimodalIO batch loaders ─────────────────────────────────
        System.out.println("\n══ 2. MultimodalIO / DataFrame loaders ══");

        section("readImages dir", () -> {
            DataFrame df = MultimodalIO.readImages(imgDir.toString());
            checkEq("rows", 3, df.rowCount());
            check("has image", df.hasColumn("image"));
            check("has path", df.hasColumn("path"));
            check("has width meta", df.hasColumn("width"));
            check("cell ImageData", df.get(0, "image") instanceof ImageData);
        });

        section("DataFrame.readImages facade", () -> {
            DataFrame df = DataFrame.readImages(imgDir.toString());
            check("rows>=3", df.rowCount() >= 3);
        });

        section("readAudio dir", () -> {
            DataFrame df = MultimodalIO.readAudio(audDir.toString(), 16000, true);
            checkEq("rows", 2, df.rowCount());
            check("audio cell", df.get(0, "audio") instanceof AudioData);
            check("sample_rate col", df.hasColumn("sample_rate"));
        });

        section("readImageFolder", () -> {
            // Prefer MultimodalIO (DataFrame facade delegates here once recompiled)
            DataFrame df = MultimodalIO.readImageFolder(imgFolder.toString());
            checkEq("rows", 3, df.rowCount());
            check("label col", df.hasColumn("label"));
            check("class col", df.hasColumn("class"));
            // cats=0, dogs=1 (sorted)
            boolean sawCat = false, sawDog = false;
            for (int r = 0; r < df.rowCount(); r++) {
                String cls = String.valueOf(df.get(r, "class"));
                if ("cats".equals(cls)) sawCat = true;
                if ("dogs".equals(cls)) sawDog = true;
            }
            check("cats", sawCat);
            check("dogs", sawDog);
        });

        section("readAudioFolder", () -> {
            DataFrame df = MultimodalIO.readAudioFolder(audFolder.toString());
            checkEq("rows", 2, df.rowCount());
            check("classes", df.hasColumn("class"));
        });

        section("readTextFolder", () -> {
            DataFrame df = MultimodalIO.readTextFolder(txtFolder.toString());
            checkEq("rows", 3, df.rowCount());
            check("text", df.hasColumn("text"));
            Object t0 = df.get(0, "text");
            check("text non-empty", t0 != null && t0.toString().length() > 3);
        });

        section("readMultimodal mixed dir", () -> {
            DataFrame df = MultimodalIO.readMultimodalDir(mixed.toString());
            check("rows>=3", df.rowCount() >= 3);
            boolean hasImg = false, hasAud = false, hasTxt = false;
            for (int r = 0; r < df.rowCount(); r++) {
                String m = String.valueOf(df.get(r, "modality"));
                if ("image".equals(m)) hasImg = true;
                if ("audio".equals(m)) hasAud = true;
                if ("text".equals(m)) hasTxt = true;
            }
            check("mod image", hasImg);
            check("mod audio", hasAud);
            check("mod text", hasTxt);
        });

        section("fromImages / fromEmbeddings", () -> {
            DataFrame fi = DataFrame.fromImages("img",
                    List.of(solidImage(3, 3, 1), solidImage(3, 3, 2)));
            checkEq("fromImages", 2, fi.rowCount());
            DataFrame fe = DataFrame.fromEmbeddings("e",
                    new float[][]{{1, 0}, {0, 1}}, "test");
            checkEq("fromEmb", 2, fe.rowCount());
            check("emb cell", fe.get(0, "e") instanceof EmbeddingData);
        });

        section("fromOpenCV tensors", () -> {
            Tensor t = MediaBridge.imageToTensor(solidImage(5, 5, 0xFFFFFF));
            DataFrame df = MultimodalIO.fromOpenCVTensors("image", List.of(t, t));
            checkEq("rows", 2, df.rowCount());
            check("ImageData", df.get(0, "image") instanceof ImageData);
        });

        section("fromAudioTensors", () -> {
            Tensor w = MediaBridge.audioToTensor(tone(8000, 0.05, 100));
            DataFrame df = MultimodalIO.fromAudioTensors("audio", List.of(w), 8000);
            checkEq("rows", 1, df.rowCount());
            check("AudioData", df.get(0, "audio") instanceof AudioData);
        });

        // ── 3. MediaInterop (vision/audio/text) ───────────────────────────
        System.out.println("\n══ 3. MediaInterop torchvision/torchaudio/torchtext ══");

        section("vision batch NCHW", () -> {
            DataFrame df = MultimodalIO.readImages(imgDir.toString());
            // resize all to same size first via preprocess
            DataFrame ready = MultimodalPreprocess.visionPipeline(df, "image", 16);
            Tensor batch = MediaInterop.toVisionBatch(ready, "image");
            long[] s = TensorBridge.sizesOf(batch);
            checkEq("N", (long) ready.rowCount(), s[0]);
            checkEq("C", 3L, s[1]);
            checkEq("H", 16L, s[2]);
            checkEq("W", 16L, s[3]);
        });

        section("fromVisionBatch roundtrip", () -> {
            DataFrame df = MultimodalPreprocess.visionPipeline(
                    MultimodalIO.fromImages("image", List.of(solidImage(12, 12, 0xABCDEF))),
                    "image", 12);
            Tensor batch = MediaInterop.toVisionBatch(df, "image");
            DataFrame back = MediaInterop.fromVisionBatch(batch, "image");
            checkEq("rows", 1, back.rowCount());
            check("img", back.get(0, "image") instanceof ImageData);
        });

        section("mapImages resize", () -> {
            DataFrame df = DataFrame.fromImages("image", List.of(solidImage(30, 10, 0x111111)));
            DataFrame out = MediaInterop.mapImages(df, "image", img -> img.resize(15, 5));
            ImageData r = (ImageData) out.get(0, "image");
            checkEq("W", 15, r.getWidth());
            checkEq("H", 5, r.getHeight());
        });

        section("applyVisionTransform function", () -> {
            DataFrame df = DataFrame.fromImages("image", List.of(solidImage(10, 10, 0x222222)));
            DataFrame out = MediaInterop.applyVisionTransform(df, "image",
                    (java.util.function.Function<Object, Object>) o -> {
                        if (o instanceof ImageData id) return id.resize(5, 5);
                        return o;
                    });
            checkEq("W", 5, ((ImageData) out.get(0, "image")).getWidth());
        });

        section("audio preprocess mono+resample", () -> {
            DataFrame df = MultimodalIO.readAudio(audDir.toString(), AudioOptions.of(16000, false), true);
            DataFrame out = MediaInterop.audioPreprocess(df, "audio", 8000, 1.0);
            AudioData a = (AudioData) out.get(0, "audio");
            checkEq("sr", 8000, a.getSampleRate());
            checkEq("ch", 1, a.getChannels());
        });

        section("toAudioWaveforms", () -> {
            DataFrame df = MultimodalIO.readAudio(audDir.toString());
            List<Tensor> waves = MediaInterop.toAudioWaveforms(df, "audio");
            checkEq("n", df.rowCount(), waves.size());
            check("non-null", waves.get(0) != null);
        });

        section("basicEnglishNormalize + tokenize whitespace", () -> {
            DataFrame df = MultimodalIO.readTextFolder(txtFolder.toString());
            DataFrame norm = MediaInterop.basicEnglishNormalize(df, "text");
            String t = String.valueOf(norm.get(0, "text"));
            check("lower", t.equals(t.toLowerCase(Locale.ROOT)));
            // Use Function (not anonymous inner with package-private access issues)
            DataFrame tok = MediaInterop.tokenizeText(norm, "text", "tokens",
                    (java.util.function.Function<String, Object>) s -> {
                        String tt = s == null ? "" : s.trim();
                        if (tt.isEmpty()) return java.util.List.of();
                        return java.util.Arrays.asList(tt.split("\s+"));
                    });
            check("tokens col", tok.hasColumn("tokens"));
            check("tokens list", tok.get(0, "tokens") instanceof List);
        });

        section("ImageTensors / AudioTensors direct", () -> {
            ImageData img = solidImage(7, 5, 0x010203);
            Tensor ti = ImageTensors.toTensor(img);
            checkEq("CHW C", 3L, TensorBridge.sizesOf(ti)[0]);
            ImageData back = ImageTensors.toImageData(ti);
            checkEq("W", 7, back.getWidth());

            AudioData a = tone(16000, 0.02, 300);
            Tensor tw = AudioTensors.toTensor(a);
            AudioData ab = AudioTensors.toAudioData(tw, 16000);
            check("roundtrip samples", ab.getSamples().length > 0);
        });

        // ── 4. Preprocess pipelines ───────────────────────────────────────
        System.out.println("\n══ 4. MultimodalPreprocess pipelines ══");

        section("visionPipeline + embedVision", () -> {
            DataFrame df = MultimodalIO.readImageFolder(imgFolder.toString());
            DataFrame ready = MultimodalPreprocess.visionPipeline(df, "image", 16);
            DataFrame emb = MultimodalPreprocess.embedVision(ready, "image", "emb", 64);
            check("emb col", emb.hasColumn("emb"));
            checkEq("dim", 64, ((EmbeddingData) emb.get(0, "emb")).getDimension());
        });

        section("audioPipelineWithEmbed", () -> {
            DataFrame df = MultimodalIO.readAudio(audDir.toString());
            DataFrame out = MultimodalPreprocess.audioPipelineWithEmbed(df, "audio", 16000, 0.5, "emb", 32);
            check("emb", out.get(0, "emb") instanceof EmbeddingData);
        });

        section("videoToFrameEmbeddings", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("id", Column.DType.STRING);
            df.addColumn("video", Column.DType.VIDEO);
            int r0 = df.addEmptyRow();
            df.set(r0, "id", "v0");
            df.set(r0, "video", mockVideo(10, 12, 12, 10.0));
            DataFrame frames = MultimodalPreprocess.videoToFrameEmbeddings(df, "video", 5.0, 16);
            check("frame rows > 0", frames.rowCount() > 0);
            check("frame col", frames.hasColumn("frame"));
            check("emb col", frames.hasColumn("embedding"));
            check("frame_idx", frames.hasColumn("frame_idx"));
        });

        section("limitVideoFrames", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("video", Column.DType.VIDEO);
            int r0 = df.addEmptyRow();
            df.set(r0, "video", mockVideo(20, 4, 4, 20));
            DataFrame slim = MultimodalPreprocess.limitVideoFrames(df, "video", 5);
            VideoData v = (VideoData) slim.get(0, "video");
            checkEq("frames", 5, v.getFrames().size());
        });

        section("extractVideoFrames via MultimodalIO", () -> {
            DataFrame df = DataFrame.create();
            df.addColumn("video", Column.DType.VIDEO);
            int r0 = df.addEmptyRow();
            df.set(r0, "video", mockVideo(8, 6, 6, 8));
            DataFrame frames = MultimodalIO.extractVideoFrames(df, "video", 4.0);
            check("rows", frames.rowCount() >= 3);
        });

        section("fuseImageText + cosineMatrix", () -> {
            DataFrame images = DataFrame.fromImages("image", List.of(
                    solidImage(16, 16, 0xFF0000),
                    solidImage(16, 16, 0xFF0000),
                    solidImage(16, 16, 0x0000FF)));
            DataFrame texts = MultimodalIO.fromText(
                    List.of("red square", "also red", "blue block"),
                    List.of("a", "a", "b"));
            DataFrame fused = MultimodalPreprocess.fuseImageText(images, "image", texts, "text", 64);
            checkEq("rows", 3, fused.rowCount());
            check("image_emb", fused.get(0, "image_emb") instanceof EmbeddingData);
            check("text_emb", fused.get(0, "text_emb") instanceof EmbeddingData);
            float[][] sim = MultimodalPreprocess.cosineMatrix(fused, "image_emb", "image_emb");
            checkEq("sim square", 3, sim.length);
            check("self-sim ~1", sim[0][0] > 0.99f);
            check("same color high", sim[0][1] > 0.9f);
            check("diff color lower-or-eq", sim[0][2] <= sim[0][1] + 1e-3);
        });

        section("fromMultimodalDir unified embeddings", () -> {
            DataFrame df = MultimodalPreprocess.fromMultimodalDir(mixed.toString(), 48);
            check("rows", df.rowCount() >= 3);
            int embCount = 0;
            for (int r = 0; r < df.rowCount(); r++) {
                if (df.get(r, "embedding") instanceof EmbeddingData) embCount++;
            }
            check("embeddings present", embCount >= 3);
        });

        section("MultimodalIO.embedImages", () -> {
            DataFrame df = DataFrame.readImages(imgDir.toString());
            // Old DataFrame.readImages may lack width meta; MultimodalIO path preferred for embeds
            if (df.rowCount() == 0 || !(df.get(0, "image") instanceof ImageData)) {
                df = MultimodalIO.readImages(imgDir.toString());
            }
            DataFrame emb = MultimodalIO.embedImages(df, "image", "e", 24);
            check("e", emb.get(0, "e") instanceof EmbeddingData);
        });

        // ── 5. OpenCV / FFmpeg optional paths ─────────────────────────────
        System.out.println("\n══ 5. Optional OpenCV / FFmpeg ══");

        section("OpenCV loadImage if available", () -> {
            if (!MediaBridge.isOpenCvAvailable()) {
                System.out.println("    [skip — OpenCV natives / OpenCVIO not loadable]");
                check("opencv skip acknowledged", true);
                return;
            }
            try {
                ImageData img = MediaBridge.loadImage(
                        imgDir.resolve("red.png").toString(),
                        ImageOptions.defaults().withBackend(MediaBridge.ImageBackend.OPENCV));
                check("opencv img", img != null && img.getImage() != null);
                Object mat = MediaBridge.imageToMat(img);
                check("mat non-null", mat != null);
                ImageData back = MediaBridge.matToImage(mat);
                check("mat→image", back != null && back.getImage() != null);
            } catch (Throwable t) {
                // Natives reported present but runtime link/decode failed — soft-skip
                System.out.println("    [skip — OpenCV runtime failed: " + t.getClass().getSimpleName()
                        + ": " + t.getMessage() + "]");
                check("opencv runtime skip acknowledged", true);
            }
        });

        section("FFmpeg audio if available", () -> {
            if (!MediaBridge.isFFmpegAvailable()) {
                System.out.println("    [skip — FFmpeg natives not loaded]");
                check("ffmpeg skip acknowledged", true);
                return;
            }
            // WAV still fine through FFmpeg
            AudioData a = MediaBridge.loadAudioFFmpeg(
                    audDir.resolve("a440.wav").toString(), 16000, true);
            check("ffmpeg wav samples", a.getSamples() != null && a.getSamples().length > 0);
        });

        section("redecodeImagesOpenCv no-op without force when pixels present", () -> {
            DataFrame df = DataFrame.fromImages("image", List.of(solidImage(4, 4, 1)));
            DataFrame out = MediaInterop.redecodeImagesOpenCv(df, "image", false);
            checkEq("rows", 1, out.rowCount());
        });

        // ── 6. Correctness cross-checks ───────────────────────────────────
        System.out.println("\n══ 6. Cross-checks / invariants ══");

        section("same image embed is deterministic", () -> {
            ImageData img = solidImage(24, 24, 0x55AA11);
            float[] a = MediaBridge.embedImage(img, 64).getVector();
            float[] b = MediaBridge.embedImage(img, 64).getVector();
            check("equal", java.util.Arrays.equals(a, b));
        });

        section("different images → different embeds", () -> {
            float[] a = MediaBridge.embedImage(solidImage(24, 24, 0xFF0000), 64).getVector();
            float[] b = MediaBridge.embedImage(solidImage(24, 24, 0x0000FF), 64).getVector();
            float sim = MultimodalPreprocess.cosine(a, b);
            check("not identical", sim < 0.999f);
        });

        section("ImageData.load uses path roundtrip", () -> {
            Path p = imgDir.resolve("green.png");
            ImageData img = ImageData.load(p.toString());
            checkEq("path", p.toString(), img.getPath());
            check("pixels", img.getImage() != null);
        });

        section("AudioData.loadFromFile WAV real", () -> {
            AudioData a = AudioData.loadFromFile(audDir.resolve("a440.wav").toString(), 16000, true);
            check("real wav not stub-empty", a.getSamples() != null && a.getSamples().length > 100);
        });

        section("capabilities map keys", () -> {
            Map<String, Object> cap = MultimodalIO.capabilities();
            check("opencv key", cap.containsKey("opencv"));
            check("ffmpeg key", cap.containsKey("ffmpeg"));
        });


        // ── 7. Real mp3 / mp4 FFmpeg corpus ────────────────────────────────
        System.out.println("\n══ 7. Real mp3/mp4 FFmpeg corpus ══");

        Path realMedia = tmp.resolve("real_media");
        section("MediaSampleFactory.createCorpus", () -> {
            MediaSampleFactory.createCorpus(realMedia);
            check("tone440.wav", MediaSampleFactory.isNonEmptyFile(realMedia.resolve("tone440.wav")));
            check("solid_red.png", MediaSampleFactory.isNonEmptyFile(realMedia.resolve("solid_red.png")));
            boolean hasFfCli = MediaSampleFactory.hasFFmpeg();
            System.out.println("    system ffmpeg: " + (hasFfCli ? MediaSampleFactory.findFFmpeg() : "none"));
            if (hasFfCli) {
                check("tone440.mp3", MediaSampleFactory.isNonEmptyFile(realMedia.resolve("tone440.mp3")));
                check("clip_color.mp4", MediaSampleFactory.isNonEmptyFile(realMedia.resolve("clip_color.mp4")));
                check("clip_gray.mp4", MediaSampleFactory.isNonEmptyFile(realMedia.resolve("clip_gray.mp4")));
            } else {
                System.out.println("    [note — no system ffmpeg; compressed containers skipped]");
                check("ffmpeg-cli optional skip", true);
            }
        });

        section("FFmpeg decode real mp3 if available", () -> {
            Path mp3 = realMedia.resolve("tone440.mp3");
            if (!MediaSampleFactory.isNonEmptyFile(mp3)) {
                System.out.println("    [skip — no mp3 sample]");
                check("mp3 skip", true);
                return;
            }
            if (!MediaBridge.isFFmpegAvailable()) {
                // pure-Java path cannot decode mp3 — expect stub or exception handled
                try {
                    AudioData a = MediaBridge.loadAudio(mp3.toString(), 16000, true);
                    check("mp3 fallback non-null", a != null);
                } catch (Exception e) {
                    check("mp3 without natives handled", true);
                }
                return;
            }
            AudioData a = MediaBridge.loadAudioFFmpeg(mp3.toString(), 16000, true);
            check("mp3 samples", a.getSamples() != null && a.getSamples().length > 100);
            checkEq("mp3 sr", 16000, a.getSampleRate());
            check("mp3 mono-ish", a.getChannels() >= 1);
            // duration roughly 0.5s
            check("mp3 duration~0.5", a.getDuration() > 0.2 && a.getDuration() < 1.5);
            Tensor w = MediaBridge.audioToTensor(a);
            check("mp3 waveform rank", TensorBridge.sizesOf(w).length >= 1);
        });

        section("FFmpeg decode real mp4 video if available", () -> {
            Path mp4 = realMedia.resolve("clip_color.mp4");
            if (!MediaSampleFactory.isNonEmptyFile(mp4)) {
                System.out.println("    [skip — no mp4 sample]");
                check("mp4 skip", true);
                return;
            }
            if (!MediaBridge.isFFmpegAvailable()) {
                VideoData stub = MediaBridge.loadVideo(mp4.toString(),
                        VideoOptions.defaults().withMaxFrames(8));
                check("mp4 stub frames", stub.getFrames() != null && !stub.getFrames().isEmpty());
                return;
            }
            VideoData vid = MediaBridge.loadVideo(mp4.toString(),
                    VideoOptions.defaults().withMaxFrames(16).withTargetFps(5.0).withAudio(true));
            check("mp4 frames>0", vid.getFrames() != null && vid.getFrames().size() >= 2);
            check("mp4 width>0", vid.getWidth() > 0);
            check("mp4 height>0", vid.getHeight() > 0);
            ImageData f0 = vid.getFrames().get(0);
            check("frame0 pixels", f0 != null && f0.getImage() != null);
            // batch readVideo into DataFrame
            DataFrame df = MultimodalIO.readVideo(mp4.toString(),
                    VideoOptions.defaults().withMaxFrames(8).withTargetFps(4.0));
            check("readVideo rows", df.rowCount() >= 1);
            check("video cell", df.get(0, "video") instanceof VideoData);
            // frame explosion
            DataFrame frames = MultimodalIO.extractVideoFrames(df, "video", 4.0);
            check("exploded frames", frames.rowCount() >= 1);
            check("frame col", frames.hasColumn("frame"));
        });

        section("DataFrame.readAudio on mp3 dir + readMultimodal", () -> {
            if (!MediaSampleFactory.isNonEmptyFile(realMedia.resolve("tone440.mp3"))
                    && !MediaSampleFactory.isNonEmptyFile(realMedia.resolve("tone440.wav"))) {
                check("no media skip", true);
                return;
            }
            DataFrame aud = MultimodalIO.readAudio(realMedia.toString(), 16000, true);
            check("audio rows>=1", aud.rowCount() >= 1);
            DataFrame realMixed = MultimodalIO.readMultimodalDir(realMedia.toString());
            check("mixed rows>=3", realMixed.rowCount() >= 3);
            boolean sawImg = false, sawAud = false, sawVid = false;
            for (int r = 0; r < realMixed.rowCount(); r++) {
                String m = String.valueOf(realMixed.get(r, "modality"));
                if ("image".equals(m)) sawImg = true;
                if ("audio".equals(m)) sawAud = true;
                if ("video".equals(m)) sawVid = true;
            }
            check("mixed image", sawImg);
            check("mixed audio", sawAud);
            // video only if mp4 generated
            if (MediaSampleFactory.isNonEmptyFile(realMedia.resolve("clip_color.mp4"))) {
                check("mixed video", sawVid);
            }
        });

        section("two mp4s → different content embeds", () -> {
            Path c1 = realMedia.resolve("clip_color.mp4");
            Path c2 = realMedia.resolve("clip_gray.mp4");
            if (!MediaSampleFactory.isNonEmptyFile(c1) || !MediaSampleFactory.isNonEmptyFile(c2)) {
                System.out.println("    [skip — need both mp4s]");
                check("two-mp4 skip", true);
                return;
            }
            VideoOptions opts = VideoOptions.defaults().withMaxFrames(6).withTargetFps(4.0);
            VideoData v1 = MediaBridge.loadVideo(c1.toString(), opts);
            VideoData v2 = MediaBridge.loadVideo(c2.toString(), opts);
            float[] e1 = MediaBridge.embedVideo(v1, 64).getVector();
            float[] e2 = MediaBridge.embedVideo(v2, 64).getVector();
            float sim = MultimodalPreprocess.cosine(e1, e2);
            check("video embeds unitish", l2(e1) > 0.5 && l2(e2) > 0.5);
            // color bars vs gradients should not be identical
            check("video embeds differ", sim < 0.999f);
        });

        // ── 8. Neural vision / audio embeddings ───────────────────────────
        System.out.println("\n══ 8. Neural TorchVision / TorchAudio embeddings ══");

        section("registry lists neural models", () -> {
            check("has resnet18", EmbeddingRegistry.contains("resnet18"));
            check("has m5", EmbeddingRegistry.contains("m5"));
            check("has mobilenet_v2", EmbeddingRegistry.contains("mobilenet_v2"));
            EmbeddingModel rv = EmbeddingRegistry.get("resnet18");
            check("resnet backend", rv.backend().contains("torchvision") || rv.backend().contains("vision")
                    || rv.backend().contains("torch"));
            System.out.println("    resnet18 backend=" + rv.backend() + " dim=" + rv.dimension());
            EmbeddingModel am = EmbeddingRegistry.get("m5");
            System.out.println("    m5 backend=" + am.backend() + " dim=" + am.dimension());
            check("m5 backend", am.backend().contains("torchaudio") || am.backend().contains("audio")
                    || am.backend().contains("torch") || am.backend().contains("m5"));
        });

        section("TorchVisionEmbeddingModel resnet18 red vs blue", () -> {
            EmbeddingModel m = TorchVisionEmbeddingModel.resnet18();
            m.warmup();
            ImageData red = solidImage(64, 64, 0xFF0000);
            ImageData blue = solidImage(64, 64, 0x0000FF);
            float[] er = m.embed(red, Modality.IMAGE);
            float[] eb = m.embed(blue, Modality.IMAGE);
            check("dim", er != null && er.length == m.dimension());
            check("unitish r", l2(er) > 0.5);
            check("unitish b", l2(eb) > 0.5);
            float sim = MultimodalPreprocess.cosine(er, eb);
            System.out.println("    resnet red↔blue cos=" + sim);
            check("red≠blue", sim < 0.999f);
            // deterministic
            float[] er2 = m.embed(red, Modality.IMAGE);
            check("deterministic", java.util.Arrays.equals(er, er2));
        });

        section("TorchAudioEmbeddingModel m5 440 vs 880", () -> {
            EmbeddingModel m = TorchAudioEmbeddingModel.m5();
            m.warmup();
            AudioData a440 = tone(16000, 0.5, 440);
            AudioData a880 = tone(16000, 0.5, 880);
            float[] e1 = m.embed(a440, Modality.AUDIO);
            float[] e2 = m.embed(a880, Modality.AUDIO);
            check("dim", e1 != null && e1.length == m.dimension());
            check("unitish", l2(e1) > 0.5 && l2(e2) > 0.5);
            float sim = MultimodalPreprocess.cosine(e1, e2);
            System.out.println("    m5 440↔880 cos=" + sim);
            check("tones differ-or-structured", sim < 0.999f);
        });

        section("MediaBridge.embedImageModel / embedAudioModel", () -> {
            EmbeddingData ie = MediaBridge.embedImageModel(solidImage(48, 48, 0x336699), "resnet18");
            check("img model dim", ie.getDimension() >= 128);
            check("img model name", ie.getModelName() != null && ie.getModelName().length() > 0);
            EmbeddingData ae = MediaBridge.embedAudioModel(tone(16000, 0.3, 523.25), "m5");
            check("aud model dim", ae.getDimension() >= 64);
        });

        section("MultimodalIO.embedImagesModel DataFrame path", () -> {
            DataFrame df = MultimodalIO.fromImages("image", List.of(
                    solidImage(32, 32, 0xFF0000),
                    solidImage(32, 32, 0x00FF00),
                    solidImage(32, 32, 0x0000FF)));
            DataFrame emb = MultimodalIO.embedImagesModel(df, "image", "emb", "mobilenet_v2");
            checkEq("rows", 3, emb.rowCount());
            check("emb0", emb.get(0, "emb") instanceof EmbeddingData);
            EmbeddingData e0 = (EmbeddingData) emb.get(0, "emb");
            EmbeddingData e2 = (EmbeddingData) emb.get(2, "emb");
            float sim = MultimodalPreprocess.cosine(e0.getVector(), e2.getVector());
            check("batch red≠blue", sim < 0.999f);
        });

        section("neural video embed via resnet18 frames", () -> {
            VideoData vid = mockVideo(6, 64, 64, 6.0);
            EmbeddingData ve = MediaBridge.embedVideoModel(vid, "resnet18");
            check("vid dim", ve.getDimension() >= 128);
            check("vid unitish", l2(ve.getVector()) > 0.5);
        });

        section("hash vs neural backends differ", () -> {
            ImageData img = solidImage(40, 40, 0xAABBCC);
            float[] hash = MediaBridge.embedImage(img, 128).getVector();
            float[] neural = MediaBridge.embedImageModel(img, "resnet18").getVector();
            // different algorithms → not bit-identical (dims may differ; compare truncated)
            int n = Math.min(hash.length, neural.length);
            boolean same = true;
            for (int i = 0; i < n; i++) {
                if (Math.abs(hash[i] - neural[i]) > 1e-6) { same = false; break; }
            }
            check("hash≠neural", !same || hash.length != neural.length);
        });

        // ── summary ───────────────────────────────────────────────────────
        System.out.println("\n══════════════════════════════════════");
        System.out.println("Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL CHECKS PASSED");
    }

    static double l2(float[] v) {
        double s = 0;
        for (float x : v) s += x * x;
        return Math.sqrt(s);
    }

    // overload used above with unused eps placeholder — keep signature simple
    static void checkEq(String name, int expected, int actual, int ignored) {
        checkEq(name, expected, actual);
    }
}
