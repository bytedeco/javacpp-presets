package media;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.audio.functional.AudioF;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.audio.datasets.FakeAudio;
import org.bytedeco.pytorch.audio.models.AudioModels;
import org.bytedeco.pytorch.audio.utils.AudioTensors;
import org.bytedeco.pytorch.audio.librosa.Librosa;
import org.bytedeco.pytorch.audio.librosa.feature.Feature;
import org.bytedeco.pytorch.utils.orm.SqlDBHelper;
import org.bytedeco.pytorch.utils.orm.dataframe.DataFrameMapper;
import org.bytedeco.pytorch.llm.spacy.Doc;
import org.bytedeco.pytorch.llm.spacy.Language;
import org.bytedeco.pytorch.llm.spacy.Spacy;
import org.bytedeco.pytorch.llm.spacy.Token;
import org.bytedeco.pytorch.llm.spacy.pipeline.Sentencizer;
import org.bytedeco.pytorch.llm.text.tokenizer.BPETokenizer;
import org.bytedeco.pytorch.llm.text.tokenizer.BasicEnglishTokenizer;
import org.bytedeco.pytorch.llm.text.tokenizer.WordPieceTokenizer;
import org.bytedeco.pytorch.llm.text.transforms.TextTransforms;
import org.bytedeco.pytorch.llm.text.vocab.Vocab;
import org.bytedeco.pytorch.plot.tqdm.ProgressBarColor;
import org.bytedeco.pytorch.plot.tqdm.Tqdm;
import org.bytedeco.pytorch.plot.tqdm.TqdmBar;
import org.bytedeco.pytorch.vision.datasets.FakeData;
import org.bytedeco.pytorch.vision.datasets.ImageFolder;
import org.bytedeco.pytorch.vision.datasets.VisionDataset;
import org.bytedeco.pytorch.vision.io.ImageIO;
import org.bytedeco.pytorch.vision.models.VisionModels;
import org.bytedeco.pytorch.vision.ops.Boxes;
import org.bytedeco.pytorch.vision.transforms.VisionCompose;
import org.bytedeco.pytorch.vision.transforms.VisionTransforms;
import org.bytedeco.pytorch.vision.transforms.functional.VisionF;
import org.bytedeco.pytorch.vision.utils.ImageTensors;
import org.bytedeco.pytorch.vision.utils.VisionUtils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Multi-dimensional benchmark for {@code org.bytedeco.pytorch.utils} ecosystem:
 * vision / audio / librosa / text / spacy / orm / tqdm.
 *
 * <p>Dimensions D1–D10 mirror the implementation plan. Exit non-zero on failure.
 *
 * <pre>
 *   java media.BenchmarkUtilsEcosystem
 * </pre>
 */
public class BenchmarkUtilsEcosystem {

    static int passed = 0;
    static int failed = 0;
    static final StringBuilder report = new StringBuilder();

    static void check(String name, boolean ok) {
        check(name, ok, null);
    }

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
            report.append("PASS  ").append(name).append('\n');
        } else {
            failed++;
            System.out.println("  FAIL  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
            report.append("FAIL  ").append(name);
            if (detail != null) {
                report.append(" — ").append(detail);
            }
            report.append('\n');
        }
    }

    static void section(String title) {
        System.out.println("\n=== " + title + " ===");
        report.append("\n=== ").append(title).append(" ===\n");
    }

    static boolean finite(Tensor t) {
        try {
            float[] d = ImageTensors.toFloatArray(t);
            for (float v : d) {
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    return false;
                }
            }
            return true;
        } catch (Throwable e) {
            return false;
        }
    }

    static boolean approxEq(float a, float b, float tol) {
        return Math.abs(a - b) <= tol;
    }

    static BufferedImage solidImage(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.setColor(Color.WHITE);
        g.fillOval(w / 4, h / 4, w / 2, h / 2);
        g.dispose();
        return img;
    }

    static float[] sineWave(int sr, double freq, double seconds) {
        int n = Math.max(1, (int) Math.round(sr * seconds));
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            y[i] = (float) Math.sin(2 * Math.PI * freq * i / sr);
        }
        return y;
    }

    static TensorOptions longOpts() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
    }

    /** Class-index target of shape [1] (Long). */
    static Tensor label1(long label) {
        return tensor(new long[][]{{label}}).reshape(1);
    }

    public static class Person {
        public long id;
        public String name;
        public int age;

        public Person() {}

        public Person(long id, String name, int age) {
            this.id = id;
            this.name = name;
            this.age = age;
        }

        public long getId() { return id; }
        public void setId(long id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public int getAge() { return age; }
        public void setAge(int age) { this.age = age; }
    }

    static void deleteTree(Path tmp) {
        try {
            if (tmp == null || !Files.exists(tmp)) {
                return;
            }
            Files.walk(tmp)
                    .sorted((a, b) -> b.compareTo(a))
                    .forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (Exception ignored) {
                        }
                    });
        } catch (Exception ignored) {
        }
    }

    // =========================================================================
    // D1 API shape / smoke
    // =========================================================================

    static void d1ApiShape() {
        section("D1 API shape / smoke");
        try {
            VisionCompose c = VisionCompose.of(
                    new VisionTransforms.Resize(32),
                    new VisionTransforms.ToTensor(),
                    new VisionTransforms.Normalize(
                            new float[]{0.5f, 0.5f, 0.5f},
                            new float[]{0.5f, 0.5f, 0.5f})
            );
            check("vision.Compose chainable", c != null && c.transforms().size() == 3);
        } catch (Throwable e) {
            check("vision.Compose chainable", false, e.toString());
        }

        try {
            float[] y = sineWave(16000, 440, 0.25);
            float[][] mfcc = Feature.mfcc(y, 16000, 13);
            check("librosa.Feature.mfcc",
                    mfcc != null && mfcc.length == 13 && mfcc[0].length > 0,
                    "shape=[" + mfcc.length + "," + mfcc[0].length + "]");
        } catch (Throwable e) {
            check("librosa.Feature.mfcc", false, e.toString());
        }

        try {
            Language nlp = Spacy.blank("en");
            Doc doc = nlp.apply("Hello world from spacy!");
            check("spacy.blank+apply",
                    doc != null && doc.length() >= 3,
                    "tokens=" + (doc == null ? -1 : doc.length()));
        } catch (Throwable e) {
            check("spacy.blank+apply", false, e.toString());
        }

        try {
            int n = 0;
            for (Integer i : Tqdm.range(5).setDisable(true)) {
                n += i;
            }
            check("tqdm.range", n == 10);
        } catch (Throwable e) {
            check("tqdm.range", false, e.toString());
        }

        try {
            List<Person> people = List.of(
                    new Person(1, "Ada", 36),
                    new Person(2, "Grace", 45));
            DataFrame df = DataFrameMapper.fromBeans(people);
            check("orm.DataFrameMapper.fromBeans",
                    df != null && df.rowCount() == 2 && df.columnCount() >= 3,
                    "rows=" + (df == null ? -1 : df.rowCount())
                            + " cols=" + (df == null ? -1 : df.columnCount()));
        } catch (Throwable e) {
            check("orm.DataFrameMapper.fromBeans", false, e.toString());
        }

        try {
            check("librosa.DEFAULT_SR", Librosa.DEFAULT_SR == 22050);
            check("Spacy.info non-empty", Spacy.info() != null && !Spacy.info().isEmpty());
        } catch (Throwable e) {
            check("facade constants", false, e.toString());
        }
    }

    // =========================================================================
    // D2 Vision numerical
    // =========================================================================

    static void d2VisionNumerical() {
        section("D2 Vision numerical");
        try (PointerScope scope = new PointerScope()) {
            BufferedImage img = solidImage(64, 48, new Color(30, 120, 200));
            Tensor t = VisionF.toTensor(img);
            long[] sz = ImageTensors.sizes(t);
            check("ToTensor CHW rank", sz.length == 3, Arrays.toString(sz));
            check("ToTensor channels", sz[0] == 3L, "C=" + sz[0]);
            check("ToTensor H", sz[1] == 48L);
            check("ToTensor W", sz[2] == 64L);
            check("ToTensor finite", finite(t));

            float[] data = ImageTensors.toFloatArray(t);
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (float v : data) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
            check("ToTensor range [0,1]", min >= -1e-5f && max <= 1f + 1e-5f,
                    "min=" + min + " max=" + max);

            Tensor norm = VisionF.normalize(t,
                    new float[]{0.5f, 0.5f, 0.5f},
                    new float[]{0.5f, 0.5f, 0.5f});
            check("Normalize finite", finite(norm));

            BufferedImage flipped = VisionF.hflip(img);
            BufferedImage twice = VisionF.hflip(flipped);
            boolean involution = true;
            for (int y = 0; y < img.getHeight() && involution; y += 7) {
                for (int x = 0; x < img.getWidth(); x += 11) {
                    if (img.getRGB(x, y) != twice.getRGB(x, y)) {
                        involution = false;
                        break;
                    }
                }
            }
            check("hflip involution", involution);

            BufferedImage cropped = VisionF.centerCrop(img, 32, 32);
            check("centerCrop size", cropped.getWidth() == 32 && cropped.getHeight() == 32);

            float[] boxes = {
                    0, 0, 10, 10,
                    1, 1, 11, 11,
                    50, 50, 60, 60
            };
            float[] scores = {0.9f, 0.8f, 0.7f};
            int[] keep = Boxes.nms(boxes, scores, 0.5f);
            check("NMS keeps non-overlapping", keep.length == 2, "kept=" + Arrays.toString(keep));
            check("NMS top score first", keep.length > 0 && keep[0] == 0);

            float iou = Boxes.box_iou(new float[]{0, 0, 10, 10}, new float[]{0, 0, 10, 10});
            check("box_iou self ~1", approxEq(iou, 1f, 1e-4f), "iou=" + iou);
        } catch (Throwable e) {
            check("D2 vision block", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    // =========================================================================
    // D3 Vision I/O + datasets
    // =========================================================================

    static void d3VisionIoDatasets() throws Exception {
        section("D3 Vision I/O + datasets");
        Path tmp = Files.createTempDirectory("bench_vision_");
        try (PointerScope scope = new PointerScope()) {
            BufferedImage img = solidImage(40, 40, Color.RED);
            Tensor t = ImageTensors.toTensor(img);
            Path png = tmp.resolve("a.png");
            ImageIO.write_image(t, png);
            check("write_image png exists", Files.exists(png) && Files.size(png) > 0);

            Tensor loaded = ImageIO.read_image(png);
            long[] sz = ImageTensors.sizes(loaded);
            check("read_image shape", sz.length == 3 && sz[0] == 3, Arrays.toString(sz));
            check("read_image finite", finite(loaded));

            byte[] encoded = ImageIO.encode_png(t);
            Tensor decoded = ImageIO.decode_image(encoded);
            check("encode/decode png", finite(decoded) && ImageTensors.sizes(decoded)[0] == 3);

            FakeData fake = new FakeData(8, 32, 4)
                    .setTransform(VisionCompose.of(new VisionTransforms.Resize(28), new VisionTransforms.ToTensor()));
            check("FakeData size", fake.size() == 8);
            VisionDataset.Sample s0 = fake.get(0);
            check("FakeData sample tensor", s0.data() instanceof Tensor, String.valueOf(s0.data()));
            if (s0.data() instanceof Tensor td) {
                long[] ds = ImageTensors.sizes(td);
                check("FakeData transformed CHW",
                        ds.length == 3 && ds[1] == 28 && ds[2] == 28,
                        Arrays.toString(ds));
            }
            check("FakeData label int", s0.target() instanceof Integer);

            Path root = tmp.resolve("ifolder");
            Files.createDirectories(root.resolve("cat"));
            Files.createDirectories(root.resolve("dog"));
            ImageIO.write_image(t, root.resolve("cat/c1.png"));
            ImageIO.write_image(t, root.resolve("dog/d1.png"));
            ImageFolder folder = new ImageFolder(root, VisionCompose.of(new VisionTransforms.ToTensor()));
            check("ImageFolder size", folder.size() == 2);
            check("ImageFolder classes", folder.classes().size() == 2);

            Tensor grid = VisionUtils.make_grid(FakeData.randomBatch(4, 3, 16, 16), 2, 2);
            long[] gsz = ImageTensors.sizes(grid);
            check("make_grid CHW", gsz.length == 3 && gsz[0] == 3, Arrays.toString(gsz));
        } catch (Throwable e) {
            check("D3 vision io", false, e.toString());
            e.printStackTrace(System.out);
        } finally {
            deleteTree(tmp);
        }
    }

    // =========================================================================
    // D4 Audio + librosa
    // =========================================================================

    static void d4AudioLibrosa() throws Exception {
        section("D4 Audio + librosa");
        Path tmp = Files.createTempDirectory("bench_audio_");
        try (PointerScope scope = new PointerScope()) {
            int sr = 16000;
            float[] y = sineWave(sr, 440, 0.5);
            AudioData ad = new AudioData(y, sr, 1);
            Tensor wave = AudioTensors.toTensor(ad);
            long[] wsz = ImageTensors.sizes(wave);
            check("AudioTensors [C,T]", wsz.length == 2 && wsz[0] == 1, Arrays.toString(wsz));

            Path wav = tmp.resolve("tone.wav");
            ad.saveAsWav(wav.toString());
            check("WAV save", Files.exists(wav) && Files.size(wav) > 44);

            AudioData loaded = AudioData.load(wav.toString(), sr, true);
            check("WAV load samples",
                    loaded.getSamples() != null && loaded.getSamples().length > 1000,
                    "n=" + (loaded.getSamples() == null ? -1 : loaded.getSamples().length));

            Tensor mel = AudioF.mel_spectrogram(wave, sr);
            check("audio.F.mel_spectrogram finite", finite(mel), "dim=" + mel.dim());
            Tensor mfccT = AudioF.mfcc(wave, sr);
            check("audio.F.mfcc finite", finite(mfccT));

            float[][] melL = Feature.melspectrogram(y, sr);
            float[][] mfccL = Feature.mfcc(y, sr, 13);
            check("librosa melspectrogram",
                    melL != null && melL.length > 0 && melL[0].length > 0,
                    "n_mels=" + melL.length + " frames=" + melL[0].length);
            check("librosa mfcc 13", mfccL != null && mfccL.length == 13);

            float[] zcr = Feature.zero_crossing_rate(y);
            float[] rms = Feature.rms(y);
            check("librosa zcr/rms", zcr != null && zcr.length > 0 && rms != null && rms.length > 0);

            float[] y2 = Librosa.resample(y, sr, 8000);
            check("librosa.resample length",
                    y2.length > 0 && y2.length < y.length,
                    "in=" + y.length + " out=" + y2.length);

            double dur = Librosa.get_duration(y, sr);
            check("librosa.get_duration ~0.5", Math.abs(dur - 0.5) < 0.05, "dur=" + dur);

            FakeAudio fa = new FakeAudio(4, 16000, 1600, 3);
            check("FakeAudio size", fa.size() == 4);
            check("FakeAudio sample", fa.get(0).data() != null);

            Tensor masked = AudioF.frequency_masking(mel, 4);
            check("frequency_masking finite", finite(masked));
        } catch (Throwable e) {
            check("D4 audio/librosa", false, e.toString());
            e.printStackTrace(System.out);
        } finally {
            deleteTree(tmp);
        }
    }

    // =========================================================================
    // D5 Text
    // =========================================================================

    static void d5Text() {
        section("D5 Text");
        try (PointerScope scope = new PointerScope()) {
            BasicEnglishTokenizer basic = new BasicEnglishTokenizer();
            List<String> toks = basic.tokenize("Hello, World! This is a TEST.");
            check("BasicEnglish tokenize", toks != null && toks.size() >= 5, String.valueOf(toks));

            List<List<String>> corpus = new ArrayList<>();
            corpus.add(Arrays.asList("hello", "world", "hello", "torch"));
            corpus.add(Arrays.asList("world", "of", "torch", "text"));
            corpus.add(Arrays.asList("hello", "text", "world"));
            Vocab vocab = Vocab.build_vocab_from_iterator(
                    corpus, 1,
                    Arrays.asList(Vocab.DEFAULT_UNK, Vocab.DEFAULT_PAD, Vocab.DEFAULT_BOS, Vocab.DEFAULT_EOS));
            check("Vocab size > specials", vocab.size() >= 6, "size=" + vocab.size());

            int[] ids = vocab.encode(Arrays.asList("hello", "world", "unknown_xyz"));
            check("Vocab encode length", ids.length == 3);
            check("Vocab unk for OOV", ids[2] == vocab.unkId(),
                    "id=" + ids[2] + " unk=" + vocab.unkId());
            List<String> back = vocab.decode(ids);
            check("Vocab decode hello", "hello".equals(back.get(0)), String.valueOf(back));

            int[] padded = new TextTransforms.PadTransform(5, vocab.padId()).apply(ids);
            check("PadTransform len 5", padded.length == 5);
            int[] trunc = new TextTransforms.TruncateIds(2).apply(ids);
            check("TruncateIds len 2", trunc.length == 2);

            WordPieceTokenizer wp = WordPieceTokenizer.buildFromCorpus(corpus, 1, 500);
            List<String> wpToks = wp.tokenize("hello world");
            check("WordPiece tokenize", wpToks != null && !wpToks.isEmpty(), String.valueOf(wpToks));

            BPETokenizer bpe = BPETokenizer.learn(
                    List.of("hello world", "world of torch text", "hello text world"), 20);
            List<String> bpeToks = bpe.tokenize("hello world");
            check("BPE tokenize", bpeToks != null && !bpeToks.isEmpty(), String.valueOf(bpeToks));

            @SuppressWarnings({"rawtypes", "unchecked"})
            TextTransforms.Sequential pipe = TextTransforms.sequential(
                    new TextTransforms.Tokenize(basic::tokenize),
                    new TextTransforms.VocabTransform(vocab),
                    new TextTransforms.PadTransform(8, vocab.padId()),
                    new TextTransforms.ToTensor()
            );
            Object ttObj = pipe.apply("hello torch text world");
            check("text Sequential → Tensor",
                    ttObj instanceof Tensor && ((Tensor) ttObj).numel() == 8,
                    "numel=" + (ttObj instanceof Tensor t ? t.numel() : -1));
        } catch (Throwable e) {
            check("D5 text", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    // =========================================================================
    // D6 spaCy
    // =========================================================================

    static void d6Spacy() {
        section("D6 spaCy");
        try {
            Language nlp = Spacy.blank("en");
            nlp.addPipe("sentencizer", new Sentencizer());
            Doc doc = nlp.apply("Hello world! This is a second sentence.");
            check("spacy tokens > 5", doc.length() > 5, "n=" + doc.length());
            check("spacy text preserved",
                    doc.getText() != null && doc.getText().contains("Hello"));

            int count = 0;
            for (Token t : doc) {
                if (t.getText() != null && !t.getText().isEmpty()) {
                    count++;
                }
            }
            check("spacy iterable tokens", count == doc.length(), "iter=" + count);
            check("spacy pipe names",
                    nlp.pipeNames() != null && nlp.pipeNames().contains("sentencizer"),
                    String.valueOf(nlp.pipeNames()));

            Language loaded = Spacy.load("en_core_web_sm");
            Doc d2 = loaded.apply("SpaCy load works.");
            check("spacy.load tokens", d2.length() >= 2, "n=" + d2.length());
            check("Spacy.version", Spacy.version() != null && !Spacy.version().isEmpty());
        } catch (Throwable e) {
            check("D6 spacy", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    // =========================================================================
    // D7 ORM
    // =========================================================================

    static void d7Orm() {
        section("D7 ORM");
        try (SqlDBHelper db = SqlDBHelper.sqliteMemory()) {
            db.createTableFromBean("person", Person.class, "id");
            check("createTableFromBean", true);

            int ins = db.insert("person", new Person(1L, "Ada", 36));
            db.insert("person", new Person(2L, "Grace", 45));
            db.insert("person", new Person(3L, "Alan", 41));
            check("insert beans", ins >= 0);

            List<Person> all = db.query(Person.class, "SELECT * FROM person ORDER BY id");
            check("query beans size 3", all != null && all.size() == 3,
                    "n=" + (all == null ? -1 : all.size()));
            check("query bean fields",
                    all != null && "Ada".equals(all.get(0).name) && all.get(0).age == 36);

            Person one = db.findById(Person.class, "person", "id", 2L);
            check("findById Grace", one != null && "Grace".equals(one.name));

            one.age = 46;
            db.updateById("person", one, "id");
            Person updated = db.findById(Person.class, "person", "id", 2L);
            check("updateById", updated != null && updated.age == 46);

            long cnt = db.count("person");
            check("count", cnt == 3, "count=" + cnt);

            db.deleteById("person", "id", 3L);
            check("deleteById", db.count("person") == 2);

            List<Person> remaining = db.findAll(Person.class, "person");
            DataFrame df = DataFrameMapper.fromBeans(remaining);
            check("DF fromBeans rows", df.rowCount() == 2);
            List<Person> round = DataFrameMapper.toBeans(df, Person.class);
            check("DF toBeans size", round.size() == 2);
            boolean namesOk = round.stream().anyMatch(p -> "Ada".equals(p.name))
                    && round.stream().anyMatch(p -> "Grace".equals(p.name));
            check("DF roundtrip names", namesOk);

            db.withTransaction(() -> {
                try {
                    db.insert("person", new Person(10L, "Txn", 1));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            check("withTransaction insert",
                    db.findById(Person.class, "person", "id", 10L) != null);
        } catch (Throwable e) {
            check("D7 orm", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    // =========================================================================
    // D8 tqdm
    // =========================================================================

    static void d8Tqdm() {
        section("D8 tqdm");
        try {
            int sum = 0;
            try (TqdmBar<Integer> bar = Tqdm.trange(100)
                    .setDisable(true)
                    .setDescription("bench")
                    .set_postfix(Map.of("loss", "0.01"))
                    .setColour(ProgressBarColor.GREEN)
                    .ascii(true)
                    .ncols(60)
                    .leave(false)) {
                for (Integer i : bar) {
                    sum += i;
                }
            }
            check("trange 100 sum", sum == 4950, "sum=" + sum);

            TqdmBar<Void> manual = Tqdm.manual(10).setDisable(true);
            for (int i = 0; i < 10; i++) {
                manual.update();
            }
            check("manual n>=10", manual.n() >= 10, "n=" + manual.n());
            Tqdm.write("benchmark write ok");
            check("Tqdm.write", true);

            int n = 0;
            for (Integer ignored : Tqdm.range(10_000).setDisable(true).setMinInterval(1.0)) {
                n++;
            }
            check("tqdm 10k smoke", n == 10_000);
        } catch (Throwable e) {
            check("D8 tqdm", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    // =========================================================================
    // D9 Mini training
    // =========================================================================

    static void d9MiniTraining() {
        section("D9 Mini training");
        try (PointerScope scope = new PointerScope()) {
            FakeData data = new FakeData(16, 16, 3, 3, 0L)
                    .setTransform(VisionCompose.of(new VisionTransforms.ToTensor()));
            Module net = new VisionModels.SimpleClassifier(3L * 16 * 16, 64, 3);
            net.train(true);
            Adam opt = new Adam(net.parameters(), new AdamOptions(1e-2));

            double lastLoss = Double.NaN;
            int steps = 0;
            try (TqdmBar<Integer> bar = Tqdm.range(4).setDisable(true).setDescription("train")) {
                for (Integer epoch : bar) {
                    for (int i = 0; i < data.size(); i++) {
                        VisionDataset.Sample s = data.get(i);
                        Tensor x = ((Tensor) s.data()).reshape(1, -1);
                        long label = ((Integer) s.target()).longValue();
                        Tensor y = label1(label);

                        opt.zero_grad();
                        Tensor logits = net.forward(x);
                        Tensor logp = log_softmax(logits, 1);
                        Tensor loss = nll_loss(logp, y);
                        loss.backward();
                        opt.step();
                        lastLoss = loss.item_float();
                        steps++;
                        bar.set_postfix(Map.of(
                                "loss", String.format("%.4f", lastLoss),
                                "ep", String.valueOf(epoch)));
                    }
                }
            }
            check("train steps > 0", steps > 0, "steps=" + steps);
            check("train loss finite",
                    !Double.isNaN(lastLoss) && !Double.isInfinite(lastLoss),
                    "loss=" + lastLoss);

            float[] yWave = sineWave(16000, 440, 0.25);
            AudioData ad = new AudioData(yWave, 16000, 1);
            Tensor wave = AudioTensors.toTensor(ad);
            Tensor mel = AudioF.mel_spectrogram(wave, 16000);
            Tensor feat = mel.reshape(1, -1);
            long inFeat = feat.size(1);
            Module audioNet = AudioModels.simple_audio_classifier(inFeat, 4);
            audioNet.train(true);
            Adam aopt = new Adam(audioNet.parameters(), new AdamOptions(1e-2));
            Tensor ay = label1(1L);
            aopt.zero_grad();
            Tensor alogits = audioNet.forward(feat);
            Tensor aloss = nll_loss(log_softmax(alogits, 1), ay);
            aloss.backward();
            aopt.step();
            float al = aloss.item_float();
            check("audio train step finite",
                    !Float.isNaN(al) && !Float.isInfinite(al),
                    "loss=" + al);
        } catch (Throwable e) {
            check("D9 mini training", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    // =========================================================================
    // D10 Interop / Tensor contract
    // =========================================================================

    static void d10Interop() {
        section("D10 Interop / Tensor contract");
        try (PointerScope scope = new PointerScope()) {
            BufferedImage bi = solidImage(32, 32, Color.GREEN);
            ImageData id = new ImageData(bi);
            Object out = VisionCompose.of(new VisionTransforms.Resize(24), new VisionTransforms.ToTensor())
                    .forward(id.getImage());
            check("ImageData→Compose→Tensor",
                    out instanceof Tensor && finite((Tensor) out));

            float[] y = sineWave(16000, 220, 0.2);
            AudioData ad = new AudioData(y, 16000, 1);
            float[][] mel = Feature.melspectrogram(ad.getSamples(), ad.getSampleRate());
            Tensor melT = AudioTensors.featureToTensor(mel);
            check("AudioData→librosa→Tensor", finite(melT) && melT.dim() == 2);

            Tensor wave = AudioTensors.toTensor(ad);
            check("sine waveform no NaN", finite(wave));

            Tensor t = tensor(new float[]{1f, 2f, 3f, 4f}).reshape(2, 2);
            check("tensor reshape finite",
                    finite(t) && t.size(0) == 2 && t.size(1) == 2);

            Module m = VisionModels.get_model("simple_cnn", 10);
            check("get_model simple_cnn", m != null);
            Module m2 = VisionModels.get_model("resnet18", 10);
            check("get_model resnet18", m2 != null);
        } catch (Throwable e) {
            check("D10 interop", false, e.toString());
            e.printStackTrace(System.out);
        }
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        System.out.println("=== Utils Ecosystem multi-dimensional benchmark (D1–D10) ===");
        System.out.println("packages: vision | audio | librosa | text | spacy | orm | tqdm\n");

        long t0 = System.nanoTime();
        d1ApiShape();
        d2VisionNumerical();
        d3VisionIoDatasets();
        d4AudioLibrosa();
        d5Text();
        d6Spacy();
        d7Orm();
        d8Tqdm();
        d9MiniTraining();
        d10Interop();
        double sec = (System.nanoTime() - t0) / 1e9;

        System.out.println("\n============================================================");
        System.out.println("SUMMARY  passed=" + passed + "  failed=" + failed
                + "  elapsed=" + String.format("%.2fs", sec));
        System.out.println("============================================================");
        if (failed > 0) {
            System.out.println("\nFailed checks:");
            report.toString().lines()
                    .filter(l -> l.startsWith("FAIL"))
                    .forEach(System.out::println);
            System.exit(1);
        }
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
