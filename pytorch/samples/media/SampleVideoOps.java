package media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.vision.ffmpeg.FFmpeg;
import org.bytedeco.pytorch.vision.ffmpeg.FFmpegLoader;
import org.bytedeco.pytorch.vision.ffmpeg.VideoFile;
import org.bytedeco.pytorch.vision.ffmpeg.VideoOps;
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;
import org.bytedeco.pytorch.vision.opencv.OpenCVOps;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * End-to-end examples: video frame extraction + common video ops (FFmpeg)
 * and enterprise OpenCV multimodal preprocess.
 *
 * <pre>
 *   javac -cp "target/classes:$(cat target/cp.txt)" -d target/samples-compile \
 *         samples/SampleVideoOps.java
 *   java  -cp "target/samples-compile:target/classes:$(cat target/cp.txt)" \
 *         media.SampleVideoOps [video.mp4] [image.jpg]
 * </pre>
 *
 * Defaults to {@code samples/fixtures/multimodal/test_video.mp4} and
 * {@code test_image.jpg} when args omitted.
 */
public class SampleVideoOps {

    public static void main(String[] args) throws Exception {
        Path video = Path.of(args.length > 0 ? args[0]
                : "samples/fixtures/multimodal/test_video.mp4");
        Path image = Path.of(args.length > 1 ? args[1]
                : "samples/fixtures/multimodal/test_image.jpg");
        Path outDir = Path.of("samples/out/video_ops_demo");
        Files.createDirectories(outDir);

        System.out.println("=== VideoOps / OpenCVOps demo ===");
        System.out.println("video = " + video.toAbsolutePath());
        System.out.println("image = " + image.toAbsolutePath());
        System.out.println("caps  = " + VideoOps.capabilities());
        System.out.println("cv    = " + OpenCVOps.capabilities().get("pipelines"));

        if (!Files.isRegularFile(video)) {
            System.err.println("SKIP: video fixture missing: " + video);
            return;
        }

        // ── 1. Probe ─────────────────────────────────────────────────────
        VideoFile.VideoMeta meta = VideoOps.probe(video.toString());
        System.out.println("\n[1] probe: " + meta);

        // ── 2. Direct frame extraction (in-process libav) ────────────────
        System.out.println("\n[2] extract frames");
        Tensor thumb = VideoOps.thumbnail(video.toString());
        System.out.println("  thumbnail shape=" + shapeOf(thumb));

        Tensor at1 = VideoOps.frameAt(video.toString(), Math.min(1.0, meta.durationSec * 0.1));
        System.out.println("  frameAt(~10%) shape=" + shapeOf(at1));

        // LLaVA / Qwen-VL style: 8 uniform keyframes
        List<Tensor> key8 = VideoOps.extractUniform(video.toString(), 8);
        System.out.println("  extractUniform(8) n=" + key8.size()
                + " first=" + shapeOf(key8.isEmpty() ? null : key8.get(0)));

        Tensor stacked = VideoOps.extractUniformStacked(video.toString(), 4);
        System.out.println("  extractUniformStacked(4)=" + shapeOf(stacked));

        // sparse fps sample (cap 12 frames so demo stays fast on long clips)
        List<Tensor> sparse = VideoOps.extractAtFps(video.toString(), 1.0, 12);
        System.out.println("  extractAtFps(1.0, max=12) n=" + sparse.size());

        // ── 3. VideoFile seek / iterator ──────────────────────────────────
        System.out.println("\n[3] VideoFile seek + every-N");
        try (VideoFile vf = FFmpegLoader.openVideo(video.toString())) {
            System.out.println("  meta via file: " + vf.width() + "x" + vf.height()
                    + " @" + vf.fps() + "fps dur=" + vf.duration() + "s codec=" + vf.codecName());
            vf.seek(Math.min(0.5, Math.max(0, vf.duration() * 0.05)));
            List<Tensor> every5 = vf.extractEveryN(5, 6);
            System.out.println("  after seek extractEveryN(5,6) n=" + every5.size());
        }

        // ── 4. VLM pack ──────────────────────────────────────────────────
        System.out.println("\n[4] sampleForVlm");
        Map<String, Object> vlm = VideoOps.sampleForVlm(video.toString(), 6);
        System.out.println("  count=" + vlm.get("count") + " stacked=" + shapeOf((Tensor) vlm.get("stacked")));

        // ── 5. OpenCV multimodal on extracted frames ─────────────────────
        System.out.println("\n[5] OpenCVOps on keyframes");
        if (!key8.isEmpty()) {
            Tensor f0 = key8.get(0);
            Tensor letter = OpenCVIO.letterbox(f0, 224, 224);
            System.out.println("  letterbox 224 → " + shapeOf(letter));

            Tensor batch = OpenCVOps.preprocessVlmFrames(key8.subList(0, Math.min(4, key8.size())), 224);
            System.out.println("  preprocessVlmFrames → " + shapeOf(batch));

            long hash = OpenCVOps.ahash(f0);
            System.out.println("  ahash=" + Long.toHexString(hash));

            if (key8.size() >= 2) {
                double energy = OpenCVOps.frameDiffEnergy(key8.get(0), key8.get(1));
                System.out.println("  frameDiffEnergy(0,1)=" + String.format("%.3f", energy));
            }

            // write a couple of debug frames
            OpenCVIO.writeImage(outDir.resolve("thumb.jpg").toString(), thumb != null ? thumb : f0);
            OpenCVIO.writeImage(outDir.resolve("letterbox.jpg").toString(), letter);
            OpenCVIO.writeImage(outDir.resolve("canny.jpg").toString(), OpenCVIO.canny(f0));
            System.out.println("  wrote debug frames under " + outDir);
        }

        // ── 6. Image enterprise pipeline ─────────────────────────────────
        if (Files.isRegularFile(image)) {
            System.out.println("\n[6] image preprocess");
            Tensor img = OpenCVIO.readImage(image.toString());
            Tensor imnet = OpenCVOps.preprocessImagenet(img, 224);
            Tensor clip = OpenCVOps.preprocessClip(img, 224);
            Tensor ocr = OpenCVOps.ocrBinarize(img);
            System.out.println("  imagenet=" + shapeOf(imnet) + " clip=" + shapeOf(clip)
                    + " ocrBin=" + shapeOf(ocr));
            OpenCVIO.writeImage(outDir.resolve("ocr_bin.png").toString(), ocr);
        }

        // ── 7. CLI ops (optional — needs system ffmpeg) ──────────────────
        System.out.println("\n[7] CLI FFmpeg ops (clip / frames-to-dir)");
        if (FFmpeg.isAvailable()) {
            Path clipOut = outDir.resolve("clip_2s.mp4");
            double start = 0.0;
            double dur = Math.min(2.0, Math.max(0.5, meta.durationSec * 0.02));
            try {
                VideoOps.clip(video.toString(), clipOut.toString(), start, dur, true);
                System.out.println("  clip → " + clipOut + " (" + Files.size(clipOut) + " bytes)");
            } catch (Exception e) {
                // stream-copy may fail on odd containers; retry re-encode
                try {
                    VideoOps.clip(video.toString(), clipOut.toString(), start, dur, false);
                    System.out.println("  clip (re-encode) → " + clipOut);
                } catch (Exception e2) {
                    System.out.println("  clip SKIP: " + e2.getMessage());
                }
            }
            Path framesDir = outDir.resolve("frames_1fps");
            try {
                List<Path> dumped = VideoOps.extractFramesToDir(video.toString(), framesDir.toString(), 0.5);
                System.out.println("  extractFramesToDir(0.5fps) n=" + dumped.size()
                        + " dir=" + framesDir);
            } catch (Exception e) {
                System.out.println("  extractFramesToDir SKIP: " + e.getMessage());
            }
        } else {
            System.out.println("  system ffmpeg binary not on PATH — CLI ops skipped"
                    + " (in-process libav path still works)");
        }

        System.out.println("\n=== done ===");
    }

    static String shapeOf(Tensor t) {
        if (t == null) return "null";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(',');
            sb.append(t.size(i));
        }
        return sb.append(']').toString();
    }
}
