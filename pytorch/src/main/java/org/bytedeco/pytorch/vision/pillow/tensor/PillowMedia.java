package org.bytedeco.pytorch.vision.pillow.tensor;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Optional interop between Pillow {@link Image} and OpenCV / FFmpeg stacks already in this repo.
 *
 * <p><b>Default path is always pure Java</b> (BufferedImage / ImageTensors). OpenCV and FFmpeg are
 * probed at runtime via {@link MediaBridge#isOpenCvAvailable()} /
 * {@link MediaBridge#isFFmpegAvailable()} — never hard-required.
 *
 * <pre>
 *   Image im = Image.open("a.png");
 *   Object mat = PillowMedia.imageToMat(im);          // BGR Mat when OpenCV present
 *   Image back = PillowMedia.matToImage(mat);
 *
 *   Tensor frame = …; // CHW float [0,255] from VideoFrame.toTensorChw()
 *   Image frameIm = PillowMedia.fromFFmpegChw(frame);
 * </pre>
 */
public final class PillowMedia {

    private PillowMedia() {}

    // ── capability ────────────────────────────────────────────────────────

    public static boolean isOpenCvAvailable() {
        return MediaBridge.isOpenCvAvailable();
    }

    public static boolean isFFmpegAvailable() {
        return MediaBridge.isFFmpegAvailable();
    }

    public static Map<String, Object> capabilities() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("opencv", isOpenCvAvailable());
        m.put("ffmpeg", isFFmpegAvailable());
        m.put("imageToMat", isOpenCvAvailable());
        m.put("matToImage", isOpenCvAvailable());
        m.put("openCvDecode", isOpenCvAvailable());
        m.put("ffmpegFrameToImage", isFFmpegAvailable());
        m.put("defaultBackend", "PURE_JAVA");
        return m;
    }

    // ── ImageData / BufferedImage (always) ────────────────────────────────

    public static ImageData toImageData(Image image) {
        return PillowTensors.toImageData(image);
    }

    public static Image fromImageData(ImageData id) {
        return PillowTensors.fromImageData(id);
    }

    public static BufferedImage toBufferedImage(Image image) {
        Objects.requireNonNull(image, "image");
        return image.toBufferedImage();
    }

    public static Image fromBufferedImage(BufferedImage bi) {
        return Image.fromBufferedImage(bi);
    }

    // ── Tensor (always, via ImageTensors) ─────────────────────────────────

    /** CHW float {@code [0,1]}. */
    public static Tensor toTensor(Image image) {
        return PillowTensors.toTensor(image);
    }

    public static Image fromTensor(Tensor t) {
        return PillowTensors.fromTensor(t);
    }

    public static Image fromTensor(Tensor t, String mode) {
        return PillowTensors.fromTensor(t, mode);
    }

    /**
     * CHW float in {@code [0,255]} (OpenCV / VideoTensors style).
     */
    public static Tensor toTensor255(Image image) {
        Objects.requireNonNull(image, "image");
        Tensor unit = PillowTensors.toTensor(image);
        return unit.mul(new Scalar(255.0));
    }

    public static Image fromTensor255(Tensor t255) {
        Objects.requireNonNull(t255, "tensor");
        Tensor unit = t255.div(new Scalar(255.0));
        return PillowTensors.fromTensor(unit);
    }

    // ── OpenCV Mat (optional) ─────────────────────────────────────────────

    /**
     * Pillow Image → OpenCV BGR {@code Mat} (as Object to keep the dep soft).
     * @throws IllegalStateException if OpenCV is not available
     */
    public static Object imageToMat(Image image) throws Exception {
        Objects.requireNonNull(image, "image");
        requireOpenCv("imageToMat");
        ImageData id = PillowTensors.toImageData(image);
        return MediaBridge.imageToMat(id);
    }

    /**
     * OpenCV BGR Mat → Pillow Image (RGB/L).
     * @throws IllegalStateException if OpenCV is not available
     */
    public static Image matToImage(Object mat) throws Exception {
        Objects.requireNonNull(mat, "mat");
        requireOpenCv("matToImage");
        ImageData id = MediaBridge.matToImage(mat);
        return PillowTensors.fromImageData(id);
    }

    /**
     * Decode path with OpenCV (when available) and wrap as Pillow Image.
     * Falls back to {@link Image#open} if OpenCV missing or decode fails and {@code fallback=true}.
     */
    public static Image openWithOpenCv(String path, boolean fallback) throws Exception {
        Objects.requireNonNull(path, "path");
        if (isOpenCvAvailable()) {
            try {
                ImageData id = MediaBridge.loadImageOpenCv(path, false);
                return PillowTensors.fromImageData(id);
            } catch (Exception e) {
                if (!fallback) throw e;
            }
        } else if (!fallback) {
            requireOpenCv("openWithOpenCv");
        }
        return Image.open(path);
    }

    public static Image openWithOpenCv(String path) throws Exception {
        return openWithOpenCv(path, true);
    }

    /**
     * Apply an OpenCVIO-style tensor op and return a new Pillow Image.
     * {@code op} receives CHW float {@code [0,255]} and must return the same layout.
     */
    public static Image mapOpenCvTensor(Image image, OpenCvTensorOp op) throws Exception {
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(op, "op");
        requireOpenCv("mapOpenCvTensor");
        Tensor t255 = toTensor255(image);
        Tensor out = op.apply(t255);
        return fromTensor255(out);
    }

    @FunctionalInterface
    public interface OpenCvTensorOp {
        Tensor apply(Tensor chw255) throws Exception;
    }

    /**
     * Common OpenCV ops on a Pillow image (resize / gray / blur) via reflection-free
     * OpenCVIO when present.
     */
    public static Image openCvResize(Image image, int height, int width) throws Exception {
        requireOpenCv("openCvResize");
        Class<?> io = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
        Tensor t255 = toTensor255(image);
        Tensor out = (Tensor) io.getMethod("resize", Tensor.class, int.class, int.class)
                .invoke(null, t255, height, width);
        return fromTensor255(out);
    }

    public static Image openCvGaussianBlur(Image image, int ksize) throws Exception {
        requireOpenCv("openCvGaussianBlur");
        Class<?> io = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
        Tensor t255 = toTensor255(image);
        Tensor out = (Tensor) io.getMethod("gaussianBlur", Tensor.class, int.class)
                .invoke(null, t255, ksize);
        return fromTensor255(out);
    }

    public static Image openCvToGray(Image image) throws Exception {
        requireOpenCv("openCvToGray");
        Class<?> io = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
        Tensor t255 = toTensor255(image);
        Tensor out = (Tensor) io.getMethod("toGrayscale", Tensor.class).invoke(null, t255);
        return fromTensor255(out).convert("L");
    }

    // ── FFmpeg frames (optional) ──────────────────────────────────────────

    /**
     * Convert an FFmpeg {@code VideoFrame} (passed as Object) to Pillow Image via
     * {@code toTensorChw()} → CHW float [0,255] → Image.
     */
    public static Image fromVideoFrame(Object videoFrame) throws Exception {
        Objects.requireNonNull(videoFrame, "videoFrame");
        requireFFmpeg("fromVideoFrame");
        Tensor chw = (Tensor) videoFrame.getClass().getMethod("toTensorChw").invoke(videoFrame);
        return fromTensor255(chw);
    }

    /**
     * CHW float [0,255] tensor (VideoTensors / VideoFrame layout) → Pillow Image.
     */
    public static Image fromFFmpegChw(Tensor chw255) {
        return fromTensor255(chw255);
    }

    /**
     * Pillow Image → CHW float [0,255] suitable for {@code VideoFrame.fromTensor} / stacking.
     */
    public static Tensor toFFmpegChw(Image image) {
        return toTensor255(image);
    }

    /**
     * Build a VideoFrame from Pillow Image via reflection ({@code VideoFrame.fromTensor}).
     * Format default {@code rgb24}.
     */
    public static Object toVideoFrame(Image image, String pixFmt) throws Exception {
        Objects.requireNonNull(image, "image");
        requireFFmpeg("toVideoFrame");
        Tensor chw = toFFmpegChw(image);
        Class<?> vf = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.VideoFrame");
        String fmt = pixFmt == null ? "rgb24" : pixFmt;
        return vf.getMethod("fromTensor", Tensor.class, String.class).invoke(null, chw, fmt);
    }

    public static Object toVideoFrame(Image image) throws Exception {
        return toVideoFrame(image, "rgb24");
    }

    /**
     * Decode a video file to a list of Pillow Images (uniform sample or all frames —
     * uses {@code FFmpegLoader.decodeVideo} which returns CHW float frames).
     * Large videos: prefer {@link #ffmpegFrames(String, int)} with a cap.
     */
    public static List<Image> ffmpegFrames(String videoPath) throws Exception {
        return ffmpegFrames(videoPath, Integer.MAX_VALUE);
    }

    public static List<Image> ffmpegFrames(String videoPath, int maxFrames) throws Exception {
        Objects.requireNonNull(videoPath, "videoPath");
        requireFFmpeg("ffmpegFrames");
        Class<?> loader = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.FFmpegLoader");
        @SuppressWarnings("unchecked")
        List<Tensor> frames = (List<Tensor>) loader.getMethod("decodeVideo", String.class)
                .invoke(null, videoPath);
        List<Image> out = new ArrayList<>();
        int n = 0;
        for (Tensor t : frames) {
            if (n++ >= maxFrames) break;
            if (t == null) continue;
            out.add(fromFFmpegChw(t));
        }
        return out;
    }

    /** Single frame at timestamp (seconds) via FFmpegLoader.frameAt. */
    public static Image ffmpegFrameAt(String videoPath, double seconds) throws Exception {
        Objects.requireNonNull(videoPath, "videoPath");
        requireFFmpeg("ffmpegFrameAt");
        Class<?> loader = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.FFmpegLoader");
        Tensor t = (Tensor) loader.getMethod("frameAt", String.class, double.class)
                .invoke(null, videoPath, seconds);
        return fromFFmpegChw(t);
    }

    // ── round-trip helpers ────────────────────────────────────────────────

    /**
     * Image → ImageData → (optional OpenCV Mat) → ImageData → Image.
     * Useful for C15-style interop checks.
     */
    public static Image roundTripImageData(Image image) {
        ImageData id = toImageData(image);
        return fromImageData(id);
    }

    public static Image roundTripOpenCv(Image image) throws Exception {
        Object mat = imageToMat(image);
        return matToImage(mat);
    }

    public static Image roundTripTensor(Image image) {
        Tensor t = toTensor(image);
        return fromTensor(t, image.mode());
    }

    // ── internals ─────────────────────────────────────────────────────────

    private static void requireOpenCv(String op) {
        if (!isOpenCvAvailable()) {
            throw new IllegalStateException(
                    "OpenCV not available for PillowMedia." + op
                            + " — use pure-Java Image path or enable javacpp-opencv");
        }
    }

    private static void requireFFmpeg(String op) {
        if (!isFFmpegAvailable()) {
            throw new IllegalStateException(
                    "FFmpeg not available for PillowMedia." + op
                            + " — use pure-Java Image path or enable javacpp-ffmpeg");
        }
    }
}
