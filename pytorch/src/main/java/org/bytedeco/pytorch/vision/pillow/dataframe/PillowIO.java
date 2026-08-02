package org.bytedeco.pytorch.vision.pillow.dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.media.MediaInterop;
import org.bytedeco.pytorch.dataframe.media.MultimodalIO;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.Pillow;
import org.bytedeco.pytorch.vision.pillow.enums.Resampling;
import org.bytedeco.pytorch.vision.pillow.features.Features;
import org.bytedeco.pytorch.vision.pillow.tensor.PillowTensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * DataFrame batch I/O for Pillow images.
 *
 * <p>Path enumeration reuses {@link MultimodalIO#expand}; decode defaults to
 * {@link Image#open} so EXIF/mode/multi-frame stay on the Pillow path.
 * Columns: {@code path}, {@code image} ({@link ImageData}), optional
 * {@code width}/{@code height}/{@code mode}/{@code format}.
 */
public final class PillowIO {

    /** Image extensions recognized by Pillow batch loaders (includes PPM). */
    public static final String[] IMAGE_EXTS = {
            ".png", ".jpg", ".jpeg", ".jpe", ".gif", ".bmp", ".dib",
            ".webp", ".tiff", ".tif", ".ppm", ".pgm", ".pbm", ".jp2", ".j2k"
    };

    private PillowIO() {}

    // ── list / expand ─────────────────────────────────────────────────────

    /** List image paths under a directory or glob (delegates to MultimodalIO). */
    public static List<Path> listImages(String pathOrGlob) throws Exception {
        return MultimodalIO.expand(pathOrGlob, IMAGE_EXTS);
    }

    public static List<Path> expand(String pathOrGlob) throws Exception {
        return listImages(pathOrGlob);
    }

    // ── read ──────────────────────────────────────────────────────────────

    /**
     * Batch-load images with the Pillow decoder.
     * Columns: {@code path}, {@code image} (ImageData), {@code width}, {@code height},
     * {@code mode}, {@code format}.
     */
    public static DataFrame readImages(String pathOrGlob) throws Exception {
        return readImages(pathOrGlob, true, false);
    }

    public static DataFrame readImages(String pathOrGlob, boolean withMeta) throws Exception {
        return readImages(pathOrGlob, withMeta, false);
    }

    /**
     * @param keepPillowImage when true, also stores a column {@code pillow} with live {@link Image}
     *                        handles (caller must {@link Image#close()} if holding long-lived refs)
     */
    public static DataFrame readImages(String pathOrGlob, boolean withMeta, boolean keepPillowImage)
            throws Exception {
        Objects.requireNonNull(pathOrGlob, "pathOrGlob");
        Pillow.init();
        List<Path> files = listImages(pathOrGlob);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("image", Column.DType.IMAGE);
        if (keepPillowImage) {
            // Object column for live Image handles
            df.addColumn("pillow", Column.DType.BINARY);
        }
        if (withMeta) {
            df.addColumn("width", Column.DType.INT32);
            df.addColumn("height", Column.DType.INT32);
            df.addColumn("mode", Column.DType.STRING);
            df.addColumn("format", Column.DType.STRING);
        }
        for (Path p : files) {
            try {
                Image im = Image.open(p);
                String mode = im.mode();
                String fmt = im.format() != null ? im.format() : formatOf(p);
                ImageData id = PillowTensors.toImageData(im);
                id.setPath(p.toString());
                if (fmt != null && id.getFormat() == null) {
                    id.setFormat(fmt.toLowerCase(Locale.ROOT));
                }
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "image", id);
                if (keepPillowImage) {
                    df.set(ri, "pillow", im);
                } else {
                    im.close();
                }
                if (withMeta) {
                    df.set(ri, "width", id.getWidth());
                    df.set(ri, "height", id.getHeight());
                    df.set(ri, "mode", mode);
                    df.set(ri, "format", fmt);
                }
            } catch (Exception ignored) {
                // skip unreadable files (same policy as MultimodalIO)
            }
        }
        return df;
    }

    private static String formatOf(Path p) {
        String n = p.getFileName() == null ? "" : p.getFileName().toString();
        int dot = n.lastIndexOf('.');
        if (dot < 0) return null;
        return n.substring(dot + 1).toUpperCase(Locale.ROOT);
    }

    // ── write ─────────────────────────────────────────────────────────────

    public static DataFrame writeImages(DataFrame df, String imageCol, String dir) throws IOException {
        return writeImages(df, imageCol, dir, null);
    }

    /**
     * Save each {@link ImageData} (or {@link Image}) cell under {@code dir}.
     * Uses row {@code path} basename when present; otherwise {@code image_<row>.<fmt>}.
     */
    public static DataFrame writeImages(DataFrame df, String imageCol, String dir, String format)
            throws IOException {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(imageCol, "imageCol");
        Objects.requireNonNull(dir, "dir");
        if (!df.hasColumn(imageCol)) {
            throw new IllegalArgumentException("no column " + imageCol);
        }
        Path outDir = Path.of(dir);
        Files.createDirectories(outDir);
        String fmt = format == null ? "png" : format.toLowerCase(Locale.ROOT);
        DataFrame out = df.copy();
        boolean hasPath = out.hasColumn("path");
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, imageCol);
            Image im = null;
            boolean closeIm = false;
            try {
                if (cell instanceof Image img) {
                    im = img;
                } else if (cell instanceof ImageData id) {
                    im = PillowTensors.fromImageData(id);
                    closeIm = true;
                } else {
                    continue;
                }
                String base;
                if (hasPath) {
                    Object pv = out.get(r, "path");
                    if (pv != null) {
                        base = Path.of(pv.toString()).getFileName().toString();
                        int d = base.lastIndexOf('.');
                        if (d > 0) base = base.substring(0, d);
                    } else {
                        base = "image_" + r;
                    }
                } else {
                    base = "image_" + r;
                }
                Path dest = outDir.resolve(base + "." + fmt);
                im.save(dest, fmt.toUpperCase(Locale.ROOT), java.util.Map.of());
            } catch (Exception e) {
                throw new IOException("writeImages failed at row " + r + ": " + e.getMessage(), e);
            } finally {
                if (closeIm && im != null) {
                    try {
                        im.close();
                    } catch (Exception ignored) {
                    }
                }
            }
        }
        return out;
    }

    // ── map / filter ──────────────────────────────────────────────────────

    /**
     * Map ImageData/Image column with a Pillow transform; result written back as ImageData.
     */
    public static DataFrame map(DataFrame df, String imageCol, Function<Image, Image> fn) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(imageCol, "imageCol");
        Objects.requireNonNull(fn, "fn");
        if (!df.hasColumn(imageCol)) {
            throw new IllegalArgumentException("no column " + imageCol);
        }
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, imageCol);
            Image im = null;
            boolean closeSrc = false;
            try {
                if (cell instanceof Image img) {
                    im = img;
                } else if (cell instanceof ImageData id && id.getImage() != null) {
                    im = PillowTensors.fromImageData(id);
                    closeSrc = true;
                } else {
                    continue;
                }
                Image mapped = fn.apply(im);
                out.set(r, imageCol, PillowTensors.toImageData(mapped));
                if (out.hasColumn("width")) out.set(r, "width", mapped.width());
                if (out.hasColumn("height")) out.set(r, "height", mapped.height());
                if (out.hasColumn("mode")) out.set(r, "mode", mapped.mode());
                if (mapped != im) {
                    try {
                        mapped.close();
                    } catch (Exception ignored) {
                    }
                }
            } finally {
                if (closeSrc && im != null) {
                    try {
                        im.close();
                    } catch (Exception ignored) {
                    }
                }
            }
        }
        return out;
    }

    /** Keep rows whose image satisfies {@code pred}. */
    public static DataFrame filter(DataFrame df, String imageCol, Predicate<Image> pred) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(imageCol, "imageCol");
        Objects.requireNonNull(pred, "pred");
        if (!df.hasColumn(imageCol)) {
            throw new IllegalArgumentException("no column " + imageCol);
        }
        DataFrame out = DataFrame.create();
        for (int c = 0; c < df.columnCount(); c++) {
            Column src = df.column(c);
            out.addColumn(src.name(), src.dtype());
        }
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, imageCol);
            Image im = null;
            boolean closeSrc = false;
            boolean keep = false;
            try {
                if (cell instanceof Image img) {
                    im = img;
                    keep = pred.test(im);
                } else if (cell instanceof ImageData id && id.getImage() != null) {
                    im = PillowTensors.fromImageData(id);
                    closeSrc = true;
                    keep = pred.test(im);
                } else {
                    keep = false;
                }
            } finally {
                if (closeSrc && im != null) {
                    try {
                        im.close();
                    } catch (Exception ignored) {
                    }
                }
            }
            if (keep) {
                int ri = out.addEmptyRow();
                for (int c = 0; c < df.columnCount(); c++) {
                    Column src = df.column(c);
                    out.set(ri, src.name(), src.get(r));
                }
            }
        }
        return out;
    }

    /** Convenience: resize every image in column. */
    public static DataFrame resize(DataFrame df, String imageCol, int w, int h, Resampling resample) {
        Resampling r = resample == null ? Resampling.BICUBIC : resample;
        return map(df, imageCol, im -> im.resize(w, h, r));
    }

    public static DataFrame convert(DataFrame df, String imageCol, String mode) {
        Objects.requireNonNull(mode, "mode");
        return map(df, imageCol, im -> im.convert(mode));
    }

    // ── vision batch ──────────────────────────────────────────────────────

    /** Stack ImageData column as NCHW float {@code [0,1]} via {@link MediaInterop}. */
    public static Tensor toVisionBatch(DataFrame df, String imageCol) {
        return MediaInterop.toVisionBatch(df, imageCol == null ? "image" : imageCol);
    }

    /** Ensure column holds ImageData (convert live Image cells). */
    public static DataFrame ensureImageData(DataFrame df, String imageCol) {
        Objects.requireNonNull(df, "df");
        String col = imageCol == null ? "image" : imageCol;
        if (!df.hasColumn(col)) {
            throw new IllegalArgumentException("no column " + col);
        }
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            if (cell instanceof Image im) {
                out.set(r, col, PillowTensors.toImageData(im));
            }
        }
        return out;
    }

    // ── capability ────────────────────────────────────────────────────────

    public static boolean check_pil() {
        return Features.checkModule("pil");
    }

    public static boolean checkPil() {
        return check_pil();
    }

    public static List<String> supportedCodecs() {
        return Features.getSupportedCodecs();
    }
}
