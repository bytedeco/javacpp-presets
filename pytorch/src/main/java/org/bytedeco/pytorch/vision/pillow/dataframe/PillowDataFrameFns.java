package org.bytedeco.pytorch.vision.pillow.dataframe;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;
import org.bytedeco.pytorch.dataframe.media.MediaInterop;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.tensor.PillowTensors;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Wiring between Pillow DataFrames and {@code dataframe.media.*} / training paths.
 *
 * <p>Keeps hard dependencies one-way: pillow.dataframe → DataFrame + MediaInterop.
 * Media packages may optionally reflect into Pillow; this class is the explicit bridge.
 */
public final class PillowDataFrameFns {

    private PillowDataFrameFns() {}

    /**
     * Ensure {@code imageCol} cells are {@link ImageData} suitable for
     * {@link MediaInterop#toVisionBatch} / {@link MediaBridge}.
     */
    public static DataFrame toImageDataFrame(DataFrame df, String imageCol) {
        return PillowIO.ensureImageData(df, imageCol);
    }

    /** NCHW float {@code [0,1]} batch from Pillow-loaded frame. */
    public static Tensor toVisionBatch(DataFrame df, String imageCol) {
        DataFrame ready = toImageDataFrame(df, imageCol == null ? "image" : imageCol);
        return MediaInterop.toVisionBatch(ready, imageCol == null ? "image" : imageCol);
    }

    /** Per-row CHW tensors (null where cell missing). */
    public static List<Tensor> toVisionTensors(DataFrame df, String imageCol) {
        DataFrame ready = toImageDataFrame(df, imageCol == null ? "image" : imageCol);
        return MediaInterop.toVisionTensors(ready, imageCol == null ? "image" : imageCol);
    }

    /**
     * Build a one-column ImageData frame from a list of Pillow {@link Image}s
     * (copies pixels into ImageData; originals are not closed).
     */
    public static DataFrame fromPillowImages(List<Image> images, String imageCol) {
        Objects.requireNonNull(images, "images");
        String col = imageCol == null ? "image" : imageCol;
        DataFrame df = DataFrame.create();
        df.addColumn(col, Column.DType.IMAGE);
        df.addColumn("width", Column.DType.INT32);
        df.addColumn("height", Column.DType.INT32);
        df.addColumn("mode", Column.DType.STRING);
        for (Image im : images) {
            if (im == null) continue;
            int ri = df.addEmptyRow();
            df.set(ri, col, PillowTensors.toImageData(im));
            df.set(ri, "width", im.width());
            df.set(ri, "height", im.height());
            df.set(ri, "mode", im.mode());
        }
        return df;
    }

    public static DataFrame fromPillowImages(List<Image> images) {
        return fromPillowImages(images, "image");
    }

    /** Inverse of batch stack: NCHW/CHW → ImageData frame, then optional Pillow open via BI. */
    public static DataFrame fromVisionBatch(Tensor batch, String imageCol) {
        return MediaInterop.fromVisionBatch(batch, imageCol == null ? "image" : imageCol);
    }

    /**
     * Materialize live {@link Image} list from an ImageData column (caller owns close).
     */
    public static List<Image> toPillowImages(DataFrame df, String imageCol) {
        String col = imageCol == null ? "image" : imageCol;
        List<Image> out = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, col);
            if (cell instanceof Image im) {
                out.add(im);
            } else if (cell instanceof ImageData id && id.getImage() != null) {
                out.add(PillowTensors.fromImageData(id));
            } else {
                out.add(null);
            }
        }
        return out;
    }

    /**
     * Re-decode path column with Pillow (when cells only have paths / stale ImageData).
     */
    public static DataFrame redecodePillow(DataFrame df, String pathCol, String imageCol) throws Exception {
        Objects.requireNonNull(df, "df");
        String pcol = pathCol == null ? "path" : pathCol;
        String icol = imageCol == null ? "image" : imageCol;
        if (!df.hasColumn(pcol)) {
            throw new IllegalArgumentException("no path column " + pcol);
        }
        if (!df.hasColumn(icol)) {
            df.addColumn(icol, Column.DType.IMAGE);
        }
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object pv = out.get(r, pcol);
            if (pv == null) continue;
            try {
                Image im = Image.open(pv.toString());
                out.set(r, icol, PillowTensors.toImageData(im));
                if (out.hasColumn("width")) out.set(r, "width", im.width());
                if (out.hasColumn("height")) out.set(r, "height", im.height());
                if (out.hasColumn("mode")) out.set(r, "mode", im.mode());
                im.close();
            } catch (Exception ignored) {
            }
        }
        return out;
    }
}
