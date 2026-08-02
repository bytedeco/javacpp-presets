package org.bytedeco.pytorch.vision.pillow.dataframe;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.enums.Resampling;

import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Column-oriented helpers over Pillow images inside a {@link DataFrame}.
 *
 * <p>Thin façade on {@link PillowIO#map} / {@link PillowIO#filter} with PIL-style names.
 */
public final class PillowColumn {

    private PillowColumn() {}

    public static DataFrame map(DataFrame df, String imageCol, Function<Image, Image> fn) {
        return PillowIO.map(df, imageCol, fn);
    }

    public static DataFrame filter(DataFrame df, String imageCol, Predicate<Image> pred) {
        return PillowIO.filter(df, imageCol, pred);
    }

    public static DataFrame resize(DataFrame df, String imageCol, int w, int h) {
        return PillowIO.resize(df, imageCol, w, h, Resampling.BICUBIC);
    }

    public static DataFrame resize(DataFrame df, String imageCol, int w, int h, Resampling resample) {
        return PillowIO.resize(df, imageCol, w, h, resample);
    }

    public static DataFrame convert(DataFrame df, String imageCol, String mode) {
        return PillowIO.convert(df, imageCol, mode);
    }

    public static DataFrame thumbnail(DataFrame df, String imageCol, int maxW, int maxH) {
        Objects.requireNonNull(df, "df");
        return PillowIO.map(df, imageCol, im -> {
            Image c = im.copy();
            c.thumbnail(new int[]{maxW, maxH});
            return c;
        });
    }

    public static DataFrame crop(DataFrame df, String imageCol, int left, int upper, int right, int lower) {
        return PillowIO.map(df, imageCol, im -> im.crop(left, upper, right, lower));
    }

    public static DataFrame transpose(DataFrame df, String imageCol,
                                     org.bytedeco.pytorch.vision.pillow.enums.Transpose method) {
        return PillowIO.map(df, imageCol, im -> im.transpose(method));
    }
}
