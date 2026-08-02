package org.bytedeco.pytorch.serving.tensorrt.enums;
import org.bytedeco.pytorch.nn.options.*;

//import org.bytedeco.pytorch.serving.tensorrt.TrtOptions;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;

/**
 * Builder precision / feature flags.
 *
 * <p>Codes are bit <em>positions</em> matching {@code nvinfer1::BuilderFlag} /
 * Python {@code tensorrt.BuilderFlag} (bytedeco
 * {@code org.bytedeco.tensorrt.global.nvinfer.BuilderFlag}). Passed to
 * {@code IBuilderConfig.setFlag(int)} as the enum ordinal / {@code .value}.
 *
 * <p>Only flags commonly used from Python samples are exposed; less common
 * flags can still be passed via {@link } once added here.
 */
public enum BuilderFlag {
    /** @deprecated in TRT 10.12 — superseded by strong typing; still widely used. */
    FP16(0),
    /** @deprecated in TRT 10.12 — superseded by strong typing. */
    INT8(1),
    DEBUG(2),
    GPU_FALLBACK(3),
    REFIT(4),
    DISABLE_TIMING_CACHE(5),
    TF32(6),
    SPARSE_WEIGHTS(7),
    SAFETY_SCOPE(8),
    OBEY_PRECISION_CONSTRAINTS(9),
    PREFER_PRECISION_CONSTRAINTS(10),
    DIRECT_IO(11),
    REJECT_EMPTY_ALGORITHMS(12),
    VERSION_COMPATIBLE(13),
    EXCLUDE_LEAN_RUNTIME(14),
    FP8(15),
    ERROR_ON_PRECISION_CONSTRAINTS(16),
    BF16(17),
    DISABLE_COMPILATION_CACHE(18),
    WEIGHT_STREAMING(19),
    INT4(20),
    STRIP_PLAN(21),
    REFIT_IDENTICAL(22);

    private final int code;

    BuilderFlag(int code) {
        this.code = code;
    }

    /** {@code nvinfer1::BuilderFlag} ordinal ({@code .value}). */
    public int code() {
        return code;
    }

    public static BuilderFlag fromCode(int code) {
        for (BuilderFlag f : values()) {
            if (f.code == code) {
                return f;
            }
        }
        throw new TrtInvalidArgumentException("Unknown BuilderFlag code: " + code);
    }
}
