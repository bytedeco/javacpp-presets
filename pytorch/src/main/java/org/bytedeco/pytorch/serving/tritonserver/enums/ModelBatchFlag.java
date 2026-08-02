package org.bytedeco.pytorch.serving.tritonserver.enums;

/**
 * Model batch property flags from {@code TRITONSERVER_ServerModelBatchProperties}.
 *
 * <p>Values are bit flags: {@code UNKNOWN=1}, {@code FIRST_DIM=2}.
 */
public enum ModelBatchFlag {
    UNKNOWN(1),
    FIRST_DIM(2);

    private final int code;

    ModelBatchFlag(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static boolean isSet(int flags, ModelBatchFlag flag) {
        return (flags & flag.code) != 0;
    }
}
