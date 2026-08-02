package org.bytedeco.pytorch.serving.tritonserver.enums;

/**
 * Model transaction property flags from {@code TRITONSERVER_ServerModelTransactionProperties}.
 *
 * <p>Values are bit flags: {@code ONE_TO_ONE=1}, {@code DECOUPLED=2}.
 */
public enum ModelTxnPropertyFlag {
    ONE_TO_ONE(1),
    DECOUPLED(2);

    private final int code;

    ModelTxnPropertyFlag(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static boolean isSet(int flags, ModelTxnPropertyFlag flag) {
        return (flags & flag.code) != 0;
    }
}
