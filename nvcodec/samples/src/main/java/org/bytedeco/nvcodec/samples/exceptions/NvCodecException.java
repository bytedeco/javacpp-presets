package org.bytedeco.nvcodec.samples.exceptions;

public class NvCodecException extends Exception {
    public NvCodecException(int result) {
        super("NvCodec error number : " + result);
    }

    public NvCodecException(String message, int errorCode) {
        super(String.format("%s (error number : %n)", errorCode));
    }
}
