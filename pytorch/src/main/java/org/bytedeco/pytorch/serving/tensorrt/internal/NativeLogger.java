//package org.bytedeco.pytorch.serving.tensorrt.internal;
//
//import org.bytedeco.tensorrt.nvinfer.ILogger;
//import org.bytedeco.pytorch.serving.tensorrt.TRTLogger;
//
///**
// * JavaCPP callback bridge from {@code nvinfer1::ILogger} to
// * {@link TRTLogger}.
// *
// * <p>TensorRT builder/runtime require a live {@code ILogger} reference for the
// * lifetime of those objects; callers must keep this instance reachable.
// */
//public final class NativeLogger extends ILogger {
//    private final TRTLogger delegate;
//
//    public NativeLogger(TRTLogger delegate) {
//        super(delegate);
//        this.delegate = delegate;
//    }
//
//    @Override
//    public void log(Severity severity, String msg) {
//        if (delegate == null) {
//            return;
//        }
//        int code = severity == null ? TRTLogger.Severity.WARNING.code() : severity.value;
//        TRTLogger.Severity mapped;
//        try {
//            mapped = TRTLogger.Severity.fromCode(code);
//        } catch (RuntimeException e) {
//            mapped = TRTLogger.Severity.WARNING;
//        }
//        delegate.log(mapped, msg);
//    }
//}
