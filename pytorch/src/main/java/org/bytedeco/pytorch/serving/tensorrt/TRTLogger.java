package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;
import org.bytedeco.pytorch.serving.tensorrt.internal.NativeLogger;

import java.io.PrintStream;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * TensorRT logger facade.
 *
 * <p>Corresponds to Python {@code tensorrt.Logger} / C++ {@code nvinfer1::ILogger}.
 * Severity values match {@code nvinfer1::ILogger::Severity}:
 * {@code INTERNAL_ERROR=0}, {@code ERROR=1}, {@code WARNING=2}, {@code INFO=3},
 * {@code VERBOSE=4}.
 *
 * <p>Messages at severity strictly greater than {@link #severity()} are filtered
 * (same rule as the common TensorRT sample Logger).
 */
public final class TRTLogger {
    /**
     * Log severity. Ordinals match {@code nvinfer1::ILogger::Severity}.
     */
    public enum Severity {
        INTERNAL_ERROR(0),
        ERROR(1),
        WARNING(2),
        INFO(3),
        VERBOSE(4);

        private final int code;

        Severity(int code) {
            this.code = code;
        }

        public int code() {
            return code;
        }

        public static Severity fromCode(int code) {
            for (Severity s : values()) {
                if (s.code == code) {
                    return s;
                }
            }
            throw new TrtInvalidArgumentException("Unknown Logger.Severity code: " + code);
        }
    }

    private static final AtomicReference<TRTLogger> DEFAULT =
            new AtomicReference<>(new TRTLogger(Severity.WARNING));

    private volatile Severity severity;
    private final PrintStream out;
    private final Object lock = new Object();
    private NativeLogger nativeLogger;

    public TRTLogger() {
        this(Severity.WARNING, System.err);
    }

    public TRTLogger(Severity severity) {
        this(severity, System.err);
    }

    public TRTLogger(Severity severity, PrintStream out) {
        this.severity = Objects.requireNonNull(severity, "severity");
        this.out = Objects.requireNonNull(out, "out");
    }

    /** Process-wide default logger (Python-style shared logger). */
    public static TRTLogger getDefaultLogger() {
        return DEFAULT.get();
    }

    public static void setDefaultLogger(TRTLogger TRTLogger) {
        DEFAULT.set(Objects.requireNonNull(TRTLogger, "logger"));
    }

    public Severity severity() {
        return severity;
    }

    /** Alias used by plan / Python {@code Logger.severity} setter style. */
    public void setLevel(Severity severity) {
        this.severity = Objects.requireNonNull(severity, "severity");
    }

    public void setLevel(int level) {
        setLevel(Severity.fromCode(level));
    }

    public void log(Severity severity, String message) {
        Objects.requireNonNull(severity, "severity");
        if (severity.code() > this.severity.code()) {
            return;
        }
        String msg = message == null ? "" : message;
        synchronized (lock) {
            out.println("[TRT][" + severity.name() + "] " + msg);
        }
    }

    public void log(int severityCode, String message) {
        log(Severity.fromCode(severityCode), message);
    }

    public void internalError(String message) {
        log(Severity.INTERNAL_ERROR, message);
    }

    public void error(String message) {
        log(Severity.ERROR, message);
    }

    public void warning(String message) {
        log(Severity.WARNING, message);
    }

    public void info(String message) {
        log(Severity.INFO, message);
    }

    public void verbose(String message) {
        log(Severity.VERBOSE, message);
    }

    /**
     * Lazily creates and returns the native {@code nvinfer1::ILogger} bridge.
     * Required when calling {@code createInferBuilder} / {@code createInferRuntime}.
     */
    public synchronized NativeLogger nativeLogger() {
        if (nativeLogger == null || nativeLogger.isNull()) {
            nativeLogger = new NativeLogger(this);
        }
        return nativeLogger;
    }
}
