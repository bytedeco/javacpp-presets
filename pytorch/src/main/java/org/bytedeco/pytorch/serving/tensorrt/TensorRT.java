package org.bytedeco.pytorch.serving.tensorrt;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Facade entry points for the high-level TensorRT Java API.
 *
 * <p>Mirrors the convenience of Python {@code import tensorrt as trt} plus the
 * common Builder / Runtime construction patterns. Native binding:
 * {@code org.bytedeco.tensorrt} 10.15.
 *
 * <pre>{@code
 * // Build from ONNX
 * Engine engine = TensorRT.engineBuilder()
 *     .withModelPath("/models/model.onnx")
 *     .withEnginePath("/models/model.engine")
 *     .enableFp16(true)
 *     .build();
 *
 * // Or load an existing plan
 * Engine engine = TensorRT.loadEngine("/models/model.engine");
 *
 * InferenceResult result = engine.infer(Map.of("input", Tensor.of(data, 1, 3, 224, 224)));
 * float[] out = result.get("output").toFloatArray();
 * engine.close();
 * }</pre>
 */
public final class TensorRT {
    private TensorRT() {}

    /** Create an {@link TRTEngineBuilder} with default {@link TrtOption}. */
    public static TRTEngineBuilder engineBuilder() {
        return TRTEngineBuilder.create();
    }

    public static TRTEngineBuilder engineBuilder(TrtOption trtOptions) {
        return TRTEngineBuilder.create(trtOptions);
    }

    public static TRTEngineBuilder engineBuilder(TRTLogger TRTLogger) {
        return TRTEngineBuilder.create(new TrtOption().logger(TRTLogger));
    }

    /** Deserialize a serialized engine plan. */
    public static TRTEngine loadEngine(byte[] plan) {
        return TRTEngine.fromSerialized(plan);
    }

    public static TRTEngine loadEngine(byte[] plan, TrtOption trtOptions) {
        return TRTEngine.fromSerialized(plan, trtOptions);
    }

    public static TRTEngine loadEngine(String path) {
        return TRTEngine.load(path);
    }

    public static TRTEngine loadEngine(String path, TrtOption trtOptions) {
        return TRTEngine.load(path, trtOptions);
    }

    public static TRTEngine loadEngine(Path path, TrtOption trtOptions) {
        return TRTEngine.load(path, trtOptions);
    }

    /** Process-wide default logger (Python-style). */
    public static TRTLogger defaultLogger() {
        return TRTLogger.getDefaultLogger();
    }

    public static void setDefaultLogger(TRTLogger TRTLogger) {
        TRTLogger.setDefaultLogger(Objects.requireNonNull(TRTLogger, "logger"));
    }

    /**
     * Library / binding identity string for diagnostics (not the native TRT
     * SO version — that requires a successful native load).
     */
    public static String bindingVersion() {
        return "bytedeco-tensorrt-10.15-1.5.13";
    }
}
