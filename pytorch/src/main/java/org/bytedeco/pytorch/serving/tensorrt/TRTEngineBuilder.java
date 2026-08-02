package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.tensorrt.global.nvinfer.MemoryPoolType;
import org.bytedeco.tensorrt.global.nvinfer.NetworkDefinitionCreationFlag;
import org.bytedeco.tensorrt.nvinfer.*;
import org.bytedeco.tensorrt.nvonnxparser.IParser;
import org.bytedeco.pytorch.serving.tensorrt.enums.BuilderFlag;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.ExecutionException;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtNotFoundException;
import org.bytedeco.pytorch.serving.tensorrt.internal.NativeError;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

import static org.bytedeco.tensorrt.global.nvinfer.createInferBuilder;
import static org.bytedeco.tensorrt.global.nvonnxparser.createParser;

/**
 * Builds a serialized TensorRT engine (plan) from an ONNX model.
 *
 * <p>Corresponds to the common Python workflow:
 * <pre>{@code
 * builder = trt.Builder(logger)
 * network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
 * parser = trt.OnnxParser(network, logger)
 * parser.parse_from_file(onnx_path)
 * config = builder.create_builder_config()
 * config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
 * engine_bytes = builder.build_serialized_network(network, config)
 * }</pre>
 *
 * <p>Chainable setters mirror the plan's {@code withModelPath} / {@code withLogger} style.
 */
public final class TRTEngineBuilder {
    private TrtOption trtOptions = new TrtOption();
    private Path onnxModelPath;
    private Path engineOutputPath;

    public TRTEngineBuilder() {}

    public TRTEngineBuilder(TrtOption trtOptions) {
        this.trtOptions = trtOptions == null ? new TrtOption() : new TrtOption(trtOptions);
    }

    public static TRTEngineBuilder create() {
        return new TRTEngineBuilder();
    }

    public static TRTEngineBuilder create(TrtOption trtOptions) {
        return new TRTEngineBuilder(trtOptions);
    }

    public TRTEngineBuilder withOptions(TrtOption trtOptions) {
        this.trtOptions = trtOptions == null ? new TrtOption() : new TrtOption(trtOptions);
        return this;
    }

    public TRTEngineBuilder withLogger(TRTLogger logger) {
        this.trtOptions.logger(logger);
        return this;
    }

    public TRTEngineBuilder withModelPath(String onnxPath) {
        if (onnxPath == null || onnxPath.isBlank()) {
            throw new TrtInvalidArgumentException("onnx model path must be non-empty");
        }
        this.onnxModelPath = Path.of(onnxPath);
        return this;
    }

    public TRTEngineBuilder withModelPath(Path onnxPath) {
        this.onnxModelPath = Objects.requireNonNull(onnxPath, "onnxPath");
        return this;
    }

    public TRTEngineBuilder withEnginePath(String enginePath) {
        if (enginePath == null || enginePath.isBlank()) {
            throw new TrtInvalidArgumentException("engine path must be non-empty");
        }
        this.engineOutputPath = Path.of(enginePath);
        return this;
    }

    public TRTEngineBuilder withEnginePath(Path enginePath) {
        this.engineOutputPath = Objects.requireNonNull(enginePath, "enginePath");
        return this;
    }

    public TRTEngineBuilder withWorkspaceBytes(long bytes) {
        this.trtOptions.workspaceBytes(bytes);
        return this;
    }

    public TRTEngineBuilder withDevice(int deviceIndex) {
        this.trtOptions.deviceIndex(deviceIndex);
        return this;
    }

    public TRTEngineBuilder enableFp16(boolean enable) {
        this.trtOptions.enableFp16(enable);
        return this;
    }

    public TRTEngineBuilder enableInt8(boolean enable) {
        this.trtOptions.enableInt8(enable);
        return this;
    }

    public TRTEngineBuilder stronglyTyped(boolean enable) {
        this.trtOptions.stronglyTyped(enable);
        return this;
    }

    public TrtOption options() {
        return trtOptions;
    }

    public Path onnxModelPath() {
        return onnxModelPath;
    }

    public Path engineOutputPath() {
        return engineOutputPath;
    }

    /**
     * Build a serialized engine and optionally write it to {@link #engineOutputPath()}.
     *
     * @return engine plan bytes (TensorRT serialized network)
     */
    public byte[] buildSerializedNetwork() {
        if (onnxModelPath == null) {
            throw new TrtInvalidArgumentException("ONNX model path is required (withModelPath)");
        }
        if (!Files.isRegularFile(onnxModelPath)) {
            throw new TrtNotFoundException("ONNX model not found: " + onnxModelPath);
        }

        TRTLogger logger = trtOptions.logger() != null ? trtOptions.logger() : TRTLogger.getDefaultLogger();
        ILogger nativeLogger = logger.nativeLogger();

        NativeError.checkCuda(
                org.bytedeco.cuda.global.cudart.cudaSetDevice(trtOptions.deviceIndex()),
                "cudaSetDevice(" + trtOptions.deviceIndex() + ")");

        IBuilder builder = createInferBuilder(nativeLogger);
        NativeError.requireNonNull(builder, "createInferBuilder");

        INetworkDefinition network = null;
        IBuilderConfig config = null;
        IParser parser = null;
        IHostMemory plan = null;
        try {
            int networkFlags = 1 ; //<< NetworkDefinitionCreationFlag.kEXPLICIT_BATCH.value;
            if (trtOptions.stronglyTyped()) {
                networkFlags |= 1 << NetworkDefinitionCreationFlag.kSTRONGLY_TYPED.value;
            }
            network = builder.createNetworkV2(networkFlags);
            NativeError.requireNonNull(network, "createNetworkV2");

            parser = createParser(network, nativeLogger);
            NativeError.requireNonNull(parser, "createParser");

            String modelPath = onnxModelPath.toAbsolutePath().toString();
            boolean parsed = parser.parseFromFile(modelPath, trtOptions.onnxVerbosity());
            if (!parsed) {
                NativeError.throwOnParserErrors(parser, "parseFromFile(" + modelPath + ")");
            }

            config = builder.createBuilderConfig();
            NativeError.requireNonNull(config, "createBuilderConfig");
            config.setMemoryPoolLimit(MemoryPoolType.kWORKSPACE, trtOptions.workspaceBytes());

            for (BuilderFlag flag : trtOptions.builderFlags()) {
                // setFlag(int) takes nvinfer1::BuilderFlag ordinal
                config.setFlag(flag.code());
            }

            if (trtOptions.maxThreads() > 0) {
                if (!builder.setMaxThreads(trtOptions.maxThreads())) {
                    logger.warning("IBuilder.setMaxThreads(" + trtOptions.maxThreads() + ") returned false");
                }
            }

            logger.info("Building serialized network from " + modelPath + " …");
            plan = builder.buildSerializedNetwork(network, config);
            NativeError.requireHostMemory(plan, "buildSerializedNetwork");

            long size = plan.size();
            if (size > Integer.MAX_VALUE) {
                throw new ExecutionException("serialized engine too large: " + size + " bytes");
            }
            byte[] bytes = new byte[(int) size];
            BytePointer data = new BytePointer(plan.data());
            data.capacity(size).limit(size).position(0);
            data.get(bytes);

            if (engineOutputPath != null) {
                try {
                    Path parent = engineOutputPath.getParent();
                    if (parent != null) {
                        Files.createDirectories(parent);
                    }
                    Files.write(engineOutputPath, bytes);
                    logger.info("Wrote engine plan to " + engineOutputPath.toAbsolutePath()
                            + " (" + bytes.length + " bytes)");
                } catch (IOException e) {
                    throw new ExecutionException("failed to write engine to " + engineOutputPath, e);
                }
            }

            return bytes;
        } finally {
            free(plan);
            free(parser);
            free(config);
            free(network);
            free(builder);
        }
    }

    /** Build plan bytes and deserialize into a ready {@link TRTEngine}. */
    public TRTEngine build() {
        byte[] plan = buildSerializedNetwork();
        return TRTEngine.fromSerialized(plan, trtOptions);
    }

    private static void free(Pointer p) {
        if (p != null && !p.isNull()) {
            p.deallocate();
        }
    }
}
