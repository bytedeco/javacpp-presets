package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.tensorrt.nvinfer.Dims64;
import org.bytedeco.tensorrt.nvinfer.ICudaEngine;
import org.bytedeco.tensorrt.nvinfer.IExecutionContext;
import org.bytedeco.tensorrt.nvinfer.IRuntime;
import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;
import org.bytedeco.pytorch.serving.tensorrt.enums.TRTDataType;
import org.bytedeco.pytorch.serving.tensorrt.enums.TRTTensorIOMode;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.ExecutionException;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.InternalException;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtNotFoundException;
import org.bytedeco.pytorch.serving.tensorrt.internal.CudaBuffers;
import org.bytedeco.pytorch.serving.tensorrt.internal.NativeError;
import org.bytedeco.pytorch.serving.tensorrt.internal.NativeLogger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

import static org.bytedeco.tensorrt.global.nvinfer.createInferRuntime;

/**
 * High-level TensorRT engine (deserialized plan + execution helpers).
 *
 * <p>Corresponds to Python:
 * <pre>{@code
 * runtime = trt.Runtime(logger)
 * engine = runtime.deserialize_cuda_engine(plan_bytes)
 * context = engine.create_execution_context()
 * context.set_input_shape(...)
 * context.set_tensor_address(name, ptr)
 * context.execute_async_v3(stream)
 * }</pre>
 *
 * <p>Native symbols: {@code nvinfer1::IRuntime}, {@code ICudaEngine},
 * {@code IExecutionContext} (bytedeco {@code org.bytedeco.tensorrt.nvinfer.*}).
 *
 * <p>MVP inference path: host {@link TrtTensor} inputs → device H2D →
 * {@code enqueueV3} → D2H outputs. Dynamic shapes must be provided via the
 * input tensor shape (and optionally {@link #infer(Map, Map)} shape overrides).
 */
public final class TRTEngine implements AutoCloseable {
    private final TrtOption trtOptions;
    private final TRTLogger logger;
    private final IRuntime runtime;
    private final ICudaEngine cudaEngine;
    private final List<TRTTensorInfo> ioTensors;
    private final ReentrantLock inferLock = new ReentrantLock();
    private volatile boolean closed;

    private TRTEngine(TrtOption trtOptions, TRTLogger logger, IRuntime runtime, ICudaEngine cudaEngine,
                      List<TRTTensorInfo> ioTensors) {
        this.trtOptions = trtOptions;
        this.logger = logger;
        this.runtime = runtime;
        this.cudaEngine = cudaEngine;
        this.ioTensors = List.copyOf(ioTensors);
    }

    /**
     * Deserialize a serialized engine plan (result of
     * {@code Builder.build_serialized_network} / {@link TRTEngineBuilder#buildSerializedNetwork()}).
     */
    public static TRTEngine fromSerialized(byte[] plan, TrtOption trtOptions) {
        Objects.requireNonNull(plan, "plan");
        if (plan.length == 0) {
            throw new TrtInvalidArgumentException("serialized engine plan is empty");
        }
        TrtOption opts = trtOptions == null ? new TrtOption() : new TrtOption(trtOptions);
        TRTLogger logger = opts.logger() != null ? opts.logger() : TRTLogger.getDefaultLogger();
        NativeLogger nativeLogger = logger.nativeLogger();

        NativeError.checkCuda(
                org.bytedeco.cuda.global.cudart.cudaSetDevice(opts.deviceIndex()),
                "cudaSetDevice(" + opts.deviceIndex() + ")");

        IRuntime runtime = createInferRuntime(nativeLogger);
        NativeError.requireNonNull(runtime, "createInferRuntime");

        if (opts.maxThreads() > 0) {
            if (!runtime.setMaxThreads(opts.maxThreads())) {
                logger.warning("IRuntime.setMaxThreads(" + opts.maxThreads() + ") returned false");
            }
        }
        if (opts.dlaCore() >= 0) {
            runtime.setDLACore(opts.dlaCore());
        }

        BytePointer blob = new BytePointer(plan);
        try {
            blob.capacity(plan.length).limit(plan.length).position(0);
            ICudaEngine engine = runtime.deserializeCudaEngine(blob, plan.length);
            NativeError.requireNonNull(engine, "deserializeCudaEngine");
            List<TRTTensorInfo> ios = readIoTensors(engine);
            logger.info("Deserialized engine '" + nullToEmpty(engine.getName())
                    + "' with " + ios.size() + " I/O tensors");
            return new TRTEngine(opts, logger, runtime, engine, ios);
        } catch (RuntimeException e) {
            freeQuietly(runtime);
            throw e;
        } finally {
            // blob is a view of the Java byte[]; do not free device resources here.
            blob.deallocate();
        }
    }

    public static TRTEngine fromSerialized(byte[] plan) {
        return fromSerialized(plan, new TrtOption());
    }

    public static TRTEngine load(Path enginePath, TrtOption trtOptions) {
        Objects.requireNonNull(enginePath, "enginePath");
        if (!Files.isRegularFile(enginePath)) {
            throw new TrtNotFoundException("engine file not found: " + enginePath);
        }
        try {
            byte[] plan = Files.readAllBytes(enginePath);
            return fromSerialized(plan, trtOptions);
        } catch (IOException e) {
            throw new ExecutionException("failed to read engine file: " + enginePath, e);
        }
    }

    public static TRTEngine load(String enginePath, TrtOption trtOptions) {
        return load(Path.of(enginePath), trtOptions);
    }

    public static TRTEngine load(String enginePath) {
        return load(enginePath, new TrtOption());
    }

    public TrtOption options() {
        return trtOptions;
    }

    public TRTLogger logger() {
        return logger;
    }

    /** Engine name from {@code ICudaEngine.getName()}, may be empty. */
    public String name() {
        ensureOpen();
        return nullToEmpty(cudaEngine.getName());
    }

    public int nbIOTensors() {
        ensureOpen();
        return ioTensors.size();
    }

    public List<TRTTensorInfo> ioTensors() {
        return ioTensors;
    }

    public List<TRTTensorInfo> inputs() {
        List<TRTTensorInfo> out = new ArrayList<>();
        for (TRTTensorInfo t : ioTensors) {
            if (t.input()) {
                out.add(t);
            }
        }
        return Collections.unmodifiableList(out);
    }

    public List<TRTTensorInfo> outputs() {
        List<TRTTensorInfo> out = new ArrayList<>();
        for (TRTTensorInfo t : ioTensors) {
            if (t.output()) {
                out.add(t);
            }
        }
        return Collections.unmodifiableList(out);
    }

    public TRTTensorInfo tensor(String name) {
        Objects.requireNonNull(name, "name");
        for (TRTTensorInfo t : ioTensors) {
            if (name.equals(t.name())) {
                return t;
            }
        }
        throw new TrtNotFoundException("I/O tensor not found: " + name);
    }

    /**
     * Serialize the live engine ({@code ICudaEngine.serialize}).
     */
    public byte[] serialize() {
        ensureOpen();
        var host = cudaEngine.serialize();
        NativeError.requireHostMemory(host, "ICudaEngine.serialize");
        try {
            long size = host.size();
            if (size > Integer.MAX_VALUE) {
                throw new ExecutionException("serialized engine too large: " + size);
            }
            byte[] bytes = new byte[(int) size];
            BytePointer data = new BytePointer(host.data());
            data.capacity(size).limit(size).position(0);
            data.get(bytes);
            return bytes;
        } finally {
            freeQuietly(host);
        }
    }

    public void save(Path path) {
        Objects.requireNonNull(path, "path");
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(path, serialize());
        } catch (IOException e) {
            throw new ExecutionException("failed to save engine to " + path, e);
        }
    }

    public void save(String path) {
        save(Path.of(path));
    }

    /**
     * Run inference with named host tensors.
     *
     * @param inputTensors map of input name → host {@link TrtTensor}
     * @return output name → host {@link TrtTensor}
     */
    public TRTInferenceResult infer(Map<String, TrtTensor> inputTensors) {
        return infer(inputTensors, null);
    }

    /**
     * Run inference with optional explicit input shapes (dynamic-shape engines).
     *
     * @param inputTensors map of input name → host {@link TrtTensor}
     * @param inputShapes  optional name → shape overrides; if null, uses each tensor's shape
     */
    public TRTInferenceResult infer(Map<String, TrtTensor> inputTensors, Map<String, long[]> inputShapes) {
        Objects.requireNonNull(inputTensors, "inputTensors");
        ensureOpen();

        inferLock.lock();
        try {
            return inferUnlocked(inputTensors, inputShapes);
        } finally {
            inferLock.unlock();
        }
    }

    /**
     * Convenience: single-input / positional binding by engine input order.
     * Keys are taken from {@link #inputs()} in order.
     */
    public TRTInferenceResult infer(TrtTensor... inputsInOrder) {
        Objects.requireNonNull(inputsInOrder, "inputs");
        List<TRTTensorInfo> ins = inputs();
        if (inputsInOrder.length != ins.size()) {
            throw new TrtInvalidArgumentException(
                    "expected " + ins.size() + " inputs, got " + inputsInOrder.length);
        }
        Map<String, TrtTensor> map = new LinkedHashMap<>();
        for (int i = 0; i < ins.size(); i++) {
            map.put(ins.get(i).name(), inputsInOrder[i]);
        }
        return infer(map);
    }

    private TRTInferenceResult inferUnlocked(Map<String, TrtTensor> inputTensors,
                                             Map<String, long[]> inputShapes) {
        NativeError.checkCuda(
                org.bytedeco.cuda.global.cudart.cudaSetDevice(trtOptions.deviceIndex()),
                "cudaSetDevice");

        IExecutionContext context = cudaEngine.createExecutionContext();
        NativeError.requireNonNull(context, "createExecutionContext");

        CUstream_st stream = null;
        List<Pointer> devicePtrs = new ArrayList<>();
        try {
            stream = CudaBuffers.createStreamPtr();

            // ---- inputs ----
            for (TRTTensorInfo info : inputs()) {
                TrtTensor host = inputTensors.get(info.name());
                if (host == null) {
                    throw new TrtInvalidArgumentException("missing input tensor: " + info.name());
                }
                long[] shape = inputShapes != null && inputShapes.containsKey(info.name())
                        ? inputShapes.get(info.name())
                        : host.shape();
                setInputShape(context, info.name(), shape);

                long bytes = host.byteSize();
                // For dynamic shapes, recompute expected size from resolved shape + dtype.
                if (info.dataType().byteSize() > 0) {
                    long expected = TrtTensor.volumeBytes(info.dataType(), shape);
                    if (bytes < expected) {
                        throw new TrtInvalidArgumentException(
                                "input " + info.name() + " has " + bytes
                                        + " bytes, need at least " + expected);
                    }
                    bytes = expected;
                }

                Pointer dev = CudaBuffers.mallocDevice(bytes);
                devicePtrs.add(dev);
                CudaBuffers.copyHostToDevice(dev, host.hostData(), bytes);
                NativeError.requireTrue(
                        context.setTensorAddress(info.name(), dev),
                        "setTensorAddress(" + info.name() + ")");
            }

            // Resolve output shapes after inputs are specified.
            if (!context.allInputDimensionsSpecified()) {
                throw new ExecutionException(
                        "not all input dimensions specified before enqueue",
                        ErrorCode.INVALID_CONFIG);
            }

            // ---- outputs ----
            Map<String, long[]> outputShapes = new LinkedHashMap<>();
            Map<String, TRTDataType> outputTypes = new LinkedHashMap<>();
            for (TRTTensorInfo info : outputs()) {
                long[] shape = dimsToArray(context.getTensorShape(info.name()));
                outputShapes.put(info.name(), shape);
                outputTypes.put(info.name(), info.dataType());
                long bytes = TrtTensor.volumeBytes(info.dataType(), shape);
                Pointer dev = CudaBuffers.mallocDevice(bytes);
                devicePtrs.add(dev);
                NativeError.requireTrue(
                        context.setTensorAddress(info.name(), dev),
                        "setTensorAddress(" + info.name() + ")");
            }

            NativeError.requireTrue(context.enqueueV3(stream), "enqueueV3");
            CudaBuffers.synchronize(stream);

            Map<String, TrtTensor> result = new LinkedHashMap<>();
            int devIndex = inputs().size(); // devicePtrs layout: inputs then outputs
            for (TRTTensorInfo info : outputs()) {
                long[] shape = outputShapes.get(info.name());
                TRTDataType dt = outputTypes.get(info.name());
                long bytes = TrtTensor.volumeBytes(dt, shape);
                Pointer dev = devicePtrs.get(devIndex++);
                byte[] hostBytes = CudaBuffers.copyDeviceToHostBytes(dev, bytes);
                result.put(info.name(), TrtTensor.of(dt, shape, hostBytes));
            }
            return new TRTInferenceResult(result);
        } finally {
            for (Pointer p : devicePtrs) {
                try {
                    CudaBuffers.freeDevice(p);
                } catch (RuntimeException e) {
                    logger.warning("freeDevice failed: " + e.getMessage());
                }
            }
            if (stream != null) {
                try {
                    CudaBuffers.destroyStream(stream);
                } catch (RuntimeException e) {
                    logger.warning("destroyStream failed: " + e.getMessage());
                }
            }
            freeQuietly(context);
        }
    }

    private static void setInputShape(IExecutionContext context, String name, long[] shape) {
        Dims64 dims = new Dims64();
        try {
            if (shape.length > Dims64.MAX_DIMS) {
                throw new TrtInvalidArgumentException(
                        "rank " + shape.length + " exceeds Dims64.MAX_DIMS=" + Dims64.MAX_DIMS);
            }
            dims.nbDims(shape.length);
            for (int i = 0; i < shape.length; i++) {
                dims.d(i, shape[i]);
            }
            NativeError.requireTrue(
                    context.setInputShape(name, dims),
                    "setInputShape(" + name + ")");
        } finally {
            dims.deallocate();
        }
    }

    private static List<TRTTensorInfo> readIoTensors(ICudaEngine engine) {
        int n = engine.getNbIOTensors();
        List<TRTTensorInfo> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            String name = engine.getIOTensorName(i);
            if (name == null) {
                throw new InternalException("getIOTensorName(" + i + ") returned null");
            }
            int modeCode = engine.getTensorIOMode(name).value;
            TRTTensorIOMode mode = TRTTensorIOMode.fromCode(modeCode);
            int typeCode = engine.getTensorDataType(name).value;
            TRTDataType dt = TRTDataType.fromCode(typeCode);
            long[] shape = dimsToArray(engine.getTensorShape(name));
            list.add(new TRTTensorInfo(name, mode, dt, shape));
        }
        return list;
    }

    private static long[] dimsToArray(Dims64 dims) {
        if (dims == null || dims.isNull()) {
            return new long[0];
        }
        int nb = dims.nbDims();
        if (nb < 0) {
            // invalid / unknown rank — surface as empty; caller may set dynamic shape
            return new long[0];
        }
        long[] out = new long[nb];
        for (int i = 0; i < nb; i++) {
            out[i] = dims.d(i);
        }
        return out;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Engine is closed");
        }
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }

    static void freeQuietly(Pointer p) {
        if (p != null && !p.isNull()) {
            p.deallocate();
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        freeQuietly(cudaEngine);
        freeQuietly(runtime);
    }
}
