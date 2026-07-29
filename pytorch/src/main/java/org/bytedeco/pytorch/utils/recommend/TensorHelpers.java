/*
 * Ported from torch-rechub-scala:
 *   torchrec/Implicits.scala
 *   torchrec/TensorImplicits.scala
 *   torchrec/TorchRec.scala (tensor helpers)
 *   torchrec/package.scala (tensor helpers)
 *
 * Explicit static helpers replacing Scala implicits / package-level defs.
 */
package org.bytedeco.pytorch.utils.recommend;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Collection;

/**
 * Tensor construction and conversion helpers for the recommend stack.
 * Replaces Scala {@code Implicits}, {@code TensorImplicits}, and {@code TorchRec} helpers.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TensorHelpers {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private TensorHelpers() {}

    // ---- construction -------------------------------------------------------

    public static Tensor tensor(float[] data, long... sizes) {
        Tensor flat = torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (sizes.length == 1 && sizes[0] == data.length) {
            return flat;
        }
        return flat.reshape(sizes);
    }

    public static Tensor tensor(int[] data, long... sizes) {
        float[] floatData = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            floatData[i] = data[i];
        }
        return tensor(floatData, sizes);
    }

    /**
     * Long data via float→Long cast path (mirrors package.scala / TorchRec.longTensor).
     * Prefer {@link #longTensorDirect(long[])} when indices must stay Long without float round-trip.
     */
    public static Tensor tensor(long[] data, long... sizes) {
        int n = data.length;
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = (float) data[i];
        }
        Tensor flat = torch.tensor(f, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        Tensor t = flat.toType(ScalarType.Long);
        flat.close();
        if (sizes.length == 1 && sizes[0] == n) {
            return t;
        }
        Tensor r = t.reshape(sizes);
        t.close();
        return r;
    }

    public static Tensor zeros(long... sizes) {
        return torch.zeros(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    public static Tensor ones(long... sizes) {
        return torch.ones(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    public static Tensor randn(long... sizes) {
        return torch.randn(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    public static Tensor rand(long... sizes) {
        return torch.rand(sizes, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    /**
     * Create a Long tensor directly (Implicits.longTensor) to avoid dtype upcasting
     * during batching/stacking / embedding index_select.
     */
    public static Tensor longTensorDirect(long[] data) {
        Tensor t = torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        return t.is_contiguous() ? t : t.contiguous();
    }

    /** TorchRec / package.scala longTensor: float array → toType(Long). */
    public static Tensor longTensor(long[] data) {
        float[] f = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            f[i] = (float) data[i];
        }
        Tensor flat = torch.tensor(f, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        Tensor t = flat.toType(ScalarType.Long);
        flat.close();
        return t;
    }

    public static Tensor floatTensor(float[] data) {
        return torch.tensor(data, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
    }

    public static Tensor arange(int start, int end) {
        return torch.arange(new Scalar(start), new Scalar(end), new Scalar(1));
    }

    public static Tensor arange(int start, int end, int step) {
        return torch.arange(new Scalar(start), new Scalar(end), new Scalar(step));
    }

    // ---- device strings -----------------------------------------------------

    public static String cpu() {
        return "cpu";
    }

    public static String mps() {
        return "mps";
    }

    public static String cuda() {
        return cuda(0);
    }

    public static String cuda(int device) {
        return "cuda:" + device;
    }

    // ---- vector helpers -----------------------------------------------------

    /**
     * Build a TensorVector for torch.cat / stack.
     * <p>IMPORTANT: {@code new TensorVector(n)} default-constructs n <em>empty</em>
     * Tensors. Using {@code push_back} afterwards yields 2n elements (n undefined + n real)
     * and {@code torch.cat} fails with "tensor does not have a device". Always
     * {@code put(i, t)} into a pre-sized vector, or {@code push_back} into an empty one.
     */
    public static TensorVector toTensorVector(Tensor... tensors) {
        TensorVector vec = new TensorVector(tensors.length);
        for (int i = 0; i < tensors.length; i++) {
            vec.put(i, tensors[i]);
        }
        return vec;
    }

    public static TensorVector toTensorVector(Collection<Tensor> tensors) {
        TensorVector vec = new TensorVector(tensors.size());
        int i = 0;
        for (Tensor t : tensors) {
            vec.put(i++, t);
        }
        return vec;
    }

    public static TensorVector toParameterVector(Collection<Tensor> params) {
        return toTensorVector(params);
    }

    public static Tensor cat(Collection<Tensor> tensors, int dim) {
        return torch.cat(toTensorVector(tensors), dim);
    }

    public static Tensor cat(Tensor[] tensors, int dim) {
        return torch.cat(toTensorVector(tensors), dim);
    }

    public static Tensor stack(Collection<Tensor> tensors, int dim) {
        return torch.stack(toTensorVector(tensors), dim);
    }

    public static Tensor stack(Tensor[] tensors, int dim) {
        return torch.stack(toTensorVector(tensors), dim);
    }

    /**
     * Cat with ensureDevice fallback (TorchRec.cat). Some tensors from from_blob may lack a device.
     */
    public static Tensor catEnsured(Collection<Tensor> tensors, int dim) {
        TensorVector vec = new TensorVector(tensors.size());
        int i = 0;
        for (Tensor t : tensors) {
            vec.put(i++, ensureDevice(t, "cpu"));
        }
        return torch.cat(vec, dim);
    }

    public static Tensor stackEnsured(Collection<Tensor> tensors, int dim) {
        TensorVector vec = new TensorVector(tensors.size());
        int i = 0;
        for (Tensor t : tensors) {
            vec.put(i++, ensureDevice(t, "cpu"));
        }
        return torch.stack(vec, dim);
    }

    /** Try several strategies to ensure a tensor has a device set. */
    public static Tensor ensureDevice(Tensor t, String deviceStr) {
        try {
            return t.to(new Device(deviceStr), t.dtype());
        } catch (Throwable ignored) {
        }
        try {
            Tensor c = t.contiguous();
            return c.to(new Device(deviceStr), t.dtype());
        } catch (Throwable ignored) {
        }
        try {
            Tensor c = t.clone().contiguous();
            return c.to(new Device(deviceStr), t.dtype());
        } catch (Throwable ignored) {
        }
        return t;
    }

    public static Tensor toDevice(Tensor tensor, String device) {
        return tensor.to(new Device(device), tensor.dtype());
    }

    // ---- scalar extraction (Implicits.RichTensor) ---------------------------

    /**
     * Safely extract a scalar value from a tensor, handling GPU tensors properly.
     * Automatically moves the tensor to CPU if needed before calling .item()
     */
    public static double itemSafe(Tensor tensor) {
        Tensor cpuTensor = toCpuSafe(tensor);
        return cpuTensor.item().toDouble();
    }

    public static float toFloat(Tensor tensor) {
        Tensor cpuTensor = toCpuSafe(tensor);
        return cpuTensor.item().toFloat();
    }

    public static int toInt(Tensor tensor) {
        try {
            return tensor.item().toInt();
        } catch (Throwable t) {
            return 0;
        }
    }

    public static long toLong(Tensor tensor) {
        try {
            return tensor.item().toLong();
        } catch (Throwable t) {
            return 0L;
        }
    }

    private static Tensor toCpuSafe(Tensor tensor) {
        try {
            if (tensor.is_cuda()) {
                return tensor.cpu();
            }
            return tensor;
        } catch (Exception e) {
            try {
                return tensor.to(new Device("cpu"), tensor.dtype());
            } catch (Exception e2) {
                return tensor;
            }
        }
    }

    public static float[] toFloatArray(Tensor tensor) {
        Tensor cpuTensor = toCpuSafe(tensor);
        Tensor contig = cpuTensor.is_contiguous() ? cpuTensor : cpuTensor.contiguous();
        int size = (int) contig.numel();
        if (size == 0) {
            return new float[0];
        }
        try {
            ByteBuffer buf = contig.data_ptr().asByteBuffer();
            buf.limit(size * 4);
            FloatBuffer floatBuf = buf.asFloatBuffer();
            float[] result = new float[size];
            for (int i = 0; i < size; i++) {
                result[i] = floatBuf.get(i);
            }
            return result;
        } catch (Exception e) {
            Tensor flat = contig.reshape(size);
            float[] result = new float[size];
            for (int i = 0; i < size; i++) {
                result[i] = (float) itemSafe(flat.select(0, i));
            }
            return result;
        }
    }

    public static long[] toLongArray(Tensor tensor) {
        Tensor cpuTensor = toCpuSafe(tensor);
        Tensor contig = cpuTensor.is_contiguous() ? cpuTensor : cpuTensor.contiguous();
        int size = (int) contig.numel();
        if (size == 0) {
            return new long[0];
        }
        try {
            ByteBuffer buf = contig.data_ptr().asByteBuffer();
            buf.limit(size * 8);
            LongBuffer longBuf = buf.asLongBuffer();
            long[] result = new long[size];
            for (int i = 0; i < size; i++) {
                result[i] = longBuf.get(i);
            }
            return result;
        } catch (Exception e) {
            Tensor flat = contig.reshape(size);
            long[] result = new long[size];
            for (int i = 0; i < size; i++) {
                result[i] = (long) itemSafe(flat.select(0, i));
            }
            return result;
        }
    }

    // ---- elementwise wrappers (TorchRec) ------------------------------------

    public static Tensor relu(Tensor x) { return x.relu(); }
    public static Tensor sigmoid(Tensor x) { return x.sigmoid(); }
    public static Tensor tanh(Tensor x) { return x.tanh(); }
    public static Tensor softmax(Tensor x, int dim) { return x.softmax(dim); }
    public static Tensor logSoftmax(Tensor x, int dim) { return x.log_softmax(dim); }

    public static Tensor mean(Tensor x) { return x.mean(); }
    public static Tensor mean(Tensor x, int dim) { return x.mean(dim); }
    public static Tensor sum(Tensor x) { return x.sum(); }
    public static Tensor sum(Tensor x, int dim) { return x.sum(dim); }

    public static Tensor add(Tensor x, Tensor y) { return x.add(y); }
    public static Tensor sub(Tensor x, Tensor y) { return x.sub(y); }
    public static Tensor mul(Tensor x, Tensor y) { return x.mul(y); }
    public static Tensor div(Tensor x, Tensor y) { return x.div(y); }

    public static Tensor addScalar(Tensor x, float s) { return x.add(new Scalar(s)); }
    public static Tensor subScalar(Tensor x, float s) { return x.sub(new Scalar(s)); }
    public static Tensor mulScalar(Tensor x, float s) { return x.mul(new Scalar(s)); }
    public static Tensor divScalar(Tensor x, float s) { return x.div(new Scalar(s)); }

    public static Tensor pow(Tensor x, float exp) { return x.pow(new Scalar(exp)); }
    public static Tensor sqrt(Tensor x) { return x.sqrt(); }
    public static Tensor abs(Tensor x) { return x.abs(); }
    public static Tensor neg(Tensor x) { return x.neg(); }
    public static Tensor exp(Tensor x) { return x.exp(); }
    public static Tensor log(Tensor x) { return x.log(); }

    public static Tensor reshape(Tensor x, long... sizes) { return x.reshape(sizes); }
    public static Tensor view(Tensor x, long... sizes) { return x.view(sizes); }

    public static Tensor squeeze(Tensor x) { return x.squeeze(); }
    public static Tensor squeeze(Tensor x, int dim) {
        if (dim < 0) {
            return x.squeeze();
        }
        return x.squeeze(dim);
    }

    public static Tensor unsqueeze(Tensor x, int dim) { return x.unsqueeze(dim); }

    public static Tensor flatten(Tensor x, int startDim) {
        // Tensor.flatten requires (start_dim, end_dim); single-arg overload is flatten().
        return x.flatten(startDim, -1L);
    }

    public static Tensor flatten(Tensor x, int startDim, int endDim) {
        long end = endDim < 0 ? -1L : endDim;
        return x.flatten(startDim, end);
    }

    public static Tensor transpose(Tensor x) { return x.t(); }

    public static Tensor maskedFill(Tensor tensor, Tensor mask, float value) {
        return tensor.masked_fill(mask, new Scalar(value));
    }

    public static void printShape(Tensor tensor) {
        long[] shape = tensor.shape();
        StringBuilder sb = new StringBuilder("Shape: ");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        System.out.println(sb);
    }
}
