/*
 * Ported from torch-rechub-scala:
 *   torchrec/TorchRec.scala
 *   torchrec/package.scala (facade helpers only; no dataframe / distributed)
 *
 * Public facade for the recommend stack. Tensor helpers are also available
 * via {@link TensorHelpers}; this class re-exports the common ones for
 * TorchRec-style call sites.
 */
package org.bytedeco.pytorch.utils.recommend;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;

import java.util.Collection;

/**
 * Facade entry point mirroring Scala {@code TorchRec} / package helpers.
 * Prefer {@link TensorHelpers} for new code; this class keeps TorchRec-style names.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Recommend {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private Recommend() {}

    // ---- type aliases (documentation only; Java uses full types) ------------
    // Tensor  -> org.bytedeco.pytorch.Tensor
    // Module  -> org.bytedeco.pytorch.nn.Module
    // ScalarType -> org.bytedeco.pytorch.global.torch.ScalarType

    public static String version() {
        return "0.1.0-recommend";
    }

    public static String backend() {
        return DeviceSupport.backend();
    }

    // ---- device strings -----------------------------------------------------

    public static String cpu() {
        return TensorHelpers.cpu();
    }

    public static String mps() {
        return TensorHelpers.mps();
    }

    public static String cuda() {
        return TensorHelpers.cuda();
    }

    public static String cuda(int device) {
        return TensorHelpers.cuda(device);
    }

    // ---- tensor construction ------------------------------------------------

    public static Tensor tensor(float[] data, long... sizes) {
        return TensorHelpers.tensor(data, sizes);
    }

    public static Tensor tensor(int[] data, long... sizes) {
        return TensorHelpers.tensor(data, sizes);
    }

    public static Tensor tensor(long[] data, long... sizes) {
        return TensorHelpers.tensor(data, sizes);
    }

    public static Tensor zeros(long... sizes) {
        return TensorHelpers.zeros(sizes);
    }

    public static Tensor ones(long... sizes) {
        return TensorHelpers.ones(sizes);
    }

    public static Tensor randn(long... sizes) {
        return TensorHelpers.randn(sizes);
    }

    public static Tensor rand(long... sizes) {
        return TensorHelpers.rand(sizes);
    }

    public static Tensor longTensor(long[] data) {
        return TensorHelpers.longTensor(data);
    }

    public static Tensor longTensorDirect(long[] data) {
        return TensorHelpers.longTensorDirect(data);
    }

    public static Tensor floatTensor(float[] data) {
        return TensorHelpers.floatTensor(data);
    }

    public static Tensor arange(int start, int end) {
        return TensorHelpers.arange(start, end);
    }

    public static Tensor arange(int start, int end, int step) {
        return TensorHelpers.arange(start, end, step);
    }

    // ---- vectors / cat / stack ----------------------------------------------

    public static TensorVector toTensorVector(Tensor... tensors) {
        return TensorHelpers.toTensorVector(tensors);
    }

    public static TensorVector toTensorVector(Collection<Tensor> tensors) {
        return TensorHelpers.toTensorVector(tensors);
    }

    public static Tensor cat(Collection<Tensor> tensors, int dim) {
        return TensorHelpers.cat(tensors, dim);
    }

    public static Tensor stack(Collection<Tensor> tensors, int dim) {
        return TensorHelpers.stack(tensors, dim);
    }

    // ---- elementwise / shape (TorchRec) -------------------------------------

    public static Tensor relu(Tensor x) { return TensorHelpers.relu(x); }
    public static Tensor sigmoid(Tensor x) { return TensorHelpers.sigmoid(x); }
    public static Tensor tanh(Tensor x) { return TensorHelpers.tanh(x); }
    public static Tensor softmax(Tensor x, int dim) { return TensorHelpers.softmax(x, dim); }
    public static Tensor logSoftmax(Tensor x, int dim) { return TensorHelpers.logSoftmax(x, dim); }

    public static Tensor mean(Tensor x) { return TensorHelpers.mean(x); }
    public static Tensor mean(Tensor x, int dim) { return TensorHelpers.mean(x, dim); }
    public static Tensor sum(Tensor x) { return TensorHelpers.sum(x); }
    public static Tensor sum(Tensor x, int dim) { return TensorHelpers.sum(x, dim); }

    public static Tensor add(Tensor x, Tensor y) { return TensorHelpers.add(x, y); }
    public static Tensor sub(Tensor x, Tensor y) { return TensorHelpers.sub(x, y); }
    public static Tensor mul(Tensor x, Tensor y) { return TensorHelpers.mul(x, y); }
    public static Tensor div(Tensor x, Tensor y) { return TensorHelpers.div(x, y); }

    public static Tensor addScalar(Tensor x, float s) { return TensorHelpers.addScalar(x, s); }
    public static Tensor subScalar(Tensor x, float s) { return TensorHelpers.subScalar(x, s); }
    public static Tensor mulScalar(Tensor x, float s) { return TensorHelpers.mulScalar(x, s); }
    public static Tensor divScalar(Tensor x, float s) { return TensorHelpers.divScalar(x, s); }

    public static Tensor pow(Tensor x, float exp) { return TensorHelpers.pow(x, exp); }
    public static Tensor sqrt(Tensor x) { return TensorHelpers.sqrt(x); }
    public static Tensor abs(Tensor x) { return TensorHelpers.abs(x); }
    public static Tensor neg(Tensor x) { return TensorHelpers.neg(x); }
    public static Tensor exp(Tensor x) { return TensorHelpers.exp(x); }
    public static Tensor log(Tensor x) { return TensorHelpers.log(x); }

    public static Tensor reshape(Tensor x, long... sizes) { return TensorHelpers.reshape(x, sizes); }
    public static Tensor view(Tensor x, long... sizes) { return TensorHelpers.view(x, sizes); }
    public static Tensor squeeze(Tensor x) { return TensorHelpers.squeeze(x); }
    public static Tensor squeeze(Tensor x, int dim) { return TensorHelpers.squeeze(x, dim); }
    public static Tensor unsqueeze(Tensor x, int dim) { return TensorHelpers.unsqueeze(x, dim); }
    public static Tensor flatten(Tensor x, int startDim) { return TensorHelpers.flatten(x, startDim); }
    public static Tensor transpose(Tensor x) { return TensorHelpers.transpose(x); }

    public static float toFloat(Tensor x) { return TensorHelpers.toFloat(x); }
    public static int toInt(Tensor x) { return TensorHelpers.toInt(x); }
    public static long toLong(Tensor x) { return TensorHelpers.toLong(x); }
    public static float[] toFloatArray(Tensor x) { return TensorHelpers.toFloatArray(x); }
    public static long[] toLongArray(Tensor x) { return TensorHelpers.toLongArray(x); }

    /** Ensure native torch is loaded (call from samples before first use). */
    public static void loadNative() {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    // silence unused-import warnings for documented aliases
    @SuppressWarnings("unused")
    private static final Class<?> MODULE_TYPE = Module.class;
    @SuppressWarnings("unused")
    private static final Class<?> SCALAR_TYPE = ScalarType.class;
}
