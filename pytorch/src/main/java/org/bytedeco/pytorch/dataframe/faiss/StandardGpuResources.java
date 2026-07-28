package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Semantic GPU resource handle — mirrors {@code faiss.StandardGpuResources}.
 *
 * <p>Does not wrap FAISS-GPU; records the preferred CUDA device index used by
 * {@link Faiss#index_cpu_to_gpu}. Actual compute goes through javacpp-pytorch
 * Tensor on that device when available.
 */
public final class StandardGpuResources implements java.io.Serializable {
    private static final long serialVersionUID = 1L;

    private int device;
    /** Soft cap hint (bytes); informational only. */
    private long tempMemory = 512L * 1024L * 1024L;

    public StandardGpuResources() {
        this(DeviceSelector.cudaDeviceIndex());
    }

    public StandardGpuResources(int device) {
        this.device = Math.max(0, device);
    }

    public int device() { return device; }

    public void setDevice(int device) {
        this.device = Math.max(0, device);
        DeviceSelector.setCudaDeviceIndex(this.device);
    }

    public void setTempMemory(long bytes) {
        this.tempMemory = Math.max(0, bytes);
    }

    public long tempMemory() { return tempMemory; }

    public boolean available() {
        return DeviceSelector.isCudaAvailable();
    }

    @Override
    public String toString() {
        return "StandardGpuResources{device=" + device
            + ", available=" + available()
            + ", tempMemory=" + tempMemory + "}";
    }
}
