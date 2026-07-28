package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Device selection for FAISS distance backends — CPU or CUDA via javacpp-pytorch.
 *
 * <p>Auto-detects CUDA once; force with {@link #setPreferred(Device)}.
 * Detection never throws; failures resolve to CPU.
 */
public final class DeviceSelector {
    public enum Device {
        CPU,
        CUDA
    }

    private static volatile Boolean cudaAvailable;
    private static volatile Device preferred; // null = auto
    private static volatile int cudaDeviceIndex = 0;
    private static volatile String lastProbeDetail = "unprobed";

    private DeviceSelector() {}

    public static void setPreferred(Device device) {
        preferred = device;
    }

    public static Device preferred() {
        return preferred;
    }

    public static void setCudaDeviceIndex(int index) {
        cudaDeviceIndex = Math.max(0, index);
    }

    public static int cudaDeviceIndex() {
        return cudaDeviceIndex;
    }

    public static Device resolve() {
        if (preferred != null) {
            if (preferred == Device.CUDA && !isCudaAvailable()) return Device.CPU;
            return preferred;
        }
        return isCudaAvailable() ? Device.CUDA : Device.CPU;
    }

    public static boolean isCudaAvailable() {
        Boolean cached = cudaAvailable;
        if (cached != null) return cached;
        synchronized (DeviceSelector.class) {
            if (cudaAvailable != null) return cudaAvailable;
            cudaAvailable = probeCuda();
            return cudaAvailable;
        }
    }

    public static void resetCache() {
        synchronized (DeviceSelector.class) {
            cudaAvailable = null;
            lastProbeDetail = "unprobed";
        }
    }

    public static String lastProbeDetail() {
        return lastProbeDetail;
    }

    private static boolean probeCuda() {
        try {
            // 0) Fast negative: no torch_cuda classes on classpath
            boolean hasCudaClasses = false;
            for (String cn : new String[]{
                "org.bytedeco.pytorch.cuda.global.torch_cuda",
                "org.bytedeco.pytorch.presets.torch_cuda"
            }) {
                try {
                    Class.forName(cn, false, DeviceSelector.class.getClassLoader());
                    hasCudaClasses = true;
                    break;
                } catch (Throwable ignored) {}
            }

            // 1) Explicit API if generated
            try {
                Class<?> torch = Class.forName("org.bytedeco.pytorch.global.torch");
                for (String name : new String[]{"cuda_is_available", "hasCUDA", "is_cuda_available"}) {
                    try {
                        java.lang.reflect.Method m = torch.getMethod(name);
                        Object r = m.invoke(null);
                        if (r instanceof Boolean b) {
                            lastProbeDetail = name + "=" + b;
                            return b;
                        }
                    } catch (NoSuchMethodException ignored) {
                    }
                }
            } catch (ClassNotFoundException e) {
                lastProbeDetail = "torch class missing";
                return false;
            }

            // 2) Try moving a 1-element tensor to cuda (may fail if native libs not loaded)
            try {
                org.bytedeco.pytorch.Tensor t =
                    org.bytedeco.pytorch.global.torch.tensor(new float[]{0f});
                try {
                    java.lang.reflect.Method cudaM = t.getClass().getMethod("cuda");
                    Object g = cudaM.invoke(t);
                    if (g instanceof org.bytedeco.pytorch.Tensor gt) {
                        boolean ok = false;
                        try {
                            ok = gt.is_cuda();
                        } catch (Throwable ignored) {
                            ok = true; // cuda() succeeded
                        }
                        try { gt.close(); } catch (Throwable ignored) {}
                        try { t.close(); } catch (Throwable ignored) {}
                        lastProbeDetail = "tensor.cuda() ok=" + ok;
                        return ok;
                    }
                } catch (NoSuchMethodException ns) {
                    try {
                        Class<?> devCls = Class.forName("org.bytedeco.pytorch.Device");
                        Object dev = devCls.getConstructor(String.class)
                            .newInstance("cuda:" + cudaDeviceIndex);
                        java.lang.reflect.Method toM = t.getClass().getMethod("to", devCls);
                        Object g = toM.invoke(t, dev);
                        boolean ok = g instanceof org.bytedeco.pytorch.Tensor gt && safeIsCuda(gt);
                        if (g instanceof AutoCloseable ac) try { ac.close(); } catch (Exception ignored) {}
                        try { t.close(); } catch (Throwable ignored) {}
                        lastProbeDetail = "tensor.to(cuda) ok=" + ok;
                        return ok;
                    } catch (Throwable e) {
                        lastProbeDetail = "to(cuda) failed: " + shortMsg(e)
                            + (hasCudaClasses ? " (cuda classes present)" : " (no cuda classes)");
                        try { t.close(); } catch (Throwable ignored) {}
                        return false;
                    }
                } catch (Throwable e) {
                    // Method exists but native call failed (no GPU driver / no lib)
                    lastProbeDetail = "tensor.cuda() failed: " + shortMsg(e);
                    try { t.close(); } catch (Throwable ignored) {}
                    return false;
                }
                try { t.close(); } catch (Throwable ignored) {}
            } catch (Throwable e) {
                // torch.tensor itself failed (e.g. openblas preload) — still not CUDA
                lastProbeDetail = "torch init failed (CPU ok via pure-Java kernels); CUDA=false: "
                    + shortMsg(e);
                return false;
            }
            lastProbeDetail = hasCudaClasses
                ? "cuda classes present but no working device"
                : "no cuda path";
            return false;
        } catch (Throwable e) {
            lastProbeDetail = "outer: " + shortMsg(e);
            return false;
        }
    }

    private static String shortMsg(Throwable e) {
        Throwable c = e;
        while (c.getCause() != null && c.getCause() != c) c = c.getCause();
        String m = c.getMessage();
        if (m == null || m.isEmpty()) m = c.getClass().getSimpleName();
        // keep one line
        m = m.replace('\n', ' ');
        if (m.length() > 160) m = m.substring(0, 160) + "...";
        return m;
    }

    private static boolean safeIsCuda(org.bytedeco.pytorch.Tensor t) {
        try {
            return t.is_cuda();
        } catch (Throwable e) {
            return false;
        }
    }

    public static String describe() {
        boolean cuda = isCudaAvailable();
        Device r = resolve();
        return "cuda_available=" + cuda
            + " preferred=" + preferred
            + " resolved=" + r
            + (r == Device.CUDA ? (" device=" + cudaDeviceIndex) : "")
            + " probe=" + lastProbeDetail;
    }
}
