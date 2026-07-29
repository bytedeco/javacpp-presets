/*
 * Ported from torch-rechub-scala: torchrec/utils/DeviceSupport.scala
 *
 * Central device selector for the recommend stack.
 */
package org.bytedeco.pytorch.utils.recommend;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;

/**
 * Central device selector for the whole recommend project.
 *
 * <p>The active backend ("cuda" | "mps" | "cpu") is resolved exactly once on first use and then
 * cached. It can be chosen, in priority order, via:
 * <ol>
 *   <li>programmatic override: {@code DeviceSupport.setDevice(DeviceType.CUDA)} (call BEFORE building the model)</li>
 *   <li>system property: {@code -Dnanovllm.device=cuda}</li>
 *   <li>environment variable: {@code NANOVLLM_DEVICE=cuda}</li>
 *   <li>AUTO (default): MPS on macOS, else CUDA if available, else CPU.</li>
 * </ol>
 *
 * <p>Every tensor-allocation site goes through {@link #deviceOf()} / {@code *Opts()} / {@link #backend()},
 * so flipping this one flag switches the entire project between CUDA and MPS (and CPU).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DeviceSupport {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Device selection flag. AUTO performs platform-aware auto-detection. */
    public enum DeviceType {
        AUTO, CUDA, MPS, CPU, XPU, NPU
    }

    private static final String osName = System.getProperty("os.name", "").toLowerCase();

    private static volatile DeviceType requested = readInitialRequest();
    private static volatile String resolvedBackend = null;

    private static Boolean cudaAvailableCache = null;
    private static Boolean mpsAvailableCache = null;
    private static String backendCache = null;
    private static Boolean acceleratorAvailableCache = null;

    private DeviceSupport() {}

    private static DeviceType readInitialRequest() {
        String prop = System.getProperty("nanovllm.device");
        if (prop == null || prop.trim().isEmpty()) {
            prop = System.getenv("NANOVLLM_DEVICE");
        }
        if (prop == null || prop.trim().isEmpty()) {
            return DeviceType.AUTO;
        }
        return parseDeviceType(prop.trim());
    }

    /** Parse a free-form device string into a {@link DeviceType}. Unknown values map to AUTO. */
    public static DeviceType parseDeviceType(String name) {
        String n = name.trim().toLowerCase();
        switch (n) {
            case "cuda":
            case "gpu":
            case "nvidia":
                return DeviceType.CUDA;
            case "mps":
            case "metal":
                return DeviceType.MPS;
            case "cpu":
                return DeviceType.CPU;
            case "auto":
            case "":
                return DeviceType.AUTO;
            case "xpu":
                return DeviceType.XPU;
            case "npu":
                return DeviceType.NPU;
            default:
                System.out.println("[DeviceSupport] Unknown device '" + name + "', falling back to AUTO");
                return DeviceType.AUTO;
        }
    }

    /** Programmatically choose the backend. Must be called before the device is first used. */
    public static synchronized void setDevice(DeviceType dt) {
        if (resolvedBackend != null) {
            String target = resolveBackend(dt);
            if (!resolvedBackend.equals(target)) {
                System.out.println("[DeviceSupport] WARNING: device already initialized as '" + resolvedBackend
                        + "'; request to switch to '" + target + "' is ignored. Choose the device before creating the model "
                        + "(use -Dnanovllm.device / NANOVLLM_DEVICE, or call setDevice earlier).");
            }
        } else {
            requested = dt;
        }
    }

    /** Convenience overload accepting a string flag ("cuda" | "mps" | "cpu" | "auto"). */
    public static void setDevice(String name) {
        setDevice(parseDeviceType(name));
    }

    /** The currently requested (possibly unresolved) selection. */
    public static DeviceType requestedDevice() {
        return requested;
    }

    // ---- availability -------------------------------------------------------

    private static boolean invokeTorchBoolean(String methodName) {
        try {
            Method method = Class.forName("org.bytedeco.pytorch.global.torch").getMethod(methodName);
            Object result = method.invoke(null);
            return result instanceof Boolean && (Boolean) result;
        } catch (Throwable t) {
            return false;
        }
    }

    public static synchronized boolean cudaAvailable() {
        if (cudaAvailableCache == null) {
            try {
                cudaAvailableCache = torch.cuda_is_available();
            } catch (Throwable t) {
                cudaAvailableCache = false;
            }
        }
        return cudaAvailableCache;
    }

    public static synchronized boolean mpsAvailable() {
        if (mpsAvailableCache == null) {
            boolean viaMethod = invokeTorchBoolean("mps_is_available") || invokeTorchBoolean("hasMPS");
            if (viaMethod) {
                mpsAvailableCache = true;
            } else {
                try {
                    Tensor probe = torch.zeros(new long[]{1L}, floatOptsFor("mps"));
                    probe.close();
                    mpsAvailableCache = true;
                } catch (Throwable t) {
                    mpsAvailableCache = false;
                }
            }
        }
        return mpsAvailableCache;
    }

    // ---- resolution ---------------------------------------------------------

    private static String resolveBackend(DeviceType dt) {
        switch (dt) {
            case CPU:
                return "cpu";
            case CUDA:
                if (cudaAvailable()) {
                    return "cuda";
                }
                System.out.println("[DeviceSupport] CUDA requested but not available; falling back to AUTO");
                return autoBackend();
            case MPS:
                if (mpsAvailable()) {
                    return "mps";
                }
                System.out.println("[DeviceSupport] MPS requested but not available; falling back to AUTO");
                return autoBackend();
            case AUTO:
            default:
                return autoBackend();
        }
    }

    private static String autoBackend() {
        if (osName.contains("mac") && mpsAvailable()) {
            return "mps";
        } else if (cudaAvailable()) {
            return "cuda";
        } else {
            return "cpu";
        }
    }

    /** Alias for {@link #backend()}: the device string to use as the project-wide default. */
    public static String defaultDevice() {
        return backend();
    }

    /** The resolved backend string: "cuda" | "mps" | "cpu". Resolved once, then cached. */
    public static synchronized String backend() {
        if (backendCache != null) {
            return backendCache;
        }
        if (resolvedBackend == null) {
            String r = resolveBackend(requested);
            resolvedBackend = r;
            System.out.println("[DeviceSupport] active device backend: " + r + " (requested=" + requested
                    + ", cuda=" + cudaAvailable() + ", mps=" + mpsAvailable() + ", os=" + osName + ")");
        }
        backendCache = resolvedBackend;
        return backendCache;
    }

    public static synchronized boolean acceleratorAvailable() {
        if (acceleratorAvailableCache == null) {
            acceleratorAvailableCache = !"cpu".equals(backend());
        }
        return acceleratorAvailableCache;
    }

    // ---- tensor option helpers ---------------------------------------------

    public static Device deviceOf() {
        return deviceOf(backend());
    }

    public static Device deviceOf(String deviceType) {
        return new Device(deviceType);
    }

    public static TensorOptions opts(ScalarType dtype) {
        return opts(dtype, backend());
    }

    public static TensorOptions opts(ScalarType dtype, String deviceType) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(dtype))
                .device(new DeviceOptional(deviceOf(deviceType)));
    }

    public static TensorOptions floatOpts() {
        return floatOpts(backend());
    }

    public static TensorOptions floatOpts(String deviceType) {
        return opts(ScalarType.Float, deviceType);
    }

    public static TensorOptions longOpts() {
        return longOpts(backend());
    }

    public static TensorOptions longOpts(String deviceType) {
        return opts(ScalarType.Long, deviceType);
    }

    // internal variant that does not trigger backend resolution (used by the mps probe)
    private static TensorOptions floatOptsFor(String deviceType) {
        return new TensorOptions()
                .dtype(new ScalarTypeOptional(ScalarType.Float))
                .device(new DeviceOptional(new Device(deviceType)));
    }

    // ---- memory estimation --------------------------------------------------

    /** Free VRAM in bytes for the active CUDA device, if queryable. */
    private static Long cudaFreeMemoryBytes() {
        try {
            // getMemoryInfo returns (free, total)
            var info = torch.getMemoryInfo((byte) 0);
            long free = info.first();
            if (free > 0L) {
                return free;
            }
            return null;
        } catch (Throwable t) {
            return null;
        }
    }

    /** Total VRAM in bytes for the active CUDA device, if queryable. */
    private static Long cudaTotalMemoryBytes() {
        try {
            var info = torch.getMemoryInfo((byte) 0);
            long total = info.second();
            if (total > 0L) {
                return total;
            }
            return null;
        } catch (Throwable t) {
            return null;
        }
    }

    private static long physicalMemoryBytes() {
        try {
            Object bean = ManagementFactory.getOperatingSystemMXBean();
            Method method = bean.getClass().getMethod("getTotalMemorySize");
            Object result = method.invoke(bean);
            if (result instanceof Long) {
                return (Long) result;
            }
            if (result instanceof Number) {
                return ((Number) result).longValue();
            }
        } catch (Throwable ignored) {
        }
        return 16L << 30;
    }

    /**
     * Estimate the memory budget available for the KV cache, scaled by {@code utilization}.
     *
     * <p>On CUDA: uses {@code total_GPU_memory * utilization}, which correctly reserves
     * space for model weights, activations, and KV cache proportionally.
     * Falls back to {@code free_GPU_memory * utilization} if total is unavailable.
     * On MPS/CPU: uses a fraction of system RAM.
     */
    public static long estimateAvailableMemoryBytes(float utilization) {
        long budget;
        if ("cuda".equals(backend())) {
            Long total = cudaTotalMemoryBytes();
            if (total != null) {
                budget = (long) (total * utilization);
            } else {
                Long free = cudaFreeMemoryBytes();
                if (free != null) {
                    budget = (long) (free * utilization);
                } else {
                    budget = (long) (physicalMemoryBytes() * Math.min(utilization, 0.5f));
                }
            }
        } else {
            budget = (long) (physicalMemoryBytes() * utilization);
        }
        return Math.max(512L << 20, budget);
    }

    /**
     * Returns the actual free GPU memory in bytes (after model weights are loaded).
     * This is what should be used for KV cache allocation when model is already on GPU.
     */
    public static long getActualFreeMemoryBytes() {
        if ("cuda".equals(backend())) {
            Long free = cudaFreeMemoryBytes();
            if (free != null) {
                return free;
            }
            return (long) (physicalMemoryBytes() * 0.3);
        }
        return (long) (physicalMemoryBytes() * 0.5);
    }

    /** Returns estimated allocated GPU memory in bytes (approximation). */
    public static long getAllocatedMemory() {
        if ("cuda".equals(backend())) {
            long free = cudaFreeMemoryBytes() != null ? cudaFreeMemoryBytes() : 0L;
            return physicalMemoryBytes() - free;
        }
        return physicalMemoryBytes() / 2;
    }
}
