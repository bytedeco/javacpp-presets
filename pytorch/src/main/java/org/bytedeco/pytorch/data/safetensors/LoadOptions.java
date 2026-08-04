package org.bytedeco.pytorch.data.safetensors;

import org.bytedeco.pytorch.Device;

import java.util.Locale;
import java.util.Objects;

/**
 * Options bag for enterprise {@code torch.load}-compatible safetensors loading.
 *
 * <p>Mirrors the common Python kwargs:
 * <pre>
 *   torch.load(path, map_location=..., weights_only=True)
 *   safetensors.torch.load_file(path, device=...)
 *   model.load_state_dict(sd, strict=True/False)
 * </pre>
 *
 * <p>Pure Java — no CPython / no external safetensors jar.
 */
public final class LoadOptions {

    /** When true, return only {@code Map&lt;String,Tensor&gt;} — never build a Module. */
    public final boolean weightsOnly;

    /**
     * Target device for tensors after load ({@code null} = leave on CPU / mmap host).
     * Accepts {@link Device} or a device string such as {@code "cpu"}, {@code "cuda:0"}, {@code "mps"}.
     */
    public final Device mapLocation;

    /** Prefer mmap / {@code from_blob} zero-copy for large tensors. */
    public final boolean zeroCopy;

    /**
     * When injecting into a Module ({@code load_into} / {@code load_state_dict}),
     * throw on missing keys or shape mismatches.
     */
    public final boolean strict;

    /**
     * When true and loading a directory / index, also run FP8 dequant
     * ({@code weight_scale_inv}) like HF compressed-tensors dumps.
     */
    public final boolean dequantFp8;

    /** Optional dtype cast after load (null = keep on-disk dtype). */
    public final org.bytedeco.pytorch.global.torch.ScalarType dtype;

    private LoadOptions(Builder b) {
        this.weightsOnly = b.weightsOnly;
        this.mapLocation = b.mapLocation;
        this.zeroCopy = b.zeroCopy;
        this.strict = b.strict;
        this.dequantFp8 = b.dequantFp8;
        this.dtype = b.dtype;
    }

    public static LoadOptions defaults() {
        return new Builder().build();
    }

    /** Python {@code weights_only=True} equivalent. */
    public static LoadOptions weightsOnly() {
        return new Builder().weightsOnly(true).build();
    }

    public static LoadOptions weightsOnly(Device device) {
        return new Builder().weightsOnly(true).mapLocation(device).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder()
                .weightsOnly(weightsOnly)
                .mapLocation(mapLocation)
                .zeroCopy(zeroCopy)
                .strict(strict)
                .dequantFp8(dequantFp8)
                .dtype(dtype);
    }

    public static final class Builder {
        private boolean weightsOnly = false;
        private Device mapLocation = null;
        private boolean zeroCopy = true;
        private boolean strict = true;
        private boolean dequantFp8 = true;
        private org.bytedeco.pytorch.global.torch.ScalarType dtype = null;

        public Builder weightsOnly(boolean v) { this.weightsOnly = v; return this; }
        public Builder mapLocation(Device d) { this.mapLocation = d; return this; }
        public Builder mapLocation(String deviceSpec) {
            this.mapLocation = parseDevice(deviceSpec);
            return this;
        }
        public Builder device(Device d) { return mapLocation(d); }
        public Builder device(String deviceSpec) { return mapLocation(deviceSpec); }
        public Builder zeroCopy(boolean v) { this.zeroCopy = v; return this; }
        public Builder strict(boolean v) { this.strict = v; return this; }
        public Builder dequantFp8(boolean v) { this.dequantFp8 = v; return this; }
        public Builder dtype(org.bytedeco.pytorch.global.torch.ScalarType d) {
            this.dtype = d;
            return this;
        }

        public LoadOptions build() {
            return new LoadOptions(this);
        }
    }

    /**
     * Parse device specs used by Python {@code map_location}:
     * {@code "cpu"}, {@code "cuda"}, {@code "cuda:0"}, {@code "mps"}, {@code null}.
     */
    public static Device parseDevice(String spec) {
        if (spec == null || spec.isBlank()) return null;
        String s = spec.trim().toLowerCase(Locale.ROOT);
        try {
            return new Device(s);
        } catch (Throwable t) {
            // fall back to common aliases
            if ("cpu".equals(s)) {
                return new Device(org.bytedeco.pytorch.global.torch.DeviceType.CPU);
            }
            if (s.startsWith("cuda")) {
                byte idx = 0;
                int colon = s.indexOf(':');
                if (colon > 0 && colon + 1 < s.length()) {
                    try { idx = Byte.parseByte(s.substring(colon + 1).trim()); } catch (NumberFormatException ignored) {}
                }
                return new Device(org.bytedeco.pytorch.global.torch.DeviceType.CUDA, idx);
            }
            if ("mps".equals(s) || "mps:0".equals(s)) {
                return new Device(org.bytedeco.pytorch.global.torch.DeviceType.MPS);
            }
            throw new IllegalArgumentException("Unrecognized map_location / device: " + spec, t);
        }
    }

    @Override
    public String toString() {
        return "LoadOptions{weightsOnly=" + weightsOnly
                + ", mapLocation=" + mapLocation
                + ", zeroCopy=" + zeroCopy
                + ", strict=" + strict
                + ", dequantFp8=" + dequantFp8
                + ", dtype=" + dtype + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof LoadOptions)) return false;
        LoadOptions that = (LoadOptions) o;
        return weightsOnly == that.weightsOnly
                && zeroCopy == that.zeroCopy
                && strict == that.strict
                && dequantFp8 == that.dequantFp8
                && Objects.equals(mapLocation, that.mapLocation)
                && Objects.equals(dtype, that.dtype);
    }

    @Override
    public int hashCode() {
        return Objects.hash(weightsOnly, mapLocation, zeroCopy, strict, dequantFp8, dtype);
    }
}
