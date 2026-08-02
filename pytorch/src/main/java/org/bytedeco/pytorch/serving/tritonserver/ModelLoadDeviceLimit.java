package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.pytorch.serving.tritonserver.enums.InstanceGroupKind;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * Memory limit for loading models on a device.
 *
 * <p>Python {@code ModelLoadDeviceLimit} / {@code TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit}.
 */
public final class ModelLoadDeviceLimit {
    private final InstanceGroupKind kind;
    private final int device;
    private final double fraction;

    public ModelLoadDeviceLimit(InstanceGroupKind kind, int device, double fraction) {
        if (kind == null) {
            throw new TritonInvalidArgumentException("kind must not be null");
        }
        this.kind = kind;
        this.device = device;
        this.fraction = fraction;
    }

    public InstanceGroupKind kind() {
        return kind;
    }

    public int device() {
        return device;
    }

    public double fraction() {
        return fraction;
    }
}
