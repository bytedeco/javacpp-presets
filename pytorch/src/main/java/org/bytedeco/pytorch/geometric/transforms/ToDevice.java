/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.ToDevice
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.geometric.data.GraphData;

/** Move all tensor fields of a {@link GraphData} onto {@code device}. */
public class ToDevice implements BaseTransform {

    private final Device device;

    public ToDevice(Device device) {
        if (device == null) {
            throw new NullPointerException("device");
        }
        this.device = device;
    }

    @Override
    public GraphData apply(GraphData data) {
        return TransformUtils.toDevice(data, device);
    }

    public Device getDevice() {
        return device;
    }
}
