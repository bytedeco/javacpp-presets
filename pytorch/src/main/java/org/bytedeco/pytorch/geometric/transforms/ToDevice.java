package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * ToDevice: 设备转换 (CPU/GPU)
 */
public class ToDevice implements BaseTransform {
    private Device device;
    public ToDevice(Device device) { this.device = device; }

    @Override
    public GraphData apply(GraphData data) {
        data.x = data.x.to(device, torch.ScalarType.Float);
        data.edge_index = data.edge_index.to(device, torch.ScalarType.Long);
        if (data.get("train_mask") != null) data.put("train_mask", data.get("train_mask").to(device, torch.ScalarType.Bool));
        return data;
    }
}