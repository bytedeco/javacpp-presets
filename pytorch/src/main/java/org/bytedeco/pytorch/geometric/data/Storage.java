package org.bytedeco.pytorch.geometric.data;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;// 通用存储基类，支持 dynamic attributes
public abstract class Storage {
    protected final Map<String, Object> _data = new ConcurrentHashMap<>();

    public void put(String key, Object value) { _data.put(key, value); }
    public Object get(String key) { return _data.get(key); }

    // 自动将存储中的所有 Tensor 转移到指定设备
    public void to(String device) {
        _data.forEach((k, v) -> {
            if (v instanceof Tensor) put(k, ((Tensor) v).to(new Device(device), torch.ScalarType.Float));
        });
    }
}


