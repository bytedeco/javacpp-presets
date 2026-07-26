package org.bytedeco.pytorch.geometric.data;

import org.bytedeco.pytorch.Tensor;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public abstract class BaseData {
    // 存储所有的张量或属性，模拟 Python 的 __dict__
    protected final Map<String, Object> attributes = new ConcurrentHashMap<>();

    // 核心设备管理：统一移动数据对象
    public abstract BaseData to(String device);

    public abstract BaseData pinMemory();

    // 属性访问器
    public void put(String key, Object value) {
        attributes.put(key, value);
    }

    @SuppressWarnings("unchecked")
    public <T> T get(String key) {
        return (T) attributes.get(key);
    }

    // 验证图的完整性（如 edge_index 是否越界）
    public abstract boolean validate();

    // 获取所有张量属性的名称，用于自动化的 .to(device) 转换
    public Set<String> tensorAttrNames() {
        return attributes.entrySet().stream()
                .filter(e -> e.getValue() instanceof Tensor)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }
}