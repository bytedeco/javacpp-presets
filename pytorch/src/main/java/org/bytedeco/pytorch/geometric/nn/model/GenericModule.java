package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Tensor;

/**
 * 模拟 Python 的底层 Module，提供统一的 forward 接口
 */
public abstract class GenericModule extends org.bytedeco.pytorch.nn.Module {
    public GenericModule() {
        super();
    }

    // 为了方便转换，提供一个静态辅助方法
    public static GenericModule cast(org.bytedeco.pytorch.nn.Module module) {
        return (GenericModule) module;
    }

    // 定义统一的输入接口，使用变长参数以适应不同的 Layer 输入需求
    public abstract Tensor forward(Tensor... inputs);
}