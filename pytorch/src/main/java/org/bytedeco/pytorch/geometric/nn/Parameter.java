package org.bytedeco.pytorch.geometric.nn;


import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorBase;
import org.bytedeco.pytorch.TensorTensorHook;
import org.bytedeco.pytorch.global.torch;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * 修复梯度传播的 Parameter 类（完全对齐 Python 版）
 * 核心：保证梯度计算时底层 Tensor 与 Parameter 完全绑定
 */
public class Parameter extends Tensor {
    private boolean requiresGrad;
    private Tensor grad; // 显式维护梯度（模仿 Python: param.grad）

    // 构造函数：深度拷贝 Tensor，避免梯度链路丢失
//    public Parameter(Tensor data, boolean requiresGrad) {
//        super(data.clone()); // 关键：克隆 Tensor，保证独立梯度链路
//        this.requiresGrad = requiresGrad;
//        if (requiresGrad) {
//            this.requires_grad_(true); // 同步设置 requires_grad 标记
//            this.grad = data.grad().retainReference();
//        }else {
//            this.grad = null; // 不需要梯度时，显式设置为 null
//        }
////        this.grad = null;
//        data.requires_grad_(true);
//        super.requires_grad_(requiresGrad);
//    }

    public Parameter(Tensor data, boolean requiresGrad) {
        super(data); // 直接引用原 Tensor，不克隆（关键！保证是叶子节点）
        this.requiresGrad = requiresGrad;

        // 核心1：标记当前 Tensor 为叶子节点 + 开启梯度
        if (requiresGrad) {
            super.requires_grad_(true); // 仅修改当前 Parameter 的 requires_grad
            super.retain_grad(); // 强制保留非叶子节点梯度（关键修复）
        } else {
            super.requires_grad_(false);
        }
        // 移除错误的 grad 赋值：避免引用空梯度
        this.grad = data;
    }

    // 重写 grad() 方法：确保返回正确的梯度
    @Override
    public Tensor grad() {
        // 先检查是否有梯度，无则返回空 Tensor（避免JNI崩溃）
        Tensor gradTensor = this.grad;
//        System.out.println("grad: "+gradTensor.grad().defined());
        if (gradTensor == null || gradTensor.getIntrusivePtr().is_empty()) {
            return new Tensor();
        }
        return gradTensor;
    }

    public Parameter(Tensor data) {
        this(data, true);
    }

    // 重载梯度相关方法（核心修复）
//    @Override
//    public Tensor grad() {
//        // 同步底层 Tensor 的梯度到 Parameter
//        this.grad = super.grad();
//        return this.grad;
//    }

//    @Override
//    public Tensor grad() {
//        // 优先返回手动设置的梯度，避免 null
//        if (this.grad != null) {
//            return this.grad;
//        }
//        return super.grad();
//    }

    public void set_grad(Tensor newGrad) {
        this.grad = newGrad;

        // 1. 新梯度为 null：清空底层梯度
        if (newGrad == null) {
            if (super.grad() != null) {
                super.grad().detach_(); // 分离旧梯度
                super.grad().zero_();   // 梯度置零
            }
            return;
        }

        // 2. 新梯度不为 null：同步到底层 Tensor
        // 确保梯度设备/类型与 Parameter 一致
        Tensor gradToSet = newGrad.to(super.device(),super.dtype());
        super.register_hook(new TensorTensorHook() {
            @Override
            public TensorBase call(TensorBase gradInput) {
                // hook 逻辑：返回手动设置的梯度（替换原梯度）
                return gradToSet;
            }
        });
        this.grad = gradToSet.clone();
//        if (super.grad() == null) {
//            // 核心修复：通过 TensorTensorHook 实现 hook，而非 Lambda
//            super.register_hook(new TensorTensorHook() {
//                @Override
//                public TensorBase call(TensorBase gradInput) {
//                    // hook 逻辑：返回手动设置的梯度（替换原梯度）
//                    return gradToSet;
//                }
//            });
//            this.grad = gradToSet.clone();
//        } else {
//            // 底层已有梯度：直接拷贝数值替换
//            super.grad().copy_(gradToSet);
//        }
    }
    @Override
    public Parameter requires_grad_(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        super.requires_grad_(requiresGrad);
        return this;
    }

    public boolean requires_grad() {
        return this.requiresGrad;
    }

    // 设备迁移时保留梯度链路
    public Parameter to(org.bytedeco.pytorch.Device device, torch.ScalarType dtype) {
        Tensor newTensor = super.to(device, dtype).clone();
        Parameter newParam = new Parameter(newTensor, this.requiresGrad);
        if (this.grad != null) {
            newParam.set_grad(this.grad.to(device, dtype));
        }
        return newParam;
    }

    // 模仿 Python: param.data
    public Tensor data() {
        return this;
    }

    // 清空梯度（模仿 Python: param.grad = None）
//    public void zero_grad() {
//        this.grad = null;
//        super.set_grad(null);
//    }
    public void zero_grad() {
        this.grad = null;
        // 原生 Tensor 无 set_grad()，直接通过 grad() 获取并释放
        if (super.grad() != null) {
            super.grad().detach_(); // 分离梯度
            super.grad().zero_();   // 置零（原生方法）
        }
    }

    private boolean isTensorEmpty(Tensor tensor) {
        if (tensor == null) return true;
        try {
            return tensor.numel() == 0 ;
        } catch (Exception e) {
            return true;
        }
    }
}
/**
 * 模仿 Python 版 torch.nn.Parameter 的 Java 实现
 * 核心逻辑：
 * 1. 继承 Tensor，增加 requires_grad 标记
 * 2. 提供与 Python 一致的 API（如 requires_grad_()）
 */
//public class Parameter extends Tensor {
//    private boolean requiresGrad; // 是否需要计算梯度（Python: param.requires_grad）
//
//    // ========== 构造函数（对齐 Python） ==========
//    /**
//     * 从 Tensor 创建 Parameter
//     * @param data 底层张量数据
//     * @param requiresGrad 是否需要梯度
//     */
//    public Parameter(Tensor data, boolean requiresGrad) {
//        super(data); // 继承 Tensor 的底层数据
//        this.requiresGrad = requiresGrad;
//        // 同步设置 Tensor 的 requires_grad 标记（核心！）
//        if (requiresGrad) {
//            this.requires_grad_(true);
//        }
//    }
//
//    /**
//     * 简化构造（默认 requires_grad=True）
//     */
//    public Parameter(Tensor data) {
//        this(data, true);
//    }
//
//    // ========== 核心 API（模仿 Python） ==========
//    /**
//     * 设置是否需要梯度（Python: param.requires_grad_(True)）
//     */
//    public Parameter requires_grad_(boolean requiresGrad) {
//        this.requiresGrad = requiresGrad;
//        super.requires_grad_(requiresGrad); // 同步到底层 Tensor
//        return this;
//    }
//
//    /**
//     * 获取是否需要梯度（Python: param.requires_grad）
//     */
//    public boolean requires_grad() {
//        return this.requiresGrad;
//    }
//
//    /**
//     * 重载 to() 方法，保证设备迁移时保留 Parameter 类型
//     */
//    public Parameter to(org.bytedeco.pytorch.Device device, torch.ScalarType dtype) {
//        Tensor newTensor = super.to(device, dtype );
//        return new Parameter(newTensor, this.requiresGrad);
//    }
//
//    /**
//     * 模仿 Python: param.data（获取底层 Tensor）
//     */
//    public Tensor data() {
//        return this;
//    }
//}