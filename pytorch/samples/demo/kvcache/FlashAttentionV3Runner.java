package samples.demo.kvcache;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.global.torch.DeviceType.CUDA;

public class FlashAttentionV3Runner {
    public static void main(String[] args) {
        // 获取全局上下文引用
        Context ctx = globalContext();

        // 1. 环境检查：确保硬件和库支持
        if (!Context.hasCUDA() || !Context.hasCuDNN()) {
            throw new RuntimeException("需要 CUDA 和 CuDNN 环境");
        }

        System.out.println("CuDNN Version: " + Context.versionCuDNN());
        // Flash-3 需要 CuDNN 8.9.7+ 或更高版本

        // 2. 配置后端优先级，强制触发 Flash Attention
        // 关闭 Math 路径和高效注意路径，强制走 Flash/CuDNN 路径
        ctx.setSDPUseMath(false);
        ctx.setSDPUseMemEfficient(false);
        ctx.setSDPUseFlash(true);
        ctx.setSDPUseCuDNN(true); // 在 H100 上，CuDNN 后端是 Flash-3 的重要入口

        // 3. 构造符合 Flash-3 要求的张量
        // 注意：Flash-3 对 Head Dimension 有严格要求（通常为 64, 128）
        // 如果是 H100，强烈建议使用 BF16 或 FP8
        Device device = new Device(kCUDA());//CUDA);
        long[] shape = {2, 16, 1024, 128}; // Batch, Heads, SeqLen, Dim

        Tensor q = randn(shape, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(device)));
        Tensor k = randn(shape, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(device)));
        Tensor v = randn(shape, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(device)));

        try {
            // 4. 执行注意力计算
            // scaled_dot_product_attention 内部会自动路由到最优的 Flash-3 内核
            Tensor result = scaled_dot_product_attention(
                    q, k, v,
                    new TensorOptional(),   // attn_mask
                    0.0d,    // dropout
                    false,  // is_causal
                    new DoubleOptional(),  // scale
                    false
            );

            System.out.println("计算成功！结果维度: " + java.util.Arrays.toString(result.sizes().vec().get()));
        } finally {
            // 5. 恢复环境标志位（模拟 Guard 行为）
            ctx.setSDPUseMath(true);
            ctx.setSDPUseMemEfficient(true);
        }
    }
}