package samples.demo.kvcache;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.kvcache.PagedBlockManager;
// use samples.demo.kvcache.PagedKvBuffer (same package)

import static org.bytedeco.pytorch.global.torch.*;

public class PagedAttentionDemo {
    public static void main(String[] args) {
        int numLayers = 12;
        int blockSize = 16; // 每页存储 16 个 Token
        int headDim = 128;
        int maxBlocks = 1000;

        // 1. 初始化全局物理内存管理器
        try (PagedBlockManager manager = PagedBlockManager.withDtypeValue(
                maxBlocks, numLayers, blockSize, headDim, torch.kFloat().value)) {

            // 2. 为一个新请求创建 Paged KV Buffer
            try (PagedKvBuffer sessionA = new PagedKvBuffer("session-123", manager)) {

                // 模拟模型输出的 Token Tensor [head_dim]
                Tensor mockK = torch.randn(new long[]{headDim}, new org.bytedeco.pytorch.TensorOptions());
                Tensor mockV = torch.randn(new long[]{headDim}, new org.bytedeco.pytorch.TensorOptions());

                // 3. 逐层写入 KV Cache
                for (int t = 0; t < 20; t++) { // 模拟生成 20 个 Token
                    for (int l = 0; l < numLayers; l++) {
                        sessionA.appendToken(l, 0, mockK); // 存 Key
                        sessionA.appendToken(l, 1, mockV); // 存 Value
                    }
                    sessionA.finishToken();

                    if (t == 15) {
                        System.out.println("Token 16 reached, new physical block allocated!");
                    }
                }

                // 4. 获取 Block Table 准备进行 Attention 计算
                int[] blockIds = sessionA.getBlockIds();
                System.out.println("Session A is using physical blocks: " + java.util.Arrays.toString(blockIds));

                // 此时你可以将 blockIds 传给自定义的 LibTorch C++ 算子 
                // 或者手动在 Java 端拼凑 Tensor（虽然后者较慢）
            }
        }
    }
}
