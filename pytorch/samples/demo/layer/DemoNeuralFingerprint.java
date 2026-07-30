package samples.demo.layer;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.nn.model.NeuralFingerprint;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.*;

public class DemoNeuralFingerprint {
   public static void main(String[] args) {
        System.out.println("=== Starting NeuralFingerprint Test ===");

        long inChannels = 16;  // 原子特征维度
        long hiddenChannels = 32; // 中间原子特征维度
        long outChannels = 64;    // 我们期望的最终指纹长度 (也是 fingerprintDim)
        int numLayers = 3;

        // 1. 构造模拟分子数据
        // 8 个原子，16 维特征
        Tensor x = randn(new long[]{8, inChannels}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        // 10 条化学键
        Tensor edge_index = randint(0, 8, new long[]{2, 10}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        // 模拟 Batch：所有原子属于同一个分子
        Tensor batch = zeros(new long[]{8}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        try {
            // 2. 初始化模型 NeuralFingerprint(long inChannels, long hiddenChannels, long fingerprintDim, int numLayers)
            NeuralFingerprint nfp = new NeuralFingerprint(inChannels, hiddenChannels, outChannels, numLayers);

            // 3. 前向传播
            // 返回形状通常为 [1, outChannels]，代表整个分子的指纹
            Tensor fingerprint = nfp.forward(x, edge_index, batch);

            System.out.println("NeuralFingerprint Shape: " + Arrays.toString(fingerprint.shape()));

            if (fingerprint.size(1) == outChannels) {
                System.out.println("Verification Passed: Fingerprint length matches.");
            } else {
                System.out.println("Verification Failed: Fingerprint length " + fingerprint.size(1) + " does not match expected " + outChannels);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}