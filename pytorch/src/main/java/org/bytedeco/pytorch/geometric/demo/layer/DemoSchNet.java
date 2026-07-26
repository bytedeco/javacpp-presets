package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.geometric.nn.model.SchNet;

import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.*;

public class DemoSchNet {
   public static void main(String[] args) {
        System.out.println("=== Starting SchNet Test ===");

        // 1. 配置参数
        long hiddenChannels = 128;
        long numFilters = 128;
        int numInteractions = 6;
        int numGaussians = 50;
        double cutoff = 10.0;

        // 2. 模拟原子数据
        // 假设有 5 个原子，其原子序数（如 1=H, 6=C 等）
        Tensor z = tensor(new long[]{1, 6, 6, 1, 8}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        // 5 个原子的 3D 坐标 [N, 3]
        Tensor pos = randn(new long[]{5, 3}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        // 模拟 Batch：所有原子属于同一个分子
        Tensor batch = zeros(new long[]{5}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        try (PointerScope scope = new PointerScope()) {
            // 3. 初始化模型
            SchNet model = new SchNet(hiddenChannels, numFilters, numInteractions, numGaussians, cutoff);

            // 将模型和数据移至 MPS (Mac GPU)
            Device device = new Device(DeviceType.CPU);// hasMPS() ? new Device(DeviceType.MPS) : new Device(DeviceType.CPU);
            model.to(device, kFloat(), false);
            z = z.to(device, ScalarType.Long);
            pos = pos.to(device, ScalarType.Float);
            batch = batch.to(device, ScalarType.Long);


            // 4. 前向传播
            // SchNet 返回 [N, 1] 或全图预测值
            Tensor out = model.forward(z, pos, batch);

            System.out.println("SchNet Output Shape: " + Arrays.toString(out.shape()));

            // 5. 梯度测试
            out.sum().backward();
            System.out.println("Backward Pass Successful!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
