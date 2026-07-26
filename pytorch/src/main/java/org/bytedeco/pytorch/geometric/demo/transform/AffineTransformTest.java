package org.bytedeco.pytorch.geometric.demo.transform;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;
import org.bytedeco.pytorch.geometric.transforms.RandomFlip;
import org.bytedeco.pytorch.geometric.transforms.RandomRotate;
import org.bytedeco.pytorch.geometric.transforms.RandomScale;

import static org.bytedeco.pytorch.global.torch.*;
public class AffineTransformTest {
    public static void main(String[] args) {
        System.out.println("=== 启动点云仿射变换测试 ===");

        // 初始化点在 (1, 0, 0)
        Tensor pos = tensor(new float[]{1f, 0f, 0f}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).view(1,3);
        GraphData data = new GraphData(randn(new long[]{1, 1}), null);
        data.pos = pos;

        // 1. 测试 RandomFlip (强制翻转)
        new RandomFlip(0, 1.0).apply(data);
        assert data.pos.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(0))).item_float() == -1f;
        System.out.println("✅ RandomFlip 测试通过 (X轴翻转成功)");

        // 2. 测试 RandomScale
        new RandomScale(2.0f, 2.0f).apply(data); // 强制放大2倍
        assert Math.abs(data.pos.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(0))).item_float() - (-2f)) < 1e-4;
        System.out.println("✅ RandomScale 测试通过 (放大2倍成功)");

        // 3. 测试 RandomRotate (绕Z轴旋转90度)
        // 注意：RandomRotate 内部是随机的，这里主要验证维度和不为空
        new RandomRotate(90f, 2).apply(data);
        assert data.pos.size(0) == 1 && data.pos.size(1) == 3;
        System.out.println("✅ RandomRotate 测试通过 (维度正确)");

        System.out.println("所有仿射变换测试 PASS！");
    }
}