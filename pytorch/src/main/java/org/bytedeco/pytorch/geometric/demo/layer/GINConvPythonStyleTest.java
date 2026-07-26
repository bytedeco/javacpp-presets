package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
import org.bytedeco.pytorch.geometric.nn.conv.GINConv;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 模仿 Python 版 PyTorch 的测试逻辑
 * 验证 GINConv 与 Python 版行为一致
 */
public class GINConvPythonStyleTest {

    public static void main(String[] args) {
        // 模仿 Python: torch.manual_seed(42)
        torch.manual_seed(42);
        Device cpu = new Device(DeviceType.CPU);
        TensorOptions floatOpts = new TensorOptions().dtype(new ScalarTypeOptional(torch.kFloat())).device(new DeviceOptional(cpu));
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(torch.kLong())).device(new DeviceOptional(cpu));

        // 测试1：基础功能（模仿 Python 版基础测试）
        System.out.println("===== 测试1：基础 GINConv（模仿 Python） =====");
        testBasicFunction(floatOpts, longOpts);

        // 测试2：eps 可训练性（核心模仿 Python）
        System.out.println("\n===== 测试2：eps 参数可训练性（模仿 Python） =====");
        testEpsTrainable(floatOpts);

        // 测试3：梯度计算（验证 Parameter 梯度功能）
        System.out.println("\n===== 测试3：梯度计算（模仿 Python 反向传播） =====");
        testGradientCalculation(floatOpts, longOpts);
    }

    /**
     * 测试1：基础功能（对应 Python 版基础前向）
     */
    private static void testBasicFunction(TensorOptions floatOpts, TensorOptions longOpts) {
        // 模仿 Python 定义超参数
        long inChannels = 4;
        long outChannels = 8;
        long numNodes = 10;
        long numEdges = 20;

        // 模仿 Python 创建数据
        // Python: x = torch.randn(numNodes, inChannels)
        Tensor x = randn(new long[]{numNodes, inChannels}, floatOpts);
        // Python: edge_index = torch.randint(0, numNodes, (2, numEdges))
        Tensor edgeIndex = randint(0, numNodes, new long[]{2, numEdges}, longOpts);

        // 模仿 Python 创建 MLP
        // Python: mlp = torch.nn.Sequential(
        //     torch.nn.Linear(inChannels, outChannels),
        //     torch.nn.ReLU(),
        //     torch.nn.Linear(outChannels, outChannels)
        // )
        SequentialImpl mlp = new SequentialImpl();
        mlp.push_back(new LinearImpl(inChannels, outChannels));
        mlp.push_back(new ReLUImpl());
        mlp.push_back(new LinearImpl(outChannels, outChannels));

        // 模仿 Python 创建 GINConv
        // Python: conv = GINConv(mlp, train_eps=True)
        GINConv conv = new GINConv(mlp, true);

        // 模仿 Python 前向传播
        // Python: out = conv(x, edge_index)
        Tensor out = conv.forward(x, edgeIndex);

        // 模仿 Python 维度检查
        // Python: assert out.shape == (numNodes, outChannels)
        if (out.size(0) == numNodes && out.size(1) == outChannels) {
            System.out.println("✅ 维度校验通过（模仿 Python assert）！");
            System.out.println("输出形状: (" + out.size(0) + ", " + out.size(1) + ")");
        } else {
            throw new AssertionError("输出形状错误！预期 (" + numNodes + ", " + outChannels + ")，实际 (" + out.size(0) + ", " + out.size(1) + ")");
        }
    }

    /**
     * 测试2：eps 可训练性（对应 Python: param.requires_grad）
     */
    private static void testEpsTrainable(TensorOptions floatOpts) {
        // 模仿 Python 创建两种 GINConv
        LinearImpl mlp = new LinearImpl(4, 8);

        // Python: conv1 = GINConv(mlp, train_eps=True)
        GINConv conv1 = new GINConv(mlp, true);
        // Python: print(conv1.eps.requires_grad) → True
        System.out.println("train_eps=True 时 eps.requires_grad: " + conv1.eps().requires_grad());

        // Python: conv2 = GINConv(mlp, train_eps=False)
        GINConv conv2 = new GINConv(mlp, false);
        // Python: print(conv2.eps.requires_grad) → False
        System.out.println("train_eps=False 时 eps.requires_grad: " + conv2.eps().requires_grad());

        // 验证逻辑
        if (conv1.eps().requires_grad() && !conv2.eps().requires_grad()) {
            System.out.println("✅ eps 可训练性验证通过！");
        } else {
            throw new RuntimeException("eps 可训练性错误！");
        }
    }

    /**
     * 测试3：梯度计算（模仿 Python 反向传播）
     */
    private static void testGradientCalculation(TensorOptions floatOpts, TensorOptions longOpts) {
        // 模仿 Python 准备数据和模型
        long inChannels = 2;
        long outChannels = 2;
        long numNodes = 3;
        long numEdges = 3;

        // Python: x = torch.randn(numNodes, inChannels, requires_grad=True)
        Tensor x = randn(new long[]{numNodes, inChannels}, floatOpts);
        x.requires_grad_(true);

        // Python: edge_index = torch.tensor([[0,1,2],[1,2,0]])
        long[] edgeData = {0,1,2, 1,2,0};
        Tensor edgeIndex = tensor(edgeData, longOpts).view(2, numEdges);

        // Python: mlp = torch.nn.Linear(inChannels, outChannels)
        LinearImpl mlp = new LinearImpl(inChannels, outChannels);
        // Python: conv = GINConv(mlp, train_eps=True)
        GINConv conv = new GINConv(mlp, true);

        // Python: out = conv(x, edge_index)
        Tensor out = conv.forward(x, edgeIndex);
        // Python: loss = out.sum()
        Tensor loss = out.sum();
        // Python: loss.backward()
        loss.backward();

        // 模仿 Python 检查梯度
        // Python: print(conv.eps.grad is not None) → True
        boolean epsHasGrad = conv.eps().grad() != null;
        // Python: print(x.grad is not None) → True
        boolean xHasGrad = x.grad() != null;

        System.out.println("eps 梯度是否存在: " + epsHasGrad);
        System.out.println("输入 x 梯度是否存在: " + xHasGrad);

        if (epsHasGrad && xHasGrad) {
            System.out.println("✅ 梯度计算验证通过！");
            System.out.println("eps 梯度值: " + conv.eps().grad().item_float());
            System.out.println("x 梯度形状: (" + x.grad().size(0) + ", " + x.grad().size(1) + ")");
        } else {
            throw new RuntimeException("梯度计算失败！");
        }
    }
}
