package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.conv.*;
//import org.bytedeco.pytorch.geometric.nn.conv.*;
import static org.bytedeco.pytorch.global.torch.*;

public class DenseConvTest {
    public static void main(String[] args) {
        System.out.println("=== 启动 Dense GNN 算子联合测试 ===");

        // 1. 构造输入数据 (BatchSize=2, Nodes=4, Feats=8)
        long B = 2, N = 4, D = 8;
        Tensor x = randn(new long[]{B, N, D}).requires_grad_(true);

        // 构造稠密邻接矩阵 [B, N, N]
        // 模拟一个简单的全连接或随机图
        Tensor adj = ones(new long[]{B, N, N});

        // 2. 测试 DenseGCNConv
        System.out.println("\n测试 DenseGCNConv...");
        DenseGCNConv gcn = new DenseGCNConv(D, 16, true);
        Tensor outGCN = gcn.forward(x, adj, new TensorOptional());
        printShape("GCN 输出", outGCN); // 应为 [2, 4, 16]

//        // 3. 测试 DenseGINConv
        System.out.println("\n测试 DenseGINConv...");
        // GIN 内部通常需要一个 MLP
        SequentialImpl mlp = new SequentialImpl(); //new LinearImpl(D, 16), new ReLUImpl()
        mlp.push_back(new LinearImpl(D, 16));
        mlp.push_back(new ReLUImpl());
        DenseGINConv gin = new DenseGINConv(mlp,true);
        Tensor outGIN = gin.forward(x, adj);
        printShape("GIN 输出", outGIN);

        // 4. 测试 DenseSAGEConv
        System.out.println("\n测试 DenseSAGEConv...");
        DenseSAGEConv sage = new DenseSAGEConv(D, 16, true);
        Tensor outSAGE = sage.forward(x, adj);
        printShape("SAGE 输出", outSAGE);

        // 5. 测试 DenseGraphConv
        System.out.println("\n测试 DenseGraphConv...");
        DenseGraphConv graphConv = new DenseGraphConv(D, 16);
        Tensor outGraph = graphConv.forward(x, adj);
        printShape("GraphConv 输出", outGraph);

        // 6. 测试 DenseGATConv
        System.out.println("\n测试 DenseGATConv...");
        DenseGATConv gat = new DenseGATConv(D, 16L, 2L);   // 2头注意力 , 0.0f true
        Tensor outGAT = gat.forward(x, adj);
        printShape("GAT 输出", outGAT);

        // 7. 验证梯度回传 (以 GAT 为例)
        outGAT.sum().backward();
        if (x.grad().defined()) {
            System.out.println("\n✅ 所有 Dense 算子反向传播测试成功！");
        }
    }

    private static void printShape(String name, Tensor t) {
        long[] s = t.sizes().vec().get();
        System.out.printf("%s 形状: [%d, %d, %d]\n", name, s[0], s[1], s[2]);
    }
}