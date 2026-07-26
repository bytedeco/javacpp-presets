package org.bytedeco.pytorch.geometric.demo.layer;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GATConv;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
import org.bytedeco.pytorch.geometric.data.GraphData;

import java.util.Arrays;
//import org.gnn.framework.data.org.bytedeco.pytorch.geometric.data.GraphData;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
//import org.gnn.framework.layers.org.bytedeco.pytorch.geometric.nn.conv.GATConv;

public class Demo {


    public static void main(String[] args) {
        // 1. 构造假数据
        // 3个节点，2个特征
        Tensor x = torch.rand(3, 2);
        // 边: 0->1, 1->2, 2->0 (闭环)
        Tensor edge_index = torch.tensor(new long[]{
                0, 1, 2, // source
                1, 2, 0  // target
        }).reshape(2, 3);

        GraphData data = new GraphData(x, edge_index);

        System.out.println("--- Testing GCN ---");
        GCNConv gcn = new GCNConv(2, 4);
        Tensor outGCN = gcn.forward(data.x, data.edge_index);
        System.out.println("GCN Output Shape: " + Arrays.toString(outGCN.shape()));
        // Expected: [3, 4]

        System.out.println("--- Testing org.bytedeco.pytorch.geometric.nn.model.GAT ---");
        // 2 features in, 4 features out, 2 heads -> total output width 8
        GATConv gat = new GATConv(2, 4, 2, 0.2);
        Tensor outGAT = gat.forward(data.x, data.edge_index);
        System.out.println("org.bytedeco.pytorch.geometric.nn.model.GAT Output Shape: " + Arrays.toString(outGAT.shape()));
//         Expected: [3, 8]

        // --- Test 3: org.bytedeco.pytorch.geometric.nn.conv.GraphConv (New!) ---
        System.out.println("\n--- Testing org.bytedeco.pytorch.geometric.nn.conv.GraphConv ---");
//        // 输入2维 -> 输出4维
//        GraphConv graphConv = new GraphConv(2, 4);
//        Tensor outGraph = graphConv.forward(data.x, data.edge_index);
//
//        System.out.println("org.bytedeco.pytorch.geometric.nn.conv.GraphConv Output: " + Arrays.toString(outGraph.shape()));
//
////         简单验证数值没崩
//        System.out.println("org.bytedeco.pytorch.geometric.nn.conv.GraphConv First Row: " + outGraph.slice(0, 0, 1, 1));
    }
}
