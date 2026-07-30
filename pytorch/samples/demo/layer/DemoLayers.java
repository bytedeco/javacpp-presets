package samples.demo.layer;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.*;

import java.util.Arrays;

public class DemoLayers {

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("   GNN Framework Layer Verification");
        System.out.println("==========================================\n");

        // 1. 构造通用假数据
        // 节点数: 5, 特征数: 16
        long numNodes = 5;
        long inChannels = 16;
        long outChannels = 32;

        Tensor x = torch.randn(new long[]{numNodes, inChannels});

        // 构造边索引 [2, NumEdges]
        // 0->1, 1->2, 2->3, 3->4, 4->0 (环状)
        Tensor edge_index = torch.tensor(new long[]{
                0, 1, 2, 3, 4,
                1, 2, 3, 4, 0
        }).reshape(2, 5);

        System.out.println("Input X Shape: " + Arrays.toString(x.shape()));
        System.out.println("Edge Index Shape: " + Arrays.toString(edge_index.shape()));
        System.out.println("------------------------------------------");

        try {
            testSAGEConv(x, edge_index, inChannels, outChannels);
            testGINConv(x, edge_index, inChannels, outChannels);
            testEdgeConv(x, edge_index, inChannels, outChannels);
            testTAGConv(x, edge_index, inChannels, outChannels);
            testSGConv(x, edge_index, inChannels, outChannels);
            testGatedGraphConv(outChannels, edge_index); // Gated 输入特征通常要匹配隐层维度
            testARMAConv(x, edge_index, inChannels, outChannels);
            testTransformerConv(x, edge_index, inChannels, outChannels);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("!!! Test Failed !!!");
        }
    }

    // --- 1. org.bytedeco.pytorch.geometric.nn.conv.SAGEConv Test ---
    private static void testSAGEConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[1] Testing org.bytedeco.pytorch.geometric.nn.conv.SAGEConv... ");
        SAGEConvV2 conv = new SAGEConvV2(in, in, out);
        Tensor res = conv.forward(x, edge_index);
        checkShape(res, new long[]{x.size(0), out});
    }

    // --- 2. org.bytedeco.pytorch.geometric.nn.conv.GINConv Test ---
    private static void testGINConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[2] Testing org.bytedeco.pytorch.geometric.nn.conv.GINConv... ");
        // org.bytedeco.pytorch.geometric.nn.model.GIN 需要传入一个 MLP (SequentialImpl)
        // MLP Input dim = in (因为 org.bytedeco.pytorch.geometric.nn.model.GIN 是 (1+e)x + agg，维度不变)
        SequentialImpl mlp = new SequentialImpl();
        mlp.push_back(new LinearImpl(in, out));
        mlp.push_back(new ReLUImpl());
        mlp.push_back(new LinearImpl(out, out));

        GINConv conv = new GINConv(mlp, true);
        Tensor res = conv.forward(x, edge_index);
        checkShape(res, new long[]{x.size(0), out});
    }

    // --- 3. org.bytedeco.pytorch.geometric.nn.conv.EdgeConv Test ---
    private static void testEdgeConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[3] Testing org.bytedeco.pytorch.geometric.nn.conv.EdgeConv... ");
        // org.bytedeco.pytorch.geometric.nn.conv.EdgeConv 内部使用 MLP，且输入是 concat(x_i, x_j - x_i)，所以输入维度是 2 * in
        EdgeConv conv = new EdgeConv(in, out);
        // 注意：我们在 org.bytedeco.pytorch.geometric.nn.conv.EdgeConv 构造函数里已经写死了 MLP 的结构，只要传入 in/out 即可

        Tensor res = conv.forward(x, edge_index);
        checkShape(res, new long[]{x.size(0), out});
    }

    // --- 4. org.bytedeco.pytorch.geometric.nn.conv.TAGConv Test ---
    private static void testTAGConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[4] Testing org.bytedeco.pytorch.geometric.nn.conv.TAGConv (K=2)... ");
        TAGConv conv = new TAGConv(in, out, 2);
        Tensor res = conv.forward(x, edge_index);
        checkShape(res, new long[]{x.size(0), out});
    }

    // --- 5. org.bytedeco.pytorch.geometric.nn.conv.SGConv Test ---
    private static void testSGConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[5] Testing org.bytedeco.pytorch.geometric.nn.conv.SGConv (K=2)... ");
        SGConv conv = new SGConv(in, out, 2);
        Tensor res = conv.forward(x, edge_index);
        checkShape(res, new long[]{x.size(0), out});
    }

    // --- 6. org.bytedeco.pytorch.geometric.nn.conv.GatedGraphConv Test ---
    private static void testGatedGraphConv(long out, Tensor edge_index) {
        System.out.print("[6] Testing org.bytedeco.pytorch.geometric.nn.conv.GatedGraphConv... ");
        // org.bytedeco.pytorch.geometric.nn.conv.GatedGraphConv 是 RNN 结构，输入特征维度必须等于输出(隐层)维度
        // 我们构造一个新的 x 匹配 out 维度
        Tensor xGated = torch.randn(new long[]{5, out});

        GatedGraphConv conv = new GatedGraphConv(out, 3); // 3 layers
        Tensor res = conv.forward(xGated, edge_index);
        checkShape(res, new long[]{5, out});
    }

    // --- 7. org.bytedeco.pytorch.geometric.nn.conv.TransformerConv Test ---
    private static void testTransformerConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[7] Testing org.bytedeco.pytorch.geometric.nn.conv.TransformerConv (Heads=2)... ");
        long heads = 2;
        TransformerConv conv = new TransformerConv(in, out, heads);
        Tensor res = conv.forward(x, edge_index);
        // 默认实现是 Concat 模式: out_dim = heads * out
        checkShape(res, new long[]{x.size(0), heads * out});
    }

    // --- 8. org.bytedeco.pytorch.geometric.nn.conv.ARMAConv Test ---
    private static void testARMAConv(Tensor x, Tensor edge_index, long in, long out) {
        System.out.print("[8] Testing org.bytedeco.pytorch.geometric.nn.conv.ARMAConv... ");
        ARMAConv conv = new ARMAConv(in, out, 2, 2); // 2 stacks, 2 layers
        Tensor res = conv.forward(x, edge_index);
        checkShape(res, new long[]{x.size(0), out});
    }

    // --- 辅助验证工具 ---
    private static void checkShape(Tensor tensor, long[] expected) {
        long[] actual = tensor.shape();
        boolean match = Arrays.equals(actual, expected);
        if (match) {
            System.out.println("PASS -> Shape: " + Arrays.toString(actual));
        } else {
            System.out.println("FAIL");
            System.out.println("    Expected: " + Arrays.toString(expected));
            System.out.println("    Actual:   " + Arrays.toString(actual));
            throw new RuntimeException("Shape Mismatch");
        }
    }
}
