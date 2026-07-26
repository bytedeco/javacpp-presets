package org.bytedeco.pytorch.geometric.demo.layer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.nn.conv.*;

import java.util.*;

import static org.bytedeco.pytorch.global.torch.*;

public class GNNLayerTestRunner {
    public static void main(String[] args) {
        manual_seed(42);
        System.out.println("🚀 开始 GNN 算子自动化测试...\n");

        // --- 1. 测试 MixHopConv ---
 
        // --- 2. 测试 PDNConv ---

        // --- 3. 测试 GPSConv ---
  
        // --- 4. 测试 AntiSymmetricConv ---
        testAntiSymmetric();
        testARMA();
        testAGNN();
        testAPPNP();
        
        testClusterGCN();
        testCuGraphRGCN();
        testCuGraphSAGE();
        testCuGraphGAT();
        testCG();
        testCheb();

        testDirGNN();
        testDNA();
        testDynamicEdge();
        
        testEG();
        testFA();
        testFiLM();
        testFastRGCN();
        testFeaSt();
        testFusedGAT();
        
        testGPS();
        testGINE();
        testGEN();
        testGMM();
        testGCN2();
        testGravNet();
        testGatedGraph();
        testGraphConv();
//        testGraph();
        testGeneral();
        testGATv2();
        
        testHEAT();
        testHAN();
        testHypergraph();
        testHetero();
        testHGT();
        
        testLG();
        testLE();
        testMF();
        testMixHop();
        testNNConv();
        
        testResGated();
        testRGCN();
        testRGAT();
        
        testSG();
        testSigned();
        testSimple();
        testSuperGAT();
        testSSG2();
        testSuperGAT2();
        testSpline();

        testPAN();
        testPDN();
        testPPF();
        testPointGNN();
        testPointNet();
        testPointTransformer();
        
        testTAG();
        testX();
        testWL();
        testWLContinuous();
        
  
        System.out.println("\n🎉 所有算子基础测试完成！");
    }


    public static void testHGT() {
        // 1. 定义异构图结构
        // 节点类型：paper (128维), author (64维)
        Map<String, Integer> inChannelsDict = new HashMap<>();
        inChannelsDict.put("paper", 128);
        inChannelsDict.put("author", 64);

        List<String> nodeTypes = Arrays.asList("paper", "author");

        // 关系类型：[源节点, 关系名, 目标节点]
        List<String[]> edgeTypes = new ArrayList<>();
        edgeTypes.add(new String[]{"paper", "cites", "paper"});
        edgeTypes.add(new String[]{"author", "writes", "paper"});

        // 2. 初始化 HGTConv (输出维度 128, 8个头)
        int outChannels = 128;
        int heads = 8;
        HGTConv hgt = new HGTConv(inChannelsDict, outChannels, nodeTypes, edgeTypes, heads);

        // 3. 构造模拟数据
        Map<String, Tensor> xDict = new HashMap<>();
        xDict.put("paper", randn(new long[]{100, 128}));  // 100个论文
        xDict.put("author", randn(new long[]{50, 64}));   // 50个作者

        Map<String[], Tensor> edgeIndexDict = new HashMap<>();
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));
        // paper -> paper (同构边)
        edgeIndexDict.put(edgeTypes.get(0), randint(0, 100, new long[]{2, 500}, longOpt));
        // author -> paper (异构边)
        edgeIndexDict.put(edgeTypes.get(1), randint(0, 50, new long[]{2, 300}, longOpt));

        // 4. 前向传播
        System.out.println("🚀 启动 HGT 前向传播测试...");
        try (PointerScope scope = new PointerScope()) {
            Map<String, Tensor> out = hgt.forward(xDict, edgeIndexDict);

            // 5. 校验结果
            for (String type : out.keySet()) {
                Tensor outTensor = out.get(type);
                System.out.println("节点类型: " + type + " | 输出维度: " + Arrays.toString(outTensor.sizes().vec().get()));

                // 验证 shape: 论文节点应为 [100, 128]
                if (type.equals("paper") && outTensor.size(0) == 100 && outTensor.size(1) == 128) {
                    System.out.println("✅ " + type + " 维度校验通过");
                }
            }
        } catch (Exception e) {
            System.err.println("❌ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    static void testCheb() {
        System.out.println("--- Testing ChebConv ---");

        long inC = 16L;
        long outC = 32L;
        int K = 3; // 切比雪夫多项式的阶数

        // 签名: (long inChannels, long outChannels, int K, String normalization, boolean hasBias)
        // normalization: "sym", "rw" 或 null
        ChebConv conv = new ChebConv(inC, outC, K, "sym", true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        int numNodes = 10;
        int numEdges = 25;

        Tensor x = randn(new long[]{numNodes, inC}, floatOpt);
        Tensor edgeIndex = getEdges(numNodes, numEdges);

        // edge_weight 通常是可选的，但在测试中可以提供
        Tensor edgeWeight =ones(new long[]{numEdges}, floatOpt);

        // lambda_max: 最大特征值。在没有预计算的情况下，通常传入 2.0 (Scalar Tensor)
        Tensor lambdaMax =tensor(2.0, floatOpt);

        // 签名: forward(Tensor x, Tensor edge_index, Tensor edge_weight, Tensor lambda_max)
        Tensor out = conv.forward(x, edgeIndex, edgeWeight, lambdaMax);

        // 校验维度: [10, 32]
        GNNTester.assertShape(out, numNodes, (int)outC);
        System.out.println("✅ ChebConv Passed!");
    }
    static void testPointTransformer() {
        System.out.println("--- Testing PointTransformerConv ---");
        long inC = 16L, outC = 32L;

        // 构造具体的实现类
        SequentialImpl posNN = new SequentialImpl();
        posNN.push_back("linear1", new LinearImpl(3, (int)outC)); //new LinearImpl(3, (int)outC)
        SequentialImpl attnNN = new SequentialImpl();
        attnNN.push_back("linear1", new LinearImpl((int)outC, (int)outC)); //new LinearImpl((int)outC, (int)outC)

        PointTransformerConv conv = new PointTransformerConv(inC, outC,3, posNN, attnNN);

        TensorOptions options = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, options);
        Tensor pos = randn(new long[]{10, 3}, options);
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, pos, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
        System.out.println("✅ PointTransformerConv Passed!");
    }
    static void testX() {
        System.out.println("--- Testing XConv ---");
        // 签名: (long in, long out, int dim, int kernelSize, Integer hiddenChannels, int dilation, boolean hasBias)
        long inC = 16L, outC = 32L;
        int dim = 3; // 3D 坐标
        int kernelSize = 8;
        int dilation = 1;

        XConv conv = new XConv(inC, outC, dim, kernelSize, 16, dilation, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        int numNodes = 20;
        Tensor x = randn(new long[]{numNodes, inC}, floatOpt);
        Tensor pos = randn(new long[]{numNodes, dim}, floatOpt);
        // batch 向量用于区分不同的点云示例
        Tensor batch = zeros(new long[]{numNodes}, longOpt);

        // 签名: forward(Tensor x, Tensor pos, Tensor batch)
        Tensor out = conv.forward(x, pos, batch);

        // 注意：XConv 可能对点云进行下采样，或者保持原规模，取决于实现
        // 假设为常规卷积保持规模：
        GNNTester.assertShape(out, numNodes, (int)outC);
    }
    static void testPointGNN() {
        System.out.println("--- Testing PointGNNConv ---");
        int inC = 16;
        int outC = 32;

        // mlpH: 16 -> 3 (生成 3D 偏移)
        SequentialImpl mlpH = new SequentialImpl();
        mlpH.push_back("h_lin", new LinearImpl(inC, 3));

        // mlpF: (16 + 3) -> 16 (特征与坐标融合)
        SequentialImpl mlpF = new SequentialImpl();
        mlpF.push_back("f_lin", new LinearImpl(inC + 3, inC));

        // mlpG: 16 -> 32 (输出投影)
        SequentialImpl mlpG = new SequentialImpl();
        mlpG.push_back("g_lin", new LinearImpl(inC, outC));

        PointGNNConv conv = new PointGNNConv(mlpH, mlpF, mlpG);

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{15, inC}, fOpt);
        Tensor pos = randn(new long[]{15, 3}, fOpt);
        Tensor edgeIndex = getEdges(15, 40);

        Tensor out = conv.forward(x, pos, edgeIndex);
        GNNTester.assertShape(out, 15, outC);
        System.out.println("✅ PointGNNConv Passed!");
    }
    static void testPointGNN3() {
        System.out.println("--- Testing PointGNNConv ---");

        int inC = 16;
        int outC = 32;

        // 1. mlpH: 节点特征(16) -> 坐标偏移(3)
        SequentialImpl mlpH = new SequentialImpl();
        mlpH.push_back("h_lin", new LinearImpl(inC, 3));

        // 2. mlpF: (节点特征16 + 相对坐标3) -> 隐藏层(16)
        // 输入维度必须是 16 + 3 = 19
        SequentialImpl mlpF = new SequentialImpl();
        mlpF.push_back("f_lin", new LinearImpl(inC + 3, inC));

        // 3. mlpG: 聚合后的特征(16) -> 输出特征(32)
        SequentialImpl mlpG = new SequentialImpl();
        mlpG.push_back("g_lin", new LinearImpl(inC, outC));

        PointGNNConv conv = new PointGNNConv(mlpH, mlpF, mlpG);

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{15, 16}, fOpt);
        Tensor pos = randn(new long[]{15, 3}, fOpt);
        Tensor edgeIndex = getEdges(15, 40);

        // 调用 forward(x, pos, edgeIndex)
        Tensor out = conv.forward(x, pos, edgeIndex);

        GNNTester.assertShape(out, 15, outC);
        System.out.println("✅ PointGNNConv Passed!");
    }
    static void testPointGNN2() {
        System.out.println("--- Testing PointGNNConv ---");

        int inC = 16;
        int outC = 32;
        int posDim = 3;

        // mlpH: 处理偏移量 Δpos [posDim] -> [inC]
        SequentialImpl mlpH = new SequentialImpl();
        mlpH.push_back("linear", new LinearImpl( posDim,inC));
//        new LinearImpl(posDim, inC)
        // mlpF: 处理拼接后的特征 [inC (x_i) + inC (h_out)] -> [inC]
        SequentialImpl mlpF = new SequentialImpl();
        mlpF.push_back("linear", new LinearImpl(inC + posDim, inC));
        // mlpG: 最终输出映射 [inC] -> [outC]
        SequentialImpl mlpG = new SequentialImpl();
        mlpG.push_back("linear", new LinearImpl(inC, outC)); //new LinearImpl(inC, outC)

        PointGNNConv conv = new PointGNNConv(mlpH, mlpF, mlpG);

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{15, inC}, fOpt);
        Tensor pos = randn(new long[]{15, posDim}, fOpt);
        Tensor edgeIndex = getEdges(15, 40);

        // 签名通常为: forward(Tensor x, Tensor pos, Tensor edge_index)
        Tensor out = conv.forward(x, pos, edgeIndex);
        GNNTester.assertShape(out, 15, outC);
    }


    static void testPPF() {
        System.out.println("--- Testing PPFConv ---");
        int inC = 16;
        int outC = 32;
        // 输入：4维PPF特征 + 16维节点特征 = 20
//        SequentialImpl mlp = new SequentialImpl();
//        mlp.push_back("linear1", new LinearImpl(20, 32)); //new LinearImpl(20, 32)

        // localNN: 处理 [4维PPF特征 + inC节点特征] -> [hiddenC]
        SequentialImpl localNN = new SequentialImpl();
        localNN.push_back("linear1", new LinearImpl(20, 24)); //new LinearImpl(20, 24) new LinearImpl(4 + inC, 24)
        // globalNN: 处理聚合后的特征 [hiddenC] -> [outC]
        SequentialImpl globalNN = new SequentialImpl();
        globalNN.push_back("linear1", new LinearImpl(24, outC)); //new LinearImpl(24, outC) new LinearImpl(24, outC)

// 签名: (Module localNN, Module globalNN, boolean addSelfLoops)
        PPFConv conv = new PPFConv(localNN, globalNN, true);
        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, fOpt);
        Tensor pos = randn(new long[]{10, 3}, fOpt);
        Tensor normal = randn(new long[]{10, 3}, fOpt); // 必须提供法向量
        Tensor edgeIndex = getEdges(10, 30);

        Tensor out = conv.forward(x, pos, normal, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
        System.out.println("✅ PPFConv Passed!");
    }
    
    
    static void testSimple() {
        System.out.println("--- Testing SimpleConv ---");
        // 签名: (String aggr, String combineRoot) -> combineRoot 可选: "sum", "cat", null
        SimpleConv conv = new SimpleConv("mean", "sum");

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, fOpt);
        Tensor edgeIndex = getEdges(10, 25);
        Tensor edgeWeight = rand(new long[]{25}, fOpt); // 可选权重

        Tensor out = conv.forward(x, edgeIndex, edgeWeight);
        GNNTester.assertShape(out, 10, 16);
    }

    static void testSigned() {
        System.out.println("--- Testing SignedConv ---");
        // 签名: (long in, long out, boolean firstAggr, boolean hasBias)
        long inC = 16L, outC = 32L;
        SignedConv conv = new SignedConv(inC, outC, true, true);

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, inC}, fOpt);
        Tensor posEdges = getEdges(10, 15);
        Tensor negEdges = getEdges(10, 15);

        // 签名: forward(Tensor x, Tensor pos_edge_index, Tensor neg_edge_index)
        Tensor out = conv.forward(x, posEdges, negEdges);
        // SignedGCN 输出通常是将正负聚合结果拼接，故维度为 outC * 2 = 64
        GNNTester.assertShape(out, 10, 64);
    }

    static void testRGCN() {
        System.out.println("--- Testing RGCNConv ---");
        // 签名: (long in, long out, int numRelations, boolean rootWeight, boolean hasBias)
        long inC = 16L, outC = 32L;
        int numRels = 3;
        RGCNConv conv = new RGCNConv(inC, outC, numRels, true, true);

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions lOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{10, inC}, fOpt);
        Tensor edgeIndex = getEdges(10, 30);
        Tensor edgeType = randint(0, numRels, new long[]{30}, lOpt);

        Tensor out = conv.forward(x, edgeIndex, edgeType);
        GNNTester.assertShape(out, 10, 32);
    }

    static void testRGAT() {
        System.out.println("--- Testing RGATConv ---");
        // 签名: (long in, long out, int numRelations, int heads, boolean concat)
        long inC = 16L, outC = 32L;
        int numRels = 3, heads = 2;
        RGATConv conv = new RGATConv(inC, outC, numRels, heads, true);

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions lOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{10, inC}, fOpt);
        Tensor edgeIndex = getEdges(10, 30);
        Tensor edgeType = randint(0, numRels, new long[]{30}, lOpt);

        Tensor out = conv.forward(x, edgeIndex, edgeType);
        // concat=true, 32 * 2 = 64
        GNNTester.assertShape(out, 10, 64);
    }


    static void testFusedGAT() {
        System.out.println("--- Testing FusedGATConv (CSR/CSC Fused) ---");
        int inC = 16, outC = 32, heads = 2, N = 10, E = 30;
        FusedGATConv conv = new FusedGATConv(inC, outC, heads, true, 0.2);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{N, inC}, floatOpt);
        // 生成随机边
        Tensor edgeIndex = randint(0, N, new long[]{2, E}, longOpt);

        // 1. 使用工具类生成正确的图格式数据
        Object[] graphData = FusedGATConv.toGraphFormat(edgeIndex, N);
        Tensor[] csr = (Tensor[]) graphData[0];
        Tensor[] csc = (Tensor[]) graphData[1];
        Tensor perm = (Tensor) graphData[2];

        // 2. 前向传播
        Tensor out = conv.forward(x, csr, csc, perm);

        // 3. 校验：concat=true, heads=2, out=32 -> 64
        GNNTester.assertShape(out, N, 64);
        System.out.println("✅ FusedGATConv Passed!");
    }
    static void testFusedGAT2() {
        System.out.println("--- Testing FusedGATConv (CSR/CSC Fused) ---");
        // 假设构造函数为: (long in, long out, int heads, boolean concat, double negativeSlope)
        FusedGATConv conv = new FusedGATConv(16L, 32L, 2, true, 0.2);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        int N = 10;
        int E = 30;
        Tensor x = randn(new long[]{N, 16}, floatOpt).contiguous();

        // 构造 CSR 数据 [rowptr, col]
        Tensor rowptr = zeros(new long[]{N + 1}, longOpt); // 需填充实际偏移
        Tensor col = randint(0, N, new long[]{E}, longOpt);
        Tensor[] csr = {rowptr.contiguous(), col.contiguous()};

        // 构造 CSC 数据 [row, colptr]
        Tensor row = randint(0, N, new long[]{E}, longOpt);
        Tensor colptr = zeros(new long[]{N + 1}, longOpt); // 需填充实际偏移
        Tensor[] csc = {row.contiguous(), colptr.contiguous()};

        // 构造排列索引 perm [E]
        Tensor perm = arange(new Scalar(0), new Scalar(E), longOpt).contiguous();

        // 调用 forward(Tensor x, Tensor[] csr, Tensor[] csc, Tensor perm)
        Tensor out = conv.forward(x, csr, csc, perm);

        // concat=true, heads=2, out=32 -> 64
        GNNTester.assertShape(out, N, 64);
    }
    static void testNNConv() {
        System.out.println("--- Testing NNConv ---");

        long inC = 16L;
        long outC = 32L;
        int edgeDim = 4; // 假设边特征是 4 维

        // 核心：nn 的输出维度必须是 inC * outC
        // 这样它才能产生一个适用于 (inC, outC) 的权重矩阵
        SequentialImpl edgeNN = new SequentialImpl();
        edgeNN.push_back("lin1", new LinearImpl(edgeDim, 64));
        edgeNN.push_back("relu", new ReLUImpl());
        edgeNN.push_back("lin2", new LinearImpl(64, (int)(inC * outC)));

        // 签名: (long inChannels, long outChannels, Module nn, String aggr, boolean rootWeight, boolean hasBias)
        NNConv conv = new NNConv(inC, outC, edgeNN, "mean", true, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        int numNodes = 10;
        int numEdges = 25;

        Tensor x = randn(new long[]{numNodes, inC}, floatOpt);
        Tensor edgeIndex = getEdges(numNodes, numEdges);
        // 必须提供 edge_attr，其维度需与 edgeNN 的输入匹配
        Tensor edgeAttr =randn(new long[]{numEdges, edgeDim}, floatOpt);

        // 执行前向传播
        Tensor out = conv.forward(x, edgeIndex, edgeAttr);

        // 校验维度: [10, 32]
        GNNTester.assertShape(out, numNodes, (int)outC);
        System.out.println("✅ NNConv Passed!");
    }
    
//import java.util.HashMap;
//import java.util.Map;

    // paper -> paper (16, 16, 64)
//        convsMap.put("paper,cites,paper", new GraphConv(16, 16, 64));
//

    /// / author -> paper (32, 16, 64)
//        convsMap.put("author,writes,paper", new GraphConv(32, 16, 64));


//        // 边类型 1: paper -> cites -> paper (同构边，使用 SAGEConv)
//        convsMap.put("paper,cites,paper", new SAGEConv(paperDim, outDim));
//
//        // 边类型 2: author -> writes -> paper (异构边，使用 GraphConv)
//        // 注意：这里的 GraphConv 内部会自动处理 Source 和 Target 维度不一致的情况
//        convsMap.put("author,writes,paper", new GraphConv(authorDim, outDim));
    static void testHetero() {
        System.out.println("--- Testing HeteroConv ---");

        // 1. 定义不同类型的节点维度
        long paperDim = 16L;
        long authorDim = 32; //16L; //32l
        long outDim = 64L;

        // 2. 构建卷积映射表 (convsMap)
        // 为每一条边类型分配一个独立的 MessagePassing 算子
        Map<String, MessagePassing> convsMap = new HashMap<>();

        convsMap.put("paper,cites,paper", new SAGEConvV2(paperDim, paperDim, outDim));
        convsMap.put("author,writes,paper", new GraphConv(authorDim, paperDim, outDim));
//        convsMap.put("author,writes,paper", new SAGEConv(authorDim,paperDim, outDim));
        // 3. 构造 HeteroConv
        // aggr: "sum", "mean", "min", "max", "mul", "cat"
        HeteroConv heteroConv = new HeteroConv(convsMap, "sum");

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        // 4. 准备异构节点特征
        Map<String, Tensor> xDict = new HashMap<>();
        xDict.put("paper", randn(new long[]{100, paperDim}, floatOpt));
        xDict.put("author",randn(new long[]{50, authorDim}, floatOpt));

        // 5. 准备异构边索引
        Map<String, Tensor> edgeIndexDict = new HashMap<>();
        edgeIndexDict.put("paper,cites,paper", getEdges(100, 200));
        edgeIndexDict.put("author,writes,paper", randint(0, 50, new long[]{2, 150}, longOpt));

        // 6. 执行前向传播
        // 返回值是一个 Map<String, Tensor>，包含更新后的各类型节点特征
        Map<String, Tensor> outDict = heteroConv.forward(xDict, edgeIndexDict);

        // 7. 校验结果
        GNNTester.assertShape(outDict.get("paper"), 100, (int)outDim);
        // 注意：由于 writes 边的目标是 paper，所以作者特征在此层 HeteroConv 中如果没有对应的被指向边，则不会更新
        System.out.println("✅ HeteroConv Passed!");
    }



    static void testGMM() {
        System.out.println("--- Testing GMMConv ---");
        // 签名: (long in, long out, int dim, int kernelSize, String aggr)
        // 签名: (long in, long out, int dim, int kernelSize, boolean rootWeight, boolean hasBias)
        long inC = 16L;
        long outC = 32L;
        GMMConv conv = new GMMConv(inC, outC, 2, 5, true, true);
        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);
        Tensor pseudo = rand(new long[]{25, 2}, floatOpt); // 伪坐标需在 [0,1]

        Tensor out = conv.forward(x, edgeIndex, pseudo);
        GNNTester.assertShape(out, 10, 32);
    }


    static void testGINE() {
        System.out.println("--- Testing GINEConv ---");
        // 签名: (Module nn, double eps, boolean trainEps, long edgeDim)
        int inC = 16;
        int edgeD = 8; // 边维度与节点维度不同
        int outC = 32;

        // 构造 MLP
        SequentialImpl mlp = new SequentialImpl();
        mlp.push_back("linear1", new LinearImpl(inC, outC)); //new LinearImpl(inC, outC)

        // 构造算子
        GINEConv conv = new GINEConv(mlp, 0.0, true, edgeD, inC);
        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);
        Tensor edgeAttr = randn(new long[]{25, 8}, floatOpt);

        Tensor out = conv.forward(x, edgeIndex, edgeAttr);
        GNNTester.assertShape(out, 10, 32);
    }


    static void testGEN() {
        System.out.println("--- Testing GENConv ---");
        // 签名: (long in, long out, String aggr, float tVal, boolean learnT, float pVal, boolean learnP, Integer edgeDim, float eps, boolean hasBias)
        long inC = 16L;
        long outC = 32L;
        GENConv conv = new GENConv(inC, outC, "softmax", 0.01f, true, 1.0f, false, null, 1e-7f, true);
        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 30);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }


    static void testGCN2() {
        System.out.println("--- Testing GCN2Conv ---");
        // 签名: (long channels, double alpha, double theta, int layer, boolean sharedWeights)
        GCN2Conv conv = new GCN2Conv(16L, 0.1f, 0.5f, 1, true, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor x0 = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 20);

        Tensor out = conv.forward(x, x0, edgeIndex,null);
        GNNTester.assertShape(out, 10, 16);
        System.out.println("✅ GCN2Conv Success!");
    }
//    static void testFusedGAT() {
//        System.out.println("--- Testing FusedGATConv (CSC) ---");
//        // 签名: (long in, long out, int heads, boolean concat, double negativeSlope)
//        FusedGATConv conv = new FusedGATConv(16L, 32L, 2, true, 0.2);
//
//        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
//        int numNodes = 10;
//        int numEdges = 30;
//        Tensor x = randn(new long[]{numNodes, 16}, floatOpt).contiguous();
//        Tensor[] csc = getCSC(numNodes, numEdges); // 使用之前定义的 getCSC 方法
//
//        // forward(Tensor x, Tensor row, Tensor colptr)
//        Tensor out = conv.forward(x, csc[0], csc[1]);
//        GNNTester.assertShape(out, numNodes, 64);
//    }

    static void testFastRGCN() {
        System.out.println("--- Testing FastRGCNConv ---");
        // 签名: (long in, long out, int numRelations, boolean rootWeight, boolean hasBias)
        long inC = 16L;
        long outC = 32L;
        int numRels = 4;
        FastRGCNConv conv = new FastRGCNConv(inC, outC, numRels, true, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{10, inC}, floatOpt);
        Tensor edgeIndex = getEdges(10, 40);
        Tensor edgeType = randint(0, numRels, new long[]{40}, longOpt);

        Tensor out = conv.forward(x, edgeIndex, edgeType);
        GNNTester.assertShape(out, 10, (int)outC);
    }
//    static void testFastRGCN() {
//        System.out.println("--- Testing FastRGCNConv ---");
//        // 签名: (long inChannels, long outChannels, int numRelations, int numBases)
//        FastRGCNConv conv = new FastRGCNConv(16L, 32L, 4, 2);
//
//        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
//        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));
//
//        Tensor x = randn(new long[]{12, 16}, floatOpt);
//        Tensor edgeIndex = getEdges(12, 40);
//        Tensor edgeType = randint(0, 4, new long[]{40}, longOpt);
//
//        Tensor out = conv.forward(x, edgeIndex, edgeType);
//        GNNTester.assertShape(out, 12, 32);
//    }

    static void testFA() {
        System.out.println("--- Testing FAConv ---");
        // 签名: (long inChannels, double eps, double dropout, boolean hasBias)
        FAConv conv = new FAConv(16L, 0.1f, 0.0f, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor x0 = randn(new long[]{10, 16}, floatOpt); // 初始特征残差
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, x0, edgeIndex, null);
        GNNTester.assertShape(out, 10, 16);
    }
    /**
     * 模拟构建 CSC 格式数据
     * @param numNodes 节点数
     * @param numEdges 边数
     * @return 包含 row 和 colptr 的数组
     */
    static Tensor[] getCSC(int numNodes, int numEdges) {
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));
        // row: 存储源节点 [E]
        Tensor row = randint(0, numNodes, new long[]{numEdges}, longOpt).contiguous();
        // colptr: 存储目标节点的偏移量 [numNodes + 1]
        // 简单起见，我们构造一个递增的偏移
        long[] colptrArr = new long[numNodes + 1];
        for (int i = 0; i <= numNodes; i++) {
            colptrArr[i] = (long) i * (numEdges / numNodes);
        }
        colptrArr[numNodes] = numEdges;
        Tensor colptr = tensor(colptrArr, longOpt).contiguous();
        return new Tensor[]{row, colptr};
    }
    static void testCuGraphSAGE() {
        System.out.println("--- Testing CuGraphSAGEConv (CSC) ---");
        // 签名: (long in, long out, String aggr, boolean normalize, boolean rootWeight, boolean hasBias)
        CuGraphSAGEConv conv = new CuGraphSAGEConv(16L, 32L, "mean", true, true, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        int numNodes = 20;
        int numEdges = 50;

        Tensor x = randn(new long[]{numNodes, 16}, floatOpt).contiguous();
        Tensor[] csc = getCSC(numNodes, numEdges);

        // forward(Tensor x, Tensor row, Tensor colptr)
        Tensor out = conv.forward(x, csc[0], csc[1]);
        GNNTester.assertShape(out, numNodes, 32);
        System.out.println("✅ CuGraphSAGEConv CSC Success!");
    }

    static void testCuGraphGAT() {
        System.out.println("--- Testing CuGraphGATConv (CSC) ---");
        // 签名: (long in, long out, long heads, boolean concat, double negativeSlope, boolean hasBias)
        long heads = 4L;
        CuGraphGATConv conv = new CuGraphGATConv(16L, 32L, heads, true, 0.2, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        int numNodes = 15;
        int numEdges = 40;

        Tensor x = randn(new long[]{numNodes, 16}, floatOpt).contiguous();
        Tensor[] csc = getCSC(numNodes, numEdges);

        // forward(Tensor x, Tensor row, Tensor colptr)
        Tensor out = conv.forward(x, csc[0], csc[1]);

        // concat=true, 32 * 4 = 128
        GNNTester.assertShape(out, numNodes, 128);
        System.out.println("✅ CuGraphGATConv CSC Success!");
    }
    static void testCuGraphRGCN() {
        System.out.println("--- Testing CuGraphRGCNConv (CSC) ---");
        // 签名: (long in, long out, int numRelations, boolean rootWeight, boolean hasBias, String aggr)
        int numRelations = 3;
        CuGraphRGCNConv conv = new CuGraphRGCNConv(16L, 32L, numRelations, true, true, "mean");

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        int numNodes = 12;
        int numEdges = 36;

        Tensor x = randn(new long[]{numNodes, 16}, floatOpt).contiguous();
        Tensor[] csc = getCSC(numNodes, numEdges);

        // edge_type 长度必须等于边数 (numEdges)
        Tensor edgeType = randint(0, numRelations, new long[]{numEdges}, longOpt).contiguous();

        // forward(Tensor x, Tensor row, Tensor colptr, Tensor edge_type)
        Tensor out = conv.forward(x, csc[0], csc[1], edgeType);

        GNNTester.assertShape(out, numNodes, 32);
        System.out.println("✅ CuGraphRGCNConv CSC Success!");
    }

    static void testDynamicEdge() {
        System.out.println("--- Testing DynamicEdgeConv ---");
        // 构造内部 MLP：DynamicEdgeConv 通常将 (x_i, x_j - x_i) 拼接，输入维度为 in*2
//        SequentialImpl nn = new SequentialImpl(new LinearImpl(16 * 2, 32));
        SequentialImpl nn = new SequentialImpl();
        nn.push_back("linear1", new LinearImpl(16 * 2, 32)); //new LinearImpl(16 * 2, 32)
        // 签名: (Module nn, int k, String aggr)
        DynamicEdgeConv conv = new DynamicEdgeConv(nn, 5, "max");

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{20, 16}, floatOpt);
        // batch 用于确保 k-NN 只在同一个图的节点内搜索
        Tensor batch = zeros(new long[]{20}, longOpt);

        Tensor out = conv.forward(x, batch);
        GNNTester.assertShape(out, 20, 32);
    }



    static void testCG() {
        System.out.println("--- Testing CGConv ---");
        // 签名: (long channels, int edgeDim, String aggr, boolean batchNorm, boolean hasBias)
        long channels = 16L;
        int edgeDim = 8;
        CGConv conv = new CGConv(channels, edgeDim, "mean", true, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, channels}, floatOpt);
        Tensor edgeIndex = getEdges(10, 30);
        Tensor edgeAttr =randn(new long[]{30, edgeDim}, floatOpt);

        Tensor out = conv.forward(x, edgeIndex, edgeAttr);
        GNNTester.assertShape(out, 10, (int)channels);
    }

    static void testAPPNP() {
        System.out.println("--- Testing APPNP ---");
        // 签名: (int K, double alpha, double dropout, boolean cached)
        APPNP conv = new APPNP(10, 0.1, 0.0, false, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x =randn(new long[]{15, 32}, floatOpt);
        Tensor edgeIndex = getEdges(15, 40);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 15, 32);
    }

    static void testAGNN() {
        System.out.println("--- Testing AGNNConv ---");
        // 签名: (boolean requires_grad, int add_self_loops)
        AGNNConv conv = new AGNNConv(true); //, 2

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 16);
    }
    
    static void testGravNet() {
        System.out.println("--- Testing GravNetConv ---");
        long inC = 16L;
        long outC = 32L;
        int spaceDim = 4;
        int propDim = 8;
        int k = 5;
        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        int numNodes = 20;
        // (long inC, long outC, int spaceDim, int propDim, int numNeighbors)
        GravNetConv conv = new GravNetConv(16L, 32L, 4, 8, 5);

//        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{20, 16}, floatOpt);
// 必须提供 batch 向量，用于 k-NN 搜索时限制在同图中
        Tensor batch = zeros(new long[]{numNodes}, longOpt);
        Tensor out = conv.forward(x,batch);
        GNNTester.assertShape(out, 20, 32);
    }
    static void testSpline() {
        System.out.println("--- Testing SplineConv ---");
        // 签名: (long inChannels, long outChannels, int dim, int kernelSize, boolean isRootWeight, boolean hasBias)
        long inC = 16L;
        long outC = 32L;
        int dim = 2; // 边坐标维度（如 2D 坐标）
        int kernelSize = 5;
        int degree = 1; // B-spline 的度数
        SplineConv conv = new SplineConv(inC, outC, dim, kernelSize,degree, true, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{10, inC}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);
        // Spline 还需要 edge_attr 作为坐标映射，范围在 [0, 1]
        Tensor pseudo = rand(new long[]{25, dim}, floatOpt);

        Tensor out = conv.forward(x, edgeIndex, pseudo);
        GNNTester.assertShape(out, 10, (int)outC);
    }

//    static void testGravNet() {
//        System.out.println("--- Testing GravNetConv ---");
//        // 签名: (long inChannels, long outChannels, int spaceDim, int propDim, int numNeighbors)
//        long inC = 16L;
//        long outC = 32L;
//        GravNetConv conv = new GravNetConv(inC, outC, 4, 8, 5);
//
//        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
//        Tensor x = randn(new long[]{15, inC}, floatOpt);
//
//        Tensor out = conv.forward(x);
//        GNNTester.assertShape(out, 15, (int)outC);
//    }

    static void testSuperGAT() {
        System.out.println("--- Testing SuperGATConv ---");
        // 签名: (long inChannels, long outChannels, int heads, boolean concat, boolean hasBias)
        long inC = 16L;
        long outC = 32L;
        int heads = 2;
        // new SuperGATConv(inC, outC, heads, true, "MX");
        SuperGATConv conv = new SuperGATConv(inC, outC, heads, true, "MX");

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, inC}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);

        // concat=true -> 32 * 2 = 64
        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 64);
    }

    //--- Testing WLConvContinuous ---
    //✅ [通过] 维度数量不匹配
    //✅ [通过] 维度 0 预期 10 实际 10
    //❌ [失败] 维度 1 预期 32 实际 16
    static void testWLContinuous2() {
        System.out.println("--- Testing WLConvContinuous ---");
        // 签名: (long inChannels, long outChannels)
        WLConvContinuous conv = new WLConvContinuous(); //16L, 32L

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }

    static void testWLContinuous() {
        System.out.println("--- Testing WLConvContinuous ---");

        long channels = 16L; // 输入 16 维
        WLConvContinuous conv = new WLConvContinuous();

        TensorOptions fOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        int numNodes = 10;
        Tensor x = randn(new long[]{numNodes, channels}, fOpt);
        Tensor edgeIndex = getEdges(numNodes, 30);

        // 执行前向传播
        Tensor out = conv.forward(x, edgeIndex);

        // 核心修正：预期维度必须等于输入维度 (16)
        GNNTester.assertShape(out, numNodes, (int)channels);
        System.out.println("✅ WLConvContinuous Passed!");
    }
    static void testResGated() {
        System.out.println("--- Testing ResGatedGraphConv ---");

        // 1. 匹配构造函数签名
        long inC = 16L;
        long outC = 32L;
        Integer edgeDim = 8; // 如果不使用边特征，可以传 null 或 0，取决于你的实现
        boolean rootWeight = true; // 是否对中心节点应用独立的权重矩阵
        boolean hasBias = true;

        ResGatedGraphConv conv = new ResGatedGraphConv(inC, outC, edgeDim, rootWeight, hasBias);

        // 2. 严格类型包装构造输入
        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));

        int numNodes = 12;
        int numEdges = 30;
        Tensor x = randn(new long[]{numNodes, inC}, floatOpt);
        Tensor edgeIndex = getEdges(numNodes, numEdges);

        // 构造边特征 [E, edgeDim]
        Tensor edgeAttr = randn(new long[]{numEdges, edgeDim.longValue()}, floatOpt);

        // 3. 执行前向传播 (注意：如果实现了带 edge_attr 的 forward)
        Tensor out = conv.forward(x, edgeIndex, edgeAttr);

        // 4. 校验输出形状
        GNNTester.assertShape(out, numNodes, (int)outC);
    }
    static void testSG() {
        System.out.println("--- Testing SGConv ---");
        // 签名: (long inChannels, long outChannels, int K, boolean hasBias)
        long inC = 16L, outC = 32L;
        int K = 2;
        SGConv conv = new SGConv(inC, outC, K);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{12, inC}, floatOpt);
        Tensor edgeIndex = getEdges(12, 40);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 12, (int)outC);
    }

    static void testFeaSt() {
        System.out.println("--- Testing FeaStConv ---");
        // 签名: (long inChannels, long outChannels, int heads, boolean hasBias)
        long inC = 16L, outC = 32L;
        int heads = 4;
        FeaStConv conv = new FeaStConv(inC, outC, heads, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, inC}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, (int)outC);
    }

    static void testLG() {
        System.out.println("--- Testing LGConv ---");
        // 签名: 通常不需要可学习权重参数，仅执行聚合
        LGConv conv = new LGConv(true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        // 特征维度在 LGConv 中保持不变
        Tensor x = randn(new long[]{15, 16}, floatOpt);
        Tensor edgeIndex = getEdges(15, 50);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 15, 16);
    }

    static void testARMA() {
        System.out.println("--- Testing ARMAConv ---");
        // 典型参数: in, out, num_stacks, num_layers, shared_weights, dropout
        ARMAConv conv = new ARMAConv(16L, 32L, 2, 1);// false, 0.1, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 20);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }
    static void testSSG2() {
        System.out.println("--- Testing SSGConv ---");
        // 签名: (long inChannels, long outChannels, double alpha, int K, boolean hasBias)
        SSGConv conv = new SSGConv(16L, 32L, 0.1, 2, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }
    static void testHypergraph() {
        System.out.println("--- Testing HypergraphConv ---");
        // 签名: (long inChannels, long outChannels, boolean useAttention, int heads, boolean concat)
        int heads = 2;
        HypergraphConv conv = new HypergraphConv(16L, 32L, true, heads, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));

        Tensor x = randn(new long[]{10, 16}, floatOpt);
        // 超图索引: [2, E], 第一行是节点, 第二行是超边
        Tensor hyperedgeIndex = randint(0, 5, new long[]{2, 20}, longOpt);

        Tensor out = conv.forward(x, hyperedgeIndex);
        // concat=true, 32 * 2 = 64
        GNNTester.assertShape(out, 10, 64);
    }

    static void testTAG() {
        System.out.println("--- Testing TAGConv ---");
        int inC = 16, outC = 32, k = 3; // k 表示聚合到 3 阶邻域
        TAGConv conv = new TAGConv(inC, outC, k);

        Tensor x = randn(new long[]{10, inC},  new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }
    static void testGatedGraph() {
        System.out.println("--- Testing GatedGraphConv ---");
        int outC = 32, numLayers = 3;
        // GatedGraphConv 通常要求输入和输出维度一致，因为它内部是循环结构
        GatedGraphConv conv = new GatedGraphConv(outC, numLayers);

        Tensor x = randn(new long[]{10, outC},new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 30);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }

    public static void testGraphConv() {
        System.out.println("--- Testing Hetero GraphConv ---");

        // 1. 定义维度
        long authorDim = 32L;
        long paperDim = 16L;
        long outDim = 64L;

        // 2. 初始化算子：源(32), 目标(16), 输出(64)
        GraphConv conv = new GraphConv(authorDim, paperDim, outDim);
        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        TensorOptions longOpt = new TensorOptions().dtype(new ScalarTypeOptional(kLong()));
        // 3. 构造数据
        // 50个作者，32维特征
        Tensor xAuthor = randn(new long[]{50, authorDim},
                floatOpt);
        // 100篇论文，16维特征
        Tensor xPaper = randn(new long[]{100, paperDim},
                floatOpt);

        // 构造 150 条边: author (0-49) -> paper (0-99)
        // 第一行是源(author), 第二行是目标(paper)
        Tensor edgeIndex = stack(new TensorVector(
                randint(0, 50, new long[]{150}, longOpt),
                randint(0, 100, new long[]{150}, longOpt)
        ), 0);

        // 模拟边权重 [150, 1]
        Tensor edgeWeight = randn(new long[]{150, 1}, floatOpt);

        // 4. 执行前向传播
        System.out.println("Running forward...");
        // 注意：这里手动调用带有 xDst 的重载
        Tensor output = conv.forward(xAuthor, xPaper, edgeIndex);

        // 5. 校验结果
        System.out.println("Output Shape: " + Arrays.toString(output.sizes().vec().get()));

        // 预期输出应为 [100, 64] (因为目标节点是 Paper)
        boolean shapeOk = output.size(0) == 100 && output.size(1) == outDim;

        if (shapeOk) {
            System.out.println("✅ GraphConv Hetero Test Passed!");
        } else {
            throw new RuntimeException("❌ Shape Mismatch!");
        }
    }
    static void testGraph() {
        System.out.println("--- Testing GraphConv ---");

        long inC = 16L;
        long outC = 32L;

        // 签名: (long inChannels, long outChannels)
        GraphConv conv = new GraphConv(inC, inC, outC);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        int numNodes = 10;
        int numEdges = 30;

        Tensor x = randn(new long[]{numNodes, inC}, floatOpt);
        Tensor edgeIndex = getEdges(numNodes, numEdges);

        // edgeWeight 是可选的，模拟权重为 1.0 的情况
        Tensor edgeWeight = ones(new long[]{numEdges}, floatOpt);

        // 签名: forward(Tensor x, Tensor edge_index, Tensor edgeWeight)
        Tensor out = conv.forward(x, edgeIndex, edgeWeight);

        // 校验维度: [10, 32]
        GNNTester.assertShape(out, numNodes, (int)outC);
        System.out.println("✅ GraphConv Passed!");
    }
    static void testPAN() {
        System.out.println("--- Testing PANConv ---");

        long inC = 16L;
        long outC = 32L;
        int filterSize = 2; // 考虑 0, 1, 2 阶路径

        // 签名: (long inChannels, long outChannels, int filterSize)
        PANConv conv = new PANConv(inC, outC, filterSize);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        int numNodes = 10;
        Tensor x = randn(new long[]{numNodes, inC}, floatOpt);
        Tensor edgeIndex = getEdges(numNodes, 25);

        // 执行前向传播
        // PANConv 内部通常会自动处理路径权重的加权求和
        Tensor out = conv.forward(x, edgeIndex);

        // 校验维度: [10, 32]
        GNNTester.assertShape(out, numNodes, (int)outC);
        System.out.println("✅ PANConv Passed!");
    }
    static void testMF() {
        System.out.println("--- Testing MFConv ---");
        // 签名: (long inChannels, long outChannels, int maxDegree, boolean hasBias)
        int maxDegree = 2;
        MFConv conv = new MFConv(16L, 32L, maxDegree, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{12, 16}, floatOpt);
        Tensor edgeIndex = getEdges(12, 30);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 12, 32);
    }

    static void testPointNet() {
        System.out.println("--- Testing PointNetConv ---");
        // 假设输入特征 16 维，坐标 3 维
        // localNN 的输入维度通常是 (inChannels + posDim)
//        SequentialImpl localNN = new SequentialImpl(new LinearImpl(16 + 3, 32));
//        SequentialImpl globalNN = new SequentialImpl(new LinearImpl(32, 32));

        SequentialImpl localNN = new SequentialImpl();
        localNN.push_back("linear1", new LinearImpl(16 + 3, 32)); //new LinearImpl(16 + 3, 32)
        SequentialImpl globalNN = new SequentialImpl();
        globalNN.push_back("linear1", new LinearImpl(32, 32)); //new LinearImpl(32, 32)

        // 签名: (Module localNN, Module globalNN, boolean addSelfLoops)
        PointNetConv conv = new PointNetConv(localNN, globalNN, true);

        TensorOptions floatOpt = new TensorOptions().dtype(new ScalarTypeOptional(kFloat()));
        Tensor x = randn(new long[]{10, 16}, floatOpt);
        Tensor pos = randn(new long[]{10, 3}, floatOpt);
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, pos, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }
//    static void testSSG() {
//        System.out.println("--- Testing SSGConv ---");
//        // 参数: in, out, alpha (残差系数), k (平滑次数)
//        SSGConv conv = new SSGConv(16, 32, 0.1f, 2);
//
//        Tensor x = randn(new long[]{10, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
//        Tensor edgeIndex = getEdges(10, 20);
//
//        Tensor out = conv.forward(x, edgeIndex);
//        GNNTester.assertShape(out, 10, 32);
//    }
    static void testEG() {
        System.out.println("--- Testing EGConv ---");

        // 1. 参数定义
        long inC = 16;
        long outC = 32;
        List<String> aggrs = Arrays.asList("mean", "max", "sum");
        int numHeads = 4;
        int numBases = 4; // 注意：通常 numBases 应该能被 outChannels 或某种逻辑整除，或者根据实现而定
        boolean hasBias = true;

        // 2. 初始化算子
        // 签名: (long inChannels, long outChannels, List<String> aggregators, int numHeads, int numBases, boolean hasBias)
        EGConv conv = new EGConv(inC, outC, aggrs, numHeads, numBases, hasBias);

        // 3. 构造测试数据
        Tensor x = randn(new long[]{12, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(12, 40);

        // 4. 前向传播
        Tensor out = conv.forward(x, edgeIndex);

        // 5. 形状校验
        GNNTester.assertShape(out, 12, (int)outC);
    }


    static void testClusterGCN() {
        System.out.println("--- Testing ClusterGCNConv ---");

        // 1. 对应构造函数参数: inC, outC, diagLambda, addSelfLoops, hasBias
        long inC = 16;
        long outC = 32;
        float diagLambda = 1.0f; // 典型的 lambda 偏移量
        boolean addSelfLoops = true;
        boolean hasBias = true;

        // 2. 初始化算子
        ClusterGCNConv conv = new ClusterGCNConv(inC, outC, diagLambda, addSelfLoops, hasBias);

        // 3. 构造测试数据
        int numNodes = 12;
        int numEdges = 30;
        Tensor x = randn(new long[]{numNodes, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(numNodes, numEdges);

        // 4. 前向传播
        // ClusterGCNConv 通常遵循标准的 (x, edge_index) 签名
        Tensor out = conv.forward(x, edgeIndex);

        // 5. 形状校验
        GNNTester.assertShape(out, numNodes, (int)outC);
    }
//    static void testClusterGCN() {
//        System.out.println("--- Testing ClusterGCNConv ---");
//        int inC = 16, outC = 32;
//        ClusterGCNConv conv = new ClusterGCNConv(inC, outC, true /* diag_lambda */);
//
//        Tensor x = randn(new long[]{10, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));
//        Tensor edgeIndex = getEdges(10, 20);
//
//        Tensor out = conv.forward(x, edgeIndex);
//        GNNTester.assertShape(out, 10, 32);
//    }

    static void testDNA() {
        System.out.println("--- Testing DNAConv ---");
        // 参数：channels=16, heads=4, groups=1, dropout=0.1, cached=false
        int channels = 16;
        int heads = 4;
        DNAConv conv = new DNAConv(channels, heads, 1,  false);

        // 关键修正：构造 3D 张量 [Nodes, Groups/Layers, Channels]
        // 假设我们模拟 1 个 group，每个节点 16 维
        Tensor x = randn(new long[]{10, 1, channels},new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 20);

        // 调用 forward
        Tensor out = conv.forward(x, edgeIndex);

        // DNAConv 的输出通常会挤压回 [Nodes, Channels] 或者保持 3D，取决于实现
        // 校验形状
        System.out.println("DNAConv output shape: " + java.util.Arrays.toString(out.sizes().vec().get()));
        GNNTester.assertShape(out, 10, channels);
    }


    static void testLE() {
        System.out.println("--- Testing LEConv ---");
        int inC = 16, outC = 32;
        LEConv conv = new LEConv(inC, outC, true /* hasBias */);

        Tensor x = randn(new long[]{10, 16},new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 25);

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }
    static void testHAN() {
        System.out.println("--- Testing HANConv ---");

        // 1. 节点输入维度字典
        Map<String, Integer> inChannelsDict = new HashMap<>();
        inChannelsDict.put("nodeA", 16);

        // 2. 节点类型和边类型元数据
        List<String> nodeTypes = new ArrayList<>(Arrays.asList("nodeA"));

        List<String[]> edgeTypes = new ArrayList<>();
        String[] rel = new String[]{"nodeA", "to", "nodeA"};
        edgeTypes.add(rel);

        // 3. 初始化算子 (inDict, outC, nodeTypes, edgeTypes, heads)
        int outChannels = 32;
        int heads = 2;
        HANConv conv = new HANConv(inChannelsDict, outChannels, nodeTypes, edgeTypes, heads);

        // 4. 准备输入数据 (xDict 和 edgeDict)
        Map<String, Tensor> xDict = new HashMap<>();
        xDict.put("nodeA", randn(new long[]{10, 16},new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));
        Map<String[], Tensor> edgeDict = new HashMap<>();
        edgeDict.put(rel, getEdges(10, 25));

        // 5. 前向传播
        Map<String, Tensor> out = conv.forward(xDict, edgeDict);

        // 6. 校验输出形状
        // 注意：HAN 输出维度通常受 heads 影响，如果内部做了 concat，则是 outChannels * heads
        // 这里假设 out.get("nodeA") 存在
        GNNTester.assertShape(out.get("nodeA"), 10, outChannels);
    }

//    static void testHAN() {
//        System.out.println("--- Testing HANConv ---");
//        // 1. 定义元路径三元组
//        List<String[]> metadata = new ArrayList<>();
//        metadata.add(new String[]{"nodeA", "to", "nodeA"});
//
//        // 2. 初始化：inChannels=16, outChannels=32, heads=2
//        HANConv conv = new HANConv(16, 32, metadata, 2, 0.1, false);
//
//        // 3. 构造节点特征
//        Map<String, Tensor> xDict = new HashMap<>();
//        xDict.put("nodeA", randn(new long[]{10, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));
//
//        // 4. 构造边索引（Key 必须匹配 metadata 中的数组引用）
//        Map<String[], Tensor> edgeDict = new HashMap<>();
//        edgeDict.put(metadata.get(0), getEdges(10, 25));
//
//        Map<String, Tensor> out = conv.forward(xDict, edgeDict);
//        GNNTester.assertShape(out.get("nodeA"), 10, 32);
//    }
//
//    static void testGPS2() {
//        System.out.println("--- Testing GPSConv ---");
//        int inC = 16, outC = 16;
//        // 内部通常需要一个消息传递算子，这里用 GINEConv (支持边特征)
//        GINEConv localConv = new GINEConv(nn.Linear(inC, outC));
//
//        // 参数：channels, local_msg_passing_layer, heads, attn_type="multihead"
//        GPSConv conv = new GPSConv(inC, localConv, 4, 0.1, "multihead", null, true, 0.1, 0.1);
//
//        Tensor x = randn(new long[]{12, inC},  new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
//        Tensor edgeIndex = getEdges(12, 30);
//
//        // GPS 通常需要 batch 索引（如果没有则传全 0）
//        Tensor batch = zeros(new long[]{12}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
//
//        // 第三个参数是 edge_attr, 第四个参数是 mask (传 empty Tensor)
//        Tensor out = conv.forward(x, edgeIndex, batch, new Tensor(), new Tensor());
//        GNNTester.assertShape(out, 12, 16);
//    }
//
////    static void testPDN() {
////        System.out.println("--- Testing PDNConv ---");
////        int inC = 16, outC = 32, edgeC = 8, hiddenC = 16;
////        PDNConv conv = new PDNConv(inC, outC, edgeC, hiddenC);
////
////        Tensor x = randn(new long[]{20, inC}, kFloat());
////        Tensor edgeIndex = getEdges(20, 50);
////        Tensor edgeAttr = randn(new long[]{50, edgeC}, kFloat());
////
////        Tensor out = conv.forward(x, edgeIndex, edgeAttr);
////        GNNTester.assertShape(out, 20, 32);
//    }
    static void testWL() {
        System.out.println("--- Testing WLConv ---");
        WLConv conv = new WLConv();

        // 注意：WL 算子的特征必须是 Long 型（离散标签/颜色）
        Tensor x = randint(0, 5, new long[]{10}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        x = x.to(kLong());
        Tensor edgeIndex = getEdges(10, 30);

        // WL 内部会返回一个更新后的特征分布
        Tensor out = conv.forward(x, edgeIndex);
        // 形状保持不变 [N]
        GNNTester.assertShape(out, 10);
    }

    static void testGATv2() {
        System.out.println("--- Testing GATv2Conv ---");

        // 匹配构造函数签名
        int inC = 16, outC = 32, heads = 2, edgeDim = 8;
        boolean concat = true;

        // 参数：inChannels, outChannels, heads, concat, negativeSlope, edgeDim, hasBias
        GATv2Conv conv = new GATv2Conv(inC, outC, heads, concat, 0.2, edgeDim, true);

        Tensor x = randn(new long[]{10, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 20);
        Tensor edgeAttr = randn(new long[]{20, edgeDim},  new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

        // 前向传播
        Tensor out = conv.forward(x, edgeIndex, edgeAttr);

        // 维度校验：32 * 2 (heads) = 64
        long expectedOut = concat ? (heads * outC) : outC;
        GNNTester.assertShape(out, 10, expectedOut);
    }
//    static void testGATv2() {
//        System.out.println("--- Testing GATv2Conv ---");
//        int inC = 16, outC = 32, heads = 3;
//        // 参数：in, out, heads, concat, dropout, edge_dim, fill_value, bias
//        GATv2Conv conv = new GATv2Conv(inC, outC, heads, true, 0.1, 0.0, true);
//
//        Tensor x = randn(new long[]{12, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
//        Tensor edgeIndex = getEdges(12, 30);
//
//        // 输出维度：32 * 3 (heads) = 96
//        Tensor out = conv.forward(x, edgeIndex);
//        GNNTester.assertShape(out, 12, 96);
//    }
    static void testHEAT() {
        System.out.println("--- Testing HEATConv ---");
        // 1. 参数定义
        int inC = 16, outC = 32, numNodes = 10, numEdges = 20;
        int numNodeTypes = 2, numEdgeTypes = 3;
        int edgeTypeEmbDim = 8, edgeDim = 4, edgeAttrEmbDim = 8, heads = 2;
        boolean concat = true;

        // 2. 初始化算子
        HEATConv conv = new HEATConv(inC, outC, numNodeTypes, numEdgeTypes,
                edgeTypeEmbDim, edgeDim, edgeAttrEmbDim, heads, concat);

        // 3. 准备节点数据
        Tensor x = randn(new long[]{numNodes, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        // 随机分配节点类型 [0, numNodeTypes-1]
        Tensor nodeType = randint(0, numNodeTypes, new long[]{numNodes},  new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        // 4. 准备边数据
        Tensor edgeIndex = getEdges(numNodes, numEdges);
        // 随机分配边类型 [0, numEdgeTypes-1]
        Tensor edgeType = randint(0, numEdgeTypes, new long[]{numEdges},  new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        // 边属性特征
        Tensor edgeAttr = randn(new long[]{numEdges, edgeDim}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

        // 5. 前向传播
        Tensor out = conv.forward(x, edgeIndex, nodeType, edgeType, edgeAttr);

        // 6. 维度校验
        // 如果 concat=true，输出维度是 heads * outChannels = 2 * 32 = 64
        long expectedDim = concat ? (heads * outC) : outC;
        GNNTester.assertShape(out, numNodes, expectedDim);
    }
//
//    static void testHEAT() {
//        System.out.println("--- Testing HEATConv ---");
//        int inC = 16, outC = 32, heads = 2, edgeC = 8;
//
//        // 定义元数据
//        List<String> nodeTypes = Arrays.asList("nodeA");
//        List<String[]> edgeTypes = new ArrayList<>();
//        String[] rel = new String[]{"nodeA", "to", "nodeA"};
//        edgeTypes.add(rel);
//
//        // 初始化：in_channels, out_channels, node_types, edge_types, edge_dim, heads
//        HEATConv conv = new HEATConv(inC, outC, nodeTypes, edgeTypes, edgeC, heads, 0.1, true);
//
//        Map<String, Tensor> xDict = new HashMap<>();
//        xDict.put("nodeA", randn(new long[]{10, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));
//
//        Map<String[], Tensor> edgeIndexDict = new HashMap<>();
//        edgeIndexDict.put(rel, getEdges(10, 20));
//
//        // HEAT 需要边特征字典 Map<String[], Tensor>
//        Map<String[], Tensor> edgeAttrDict = new HashMap<>();
//        edgeAttrDict.put(rel, randn(new long[]{20, edgeC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));
//
//        Map<String, Tensor> out = conv.forward(xDict, edgeIndexDict, edgeAttrDict);
//        GNNTester.assertShape(out.get("nodeA"), 10, 32);
//    }
    static void testMixHop() {
        System.out.println("--- Testing MixHopConv ---");
        int inC = 16, outC = 32;
        List powers =new ArrayList<>(Arrays.asList(0, 1, 2));
        MixHopConv conv = new MixHopConv(inC, outC, powers, true);

        Tensor x = randn(new long[]{10, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = randint(0, 10, new long[]{2, 20},new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 96); // 32 * 3 = 96
    }

    static void testPDN() {
        System.out.println("--- Testing PDNConv ---");
        PDNConv conv = new PDNConv(16, 32, 8, 16, true, true);
        Tensor x = randn(new long[]{10, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = randint(0, 10, new long[]{2, 15},new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        Tensor edgeAttr = randn(new long[]{15, 8},new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

        Tensor out = conv.forward(x, edgeIndex, edgeAttr);
        GNNTester.assertShape(out, 10, 32);
    }

    static void testGPS() {
        System.out.println("--- Testing GPSConv ---");
        GCNConv local = new GCNConv(32, 32);
        GPSConv conv = new GPSConv(32, local, 4, 0.1f);

        Tensor x = randn(new long[]{15, 32},new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = randint(0, 15, new long[]{2, 30}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
        Tensor batch = zeros(new long[]{15}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

        Tensor out = conv.forward(x, edgeIndex, batch);
        GNNTester.assertShape(out, 15, 32);
    }

    static void testAntiSymmetric() {
        System.out.println("--- Testing AntiSymmetricConv ---");
        AntiSymmetricConv conv = new AntiSymmetricConv(16, null, 2, 0.1f, 0.1f, true);
        Tensor x = randn(new long[]{8, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = randint(0, 8, new long[]{2, 12}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 8, 16);
    }
    static void testSuperGAT2() {
        System.out.println("--- Testing SuperGATConv ---");
        int inC = 16, outC = 32, heads = 2;
        SuperGATConv conv = new SuperGATConv(inC, outC, heads, true, "MX");

        Tensor x = randn(new long[]{15, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(15, 40);

        // SuperGAT 不需要额外的 edge_attr
        Tensor out = conv.forward(x, edgeIndex, new Tensor());
        // 因为 heads=2 且默认 concat=true，输出维度为 32*2=64
        GNNTester.assertShape(out, 15, 64);

//        // 测试自监督 Loss 提取
//        Tensor attnLoss = conv.get_attention_loss();
//        System.out.println("SuperGAT Attention Loss: " + attnLoss.item_float());
    }

    static void testDirGNN() {
        System.out.println("--- Testing DirGNNConv ---");
        // 内部使用 SAGEConv
        SAGEConvV2 inner = new SAGEConvV2(16, 16, 32);
        // alpha=0.5 代表入边出边等权
        DirGNNConv conv = new DirGNNConv(inner, 0.5f, true, 16, 32);

        Tensor x = randn(new long[]{10, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 25);

        // 关键修正：确保内部调用的 forward 签名匹配
        Tensor out = conv.forward(x, edgeIndex);
        GNNTester.assertShape(out, 10, 32);
    }
    static void testGeneral() {
        System.out.println("--- Testing GeneralConv ---");
        int inC = 16, outC = 32, edgeC = 8;
        // 参数：in, out, in_edge, heads, concat, aggr, l2_norm, use_edge, has_bias
        GeneralConv conv = new GeneralConv(inC, outC, edgeC, 1, false, "add", true, true, true);

        Tensor x = randn(new long[]{12, inC},new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(12, 30);
        Tensor edgeAttr = randn(new long[]{30, edgeC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));

        Tensor out = conv.forward(x, edgeIndex, edgeAttr);
        GNNTester.assertShape(out, 12, 32);
    }
    static void testFiLM() {
        System.out.println("--- Testing FiLMConv ---");
        int inC = 16, outC = 32, numRelations = 2;
        FiLMConv conv = new FiLMConv(inC, outC, numRelations, null);

        Tensor x = randn(new long[]{10, inC}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor edgeIndex = getEdges(10, 20);
        // 随机生成 0 或 1 的关系索引
        Tensor edgeType = randint(0, numRelations, new long[]{20}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())));

        Tensor out = conv.forward(x, edgeIndex, edgeType);
        GNNTester.assertShape(out, 10, 32);
    }


    /**
     * 生成随机的 edge_index 张量
     * @param n 节点数量 (用于限制索引最大值)
     * @param e 边数量
     * @return 形状为 [2, e] 的 LongTensor
     */
    static Tensor getEdges(int n, int e) {
        // randint 的参数依次为: min, max, size, options
        // 注意: max 是开区间，所以用 n 刚好能取到 0 到 n-1
        return randint(0, n, new long[]{2, e},
                new TensorOptions().dtype(new ScalarTypeOptional(kLong())));
    }

//    static void testHGT2() {
//        System.out.println("--- Testing HGTConv ---");
//        java.util.Map<String, Integer> inDict = new java.util.HashMap<>();
//        inDict.put("nodeA", 16);
//        List nodeTypes = new ArrayList<>(Arrays.asList("nodeA"));// java.util.List.of("nodeA");
//        List edgeTypes = new ArrayList<>(Arrays.asList("nodeA", "to", "nodeA"));// java.util.List.of(new String[]{"nodeA", "to", "nodeA"});
//
//        HGTConv conv = new HGTConv(inDict, 32, nodeTypes, edgeTypes, 4);
//
//        java.util.Map<String, Tensor> xDict = new java.util.HashMap<>();
//        xDict.put("nodeA", randn(new long[]{5, 16}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));
//
//        java.util.Map<String[], Tensor> edgeDict = new java.util.HashMap<>();
//        edgeDict.put(edgeTypes.get(0), randint(0, 5, new long[]{2, 10},new TensorOptions().dtype(new ScalarTypeOptional(kLong()))));
//
//        List out = conv.forward(xDict, edgeDict);
//        GNNTester.assertShape(out.get("nodeA"), 5, 32);
//    }
    static void testHGT2() {
        System.out.println("--- Testing HGTConv ---");

        // 1. 输入维度字典
        Map<String, Integer> inDict = new HashMap<>();
        inDict.put("nodeA", 16);

        // 2. 节点类型列表
        List<String> nodeTypes = new ArrayList<>(Arrays.asList("nodeA"));

        // 3. 边类型三元组列表 (关键修正：HGT 期待 List<String[]>)
        List<String[]> edgeTypes = new ArrayList<>();
        String[] rel = new String[]{"nodeA", "to", "nodeA"};
        edgeTypes.add(rel);

        // 4. 初始化算子
        HGTConv conv = new HGTConv(inDict, 32, nodeTypes, edgeTypes, 4);

        // 5. 准备节点特征字典
        Map<String, Tensor> xDict = new HashMap<>();
        xDict.put("nodeA", randn(new long[]{5, 16},new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))));

        // 6. 准备边索引字典 (关键修正：Key 必须与 edgeTypes 中的数组对象一致)
        Map<String[], Tensor> edgeDict = new HashMap<>();
        edgeDict.put(rel, randint(0, 5, new long[]{2, 10},new TensorOptions().dtype(new ScalarTypeOptional(kLong()))));

        // 7. 前向传播 (关键修正：返回值是 Map<String, Tensor>)
        Map<String, Tensor> out = conv.forward(xDict, edgeDict);

        // 8. 校验结果
        GNNTester.assertShape(out.get("nodeA"), 5, 32);
    }
}