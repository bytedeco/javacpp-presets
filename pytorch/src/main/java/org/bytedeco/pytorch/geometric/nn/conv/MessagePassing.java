package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.demo.layer.SimpleGAT;
import org.bytedeco.pytorch.geometric.utils.Scatter;

import static org.bytedeco.pytorch.global.torch.zeros;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 模仿 PyG 的 org.bytedeco.pytorch.geometric.nn.conv.MessagePassing 基类
 */
public abstract class MessagePassing extends Module implements AutoCloseable {

    protected String aggr; // "add", "mean", "max"
    protected String flow; // "source_to_target" (default)
    protected final TensorVector tensors = new TensorVector();
//    public MessagePassing() {
//        this("add", "source_to_target");
//    }

    public MessagePassing(String aggr, String flow) {
        super();
        // 校验聚合方式合法性
        if (!aggr.equals("add") && !aggr.equals("mean") && !aggr.equals("max")) {
            throw new IllegalArgumentException("不支持的聚合方式: " + aggr + "，仅支持 add/mean/max");
        }
        // 校验流向合法性
        if (!flow.equals("source_to_target") && !flow.equals("target_to_source")) {
            throw new IllegalArgumentException("不支持的流向: " + flow + "，仅支持 source_to_target/target_to_source");
        }
        this.aggr = aggr;
        this.flow = flow;
    }
    
    public MessagePassing(String aggr) {
        super();
        this.aggr = aggr;
        this.flow = "source_to_target";
    }

    public MessagePassing(Pointer p) {
        
        super(p);
        this.aggr = "add";
        this.flow = "source_to_target";
        
    }

    public MessagePassing() {
        this.aggr = "add";
        this.flow = "source_to_target";
    }

    public abstract Tensor forward(Tensor x, Tensor edge_index);

    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        // 默认行为：如果子类没实现带 edge_attr 的，就退化调用基础版本
//        return forward(x, edge_index);
        throw new UnsupportedOperationException(this.getClass().getName() + " requires edge_attr");
    }
    

    public final Tensor forward(Tensor[] args) {
        if (args.length == 2) return forward(args[0], args[1]);
        if (args.length == 3) return forward(args[0], args[1], args[2]);
        throw new IllegalArgumentException("Unsupported number of arguments for GNN forward");
    }
//    forward(*args: Any, **kwargs: Any)→ Any
    /**
     * 核心传播方法
     * @param edge_index [2, E]
     * @param x [N, F] 节点特征
     *          propagate(edge_index: Union[Tensor, SparseTensor], size: Optional[Tuple[int, int]] = None, **kwargs: Any)→ Tensor
     */
    public Tensor propagate(Tensor edge_index, Tensor x) {
//        return propagate(edge_index, x, null);
        return propagate(edge_index, x, new long[]{x.size(0), x.size(0)});
    }


    /**
     * 核心实现：支持二部图的传播
     *
     * @param edgeIndex [2, E]
     * @param x         源节点特征 [N_src, Dim]
     * @param size      {N_src, N_dst}
     */
    public Tensor propagate(Tensor edgeIndex, Tensor x, long[] size) {
        // 调试打印：确认进入 propagate 的 Tensor 状态
        // System.out.println("Propagate -> x shape: " + Arrays.toString(x.sizes().vec().get()) + ", size: " + Arrays.toString(size));

        // 获取 row (source) 索引
        Tensor index_j = edgeIndex.select(0, 0);

        // 【核心防御检查】
        long numNodesInX = x.size(0);
        long maxIdxInEdge = index_j.max().item_long();

        if (maxIdxInEdge >= numNodesInX) {
            throw new RuntimeException(String.format(
                    "IndexSelect 越界！试图从长度为 %d 的 Tensor 中通过 index_select 提取索引 %d",
                    numNodesInX, maxIdxInEdge));
        }

        // 执行 index_select
        // 如果这里依然报错 "index out of range"，说明 x 根本不是你以为的那个 x
        Tensor x_j = x.index_select(0, index_j);

        // 后续逻辑...
        Tensor msg = message(x_j);

        Tensor col = edgeIndex.select(0, 1); // target
        Tensor out = torch.zeros(new long[]{size[1], msg.size(1)}, msg.options());
        return out.index_add_(0, col, msg);
    }

    public Tensor propagate4(Tensor edgeIndex, Tensor x, long[] size) {
        // 1. 解析 Source 和 Target 索引
        Tensor row = edgeIndex.select(0, 0); // 源节点索引 (index_j)
        Tensor col = edgeIndex.select(0, 1); // 目标节点索引 (index_i)

        // 2. Lift: 将源节点特征映射到每一条边上
        // [N_src, Dim] -> [E, Dim]
        // 这一步解决了你看到的 "size 32 must match 150" 的问题
        Tensor x_j = x.index_select(0, row);

        // 3. 构建消息 (调用子类实现的 message)
        // 此时 x_j 的 shape 是 [150, Dim]，可以安全地与边特征运算
        Tensor msg = message(x_j);

        // 4. 聚合 (Aggregate)
        // 将消息 [150, Dim] 根据 col 聚合到目标节点 [N_dst, Dim]
        // index_reduce 是 PyTorch 处理 GNN 聚合的高效算子
        Tensor out = zeros(new long[]{size[1], msg.size(1)}, msg.options());

        // 使用 index_add 或 scatter 将消息累加到目标节点
        // "sum" 聚合示例：
        out.index_add_(0, col, msg);

        return out;
    }


    public Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_attr) {
        long numNodes = x.size(0);
        Tensor sourceIdx = edge_index.select(0, 0).to(torch.kLong()); ;
        Tensor targetIdx = edge_index.select(0, 1).to(torch.kLong()); ;

        Tensor x_j = x.index_select(0, sourceIdx);
        Tensor x_i = x.index_select(0, targetIdx);

        // 核心修正：显式传入 edge_index 和额外的 edge_attr
        Tensor out = aggregate(message(x_j, x_i, edge_index, edge_attr, numNodes), targetIdx, numNodes);
        return update(out, x);
    }

    public Tensor propagate(Tensor edge_index, Tensor x, Tensor pos, Tensor deltaPos) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 1. 手动执行 message 逻辑 (对应四个参数)
        Tensor x_j = x.index_select(0, row);
        Tensor p_j = pos.index_select(0, row);
        Tensor dp_i = deltaPos.index_select(0, col);

        // 这里的逻辑就是你原本想写在 message 里的内容
        Tensor msg = message(x_j, p_j, dp_i);

        // 2. 手动执行 aggregate 逻辑
        Tensor out = zeros(new long[]{x.size(0), msg.size(1)}, x.options());
        // 使用 scatter_add 模拟 "add" 聚合
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg);

        // 3. 手动执行 update 逻辑
        return update(out, x);
    }

    /**
     * 二部图传播（源/目标节点分离）+ 边权重支持
     */
    public Tensor propagate(Tensor edge_index, Tensor xSrc, Tensor xDst, Tensor edge_weight, long numNodes) {
        // 1. 提取边索引
        Tensor row = edge_index.select(0, 0); // 源节点索引 [E]
        Tensor col = edge_index.select(0, 1); // 目标节点索引 [E]

        // 2. 提取源节点特征 x_j [E, *]
        Tensor x_j = xSrc.index_select(0, row);

        // 3. 提取目标节点特征 x_i [E, *]
        Tensor x_i = xDst.index_select(0, col);

        // 4. 执行 message 逻辑（传入边权重）
        Tensor msg = message(x_j, x_i, edge_index, edge_weight, numNodes);

        // 5. 执行聚合
        Tensor out = aggregate(msg, col, numNodes);

        // 6. 执行更新
        out = update(out, xDst);

        // 释放临时张量
        row.close();
        col.close();
        x_j.close();
        x_i.close();
        msg.close();

        return out;
    }
    
    /**
     * 核心逻辑：支持二部图的传播
     * @param edge_index 边索引 [2, E]
     * @param xSrc       源节点特征 (例如 50 个作者)
     * @param xDst       目标节点特征 (例如 100 篇论文)
     * @param size       目标节点总数 (100)
     */
    public Tensor propagate(Tensor edge_index, Tensor xSrc, Tensor xDst, long size) {
        // 1. 提取索引
        Tensor row = edge_index.select(0, 0); // Source indices
        Tensor col = edge_index.select(0, 1); // Destination indices

        // 2. 准备消息：从源特征中 index_select
        // x_j 形状为 [E, channels]
        Tensor x_j = xSrc.index_select(0, row);

        // 3. 准备目标引用（如果需要，如 SAGEConv 的 x_i）
        Tensor x_i = xDst.index_select(0, col);

        // 4. 执行 message 逻辑 (由子类重写)
        Tensor msg = message(x_j, x_i, edge_index);

        // 5. 执行聚合：关键点在于传入 size (100)
        // 这样创建的聚合张量形状就是 [100, channels]
        Tensor out = aggregate(msg, col, size);

        // 6. 执行更新
        return update(out, xDst);
    }

//    protected Tensor propagate(Tensor edgeIndex, Tensor x) {
//        return propagate(edgeIndex, x, null, new long[]{x.size(0), x.size(0)});
//    }
//
//    /**
//     * 带边特征的消息传递（默认二部图大小为 [N, N]）
//     */
//    protected Tensor propagate(Tensor edgeIndex, Tensor x, Tensor edgeAttr) {
//        return propagate(edgeIndex, x, edgeAttr, new long[]{x.size(0), x.size(0)});
//    }

//    public abstract Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr);


    public Tensor propagate2(Tensor edge_index, Tensor x, Tensor edge_attr) {
        long numNodes = x.size(0);

        // =========================================================================
        // 修正点 1: 使用 .select() 而不是 .index_select()
        // .select(0, 0) 表示在第 0 维取第 0 个元素，结果会自动降维
        // [2, E] -> [E] (这才是 index_select 需要的 1D 向量格式)
        // =========================================================================
        Tensor sourceIdx = edge_index.select(0, 0).to(torch.kLong()); // 对应 source (j)
        Tensor targetIdx = edge_index.select(0, 1).to(torch.kLong());// 对应 target (i)

        // 2. 收集特征 (Lift)
        // x_j: 源节点特征 [E, F]
        // x.index_select(0, idx) 要求 idx 必须是 1D LongTensor
        Tensor x_j = x.index_select(0, sourceIdx);

        // x_i: 目标节点特征 [E, F]
        Tensor x_i = x.index_select(0, targetIdx);

        // 3. 生成消息 (Message)
        Tensor msg = message(x_j, x_i, edge_attr);

        // 4. 聚合 (Aggregate) -> [N, F]
        Tensor out = aggregate(msg, targetIdx, numNodes);

        // 5. 更新 (Update)
        return update(out, x);
    }


    // 子类需实现的钩子
    protected Tensor message(Tensor x_j) {
        return x_j; // 默认恒等映射
    }

    /**
     * 构造消息，默认直接返回 x_j (如同 GCN) 
     * 子类可以重写此方法 (如 org.bytedeco.pytorch.geometric.nn.model.GAT 加入 Attention)
     * message(x_j: Tensor)→ Tensor
     */
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_attr) {
        return (edge_attr != null) ? x_j.mul(edge_attr) : x_j;
    }

    public abstract Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes);

    /**
     * 生成消息（子类必须实现）
     * @param x_j 源节点特征 [E, F]
     * @param x_i 目标节点特征 [E, F]
     * @param edgeAttr 边特征 [E, F_e]（可为 null）
     * @param edgeIndex 边索引 [2, E]
     * @return 边消息 [E, F_out]
     */
    protected Tensor message(Tensor x_j, Tensor x_i, Tensor edgeIndex, Tensor edgeAttr){
       return message( x_j,  x_i,  edgeIndex,  edgeAttr,  x_j.size(0));
    }

//    public Tensor propagate2(Tensor edge_index, Tensor x) {
//        long numNodes = x.size(0);
//
//        // 1. 区分源节点和目标节点索引
//        // edge_index[0] 是 source (j), edge_index[1] 是 target (i)
//        Tensor sourceIdx = edge_index.index_select(0, torch.tensor(0)); // row 0
//        Tensor targetIdx = edge_index.index_select(0, torch.tensor(1)); // row 1
//
//        // 2. 收集特征 (Lift)
//        // x_j: 源节点特征 [E, F]
//        Tensor x_j = x.index_select(0, sourceIdx);
//        // x_i: 目标节点特征 [E, F] (有些层如 org.bytedeco.pytorch.geometric.nn.model.GAT 需要这个)
//        Tensor x_i = x.index_select(0, targetIdx);
//
//        // 3. 生成消息 (Message)
//        Tensor msg = message(x_j, x_i, edge_index);
//
//        // 4. 聚合 (Aggregate) -> [N, F]
//        Tensor out = aggregate(msg, targetIdx, numNodes);
//
//        // 5. 更新 (Update)
//        return update(out, x);
//    }


//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index) {
//        return x_j;
//    }

    /**
     * 聚合消息（默认支持 add/mean/max）
     * @param inputs 边消息 [E, F_out]
     * @param index 目标节点索引 [E]
     * @param dimSize 目标节点数量
     * @return 聚合后的特征 [dimSize, F_out]
     * aggregate(inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None)→ Tensor
     */
    public Tensor aggregate(Tensor inputs, Tensor index, long dimSize) {
        return Scatter.scatter(inputs, index, dimSize, this.aggr);
    }
    
    /**
     * 更新节点嵌入，默认直接返回聚合结果 ok
     * 更新节点特征（默认直接返回聚合结果）
     * @param inputs 聚合后的特征 [N_dst, F_out]
     * @param x 原始节点特征 [N, F]
     * @return 更新后的特征 [N_dst, F_out]
     * update(inputs: Tensor)→ Tensor
     */
    public Tensor update(Tensor inputs, Tensor x) {
        return inputs;
    }

    public String getAggr() {
        return aggr;
    }

    public String getFlow() {
        return flow;
    }


//    protected abstract void reset_parameters();
}