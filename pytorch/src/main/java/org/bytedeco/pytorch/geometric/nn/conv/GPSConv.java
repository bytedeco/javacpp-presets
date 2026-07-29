package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.GPSConv
 * 混合了局部消息传递 (MPNN) 和全局注意力 (Transformer) 的强大算子。
 */
public class GPSConv extends Module {
    private MessagePassing localConv; // 局部 MPNN 算子 (如 GINEConv, GatedGraphConv)
    private MultiheadAttentionImpl globalAttn; // 全局注意力 (Strictly MultiheadAttentionImpl)

    // 融合层与后处理 (严格使用 LinearImpl)
    private LinearImpl linOut;
    private LinearImpl linMlp1, linMlp2; // 用于特征增强的 MLP
    private LayerNormImpl norm1, norm2, norm3;

    public GPSConv(int channels, MessagePassing localConv, int heads, float dropout) {
        super();
        this.localConv = localConv;

        // 1. 局部路径注册
        if (localConv != null) {
            register_module("local_conv", localConv);
        }

        // 2. 全局路径注册: 严格使用 MultiheadAttentionImpl
        // 注意：Global Attention 作用于所有节点
        this.globalAttn = new MultiheadAttentionImpl(channels, heads);
        this.globalAttn.options().dropout().put(dropout);
        register_module("global_attn", globalAttn);

        // 3. 特征融合与映射 (LinearImpl)
        this.linOut = new LinearImpl(channels, channels);
        this.linMlp1 = new LinearImpl(channels, channels * 2);
        this.linMlp2 = new LinearImpl(channels * 2, channels);

        register_module("lin_out", linOut);
        register_module("lin_mlp1", linMlp1);
        register_module("lin_mlp2", linMlp2);

        // 4. 归一化层注册
        LayerNormOptions options = new LayerNormOptions(new LongPointer(channels));
        options.normalized_shape().put(channels);
        this.norm1 = new LayerNormImpl(options);
        this.norm2 = new LayerNormImpl(options);
        this.norm3 = new LayerNormImpl(options);

        register_module("norm1", norm1);
        register_module("norm2", norm2);
        register_module("norm3", norm3);
    }

    public Tensor forward(Tensor x, Tensor edge_index, Tensor batch) {
        // --- 1. 局部路径 (Local MPNN) ---
        Tensor hLocal = x;
        if (localConv != null) {
//            Tensor hLocal  = null;
            if (localConv instanceof GCNConv) {
                hLocal = ((GCNConv) localConv).forward(x, edge_index);
            }else {
                hLocal = localConv.asSequential().forward(x, edge_index);
            }
//            = localConv.asSequential().forward(x, edge_index);
            hLocal = torch.dropout(hLocal, 0.1, is_training());
        }
        Tensor undefined = new Tensor();

        // --- 2. 全局路径 (Global Attention) ---
        // 注意：Transformer 需要输入形状为 [Seq_len, Batch, Channels]
        // 这里需要处理 batch 向量，将节点转换为序列
        Tensor hGlobal = x.unsqueeze(1); // 简化演示，实际需按 batch 拆分
        hGlobal = globalAttn.forward(hGlobal, hGlobal, hGlobal,undefined, // key_padding_mask (对应 torch::Tensor{})
                false,       // need_weights=false (我们不需要注意力矩阵，只要特征)
                undefined,// attn_mask
                true).get0();
        hGlobal = hGlobal.squeeze(1);

        // --- 3. 路径融合 (Residual + Norm) ---
        Tensor h = x.add(hLocal).add(hGlobal);
        h = norm1.forward(h);

        // --- 4. MLP 后处理 (Feed Forward) ---
        Tensor h_post = linMlp1.forward(h);
        h_post = torch.relu(h_post);
        h_post = linMlp2.forward(h_post);
        h_post = torch.dropout(h_post, 0.1, is_training());

        return norm2.forward(h.add(h_post));
    }
}