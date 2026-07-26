package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import static org.bytedeco.pytorch.global.torch.nan_to_num;
import static org.bytedeco.pytorch.global.torch.zeros;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
/**
 * org.bytedeco.pytorch.geometric.aggr.PatchTransformerAggregation
 * 将邻居视为 Patch，应用 Transformer Encoder 进行交互，然后求和。
 */
public class PatchTransformerAggregation extends Aggregation {
    private TransformerEncoderImpl transformerEncoder;
    private LinearImpl linOut; // 可选：输出投影

    public PatchTransformerAggregation(long channels, long numHeads, long numLayers) {
        super();
        // 1. 配置 Layer
        TransformerEncoderLayerOptions layerOpts = new TransformerEncoderLayerOptions(channels, numHeads);
        layerOpts.dim_feedforward().put(channels * 4);
        // 强制不依赖 batch_first，我们手动 transpose

        TransformerEncoderLayerImpl encoderLayer = new TransformerEncoderLayerImpl(layerOpts);

        // 2. 配置 Encoder
        TransformerEncoderOptions encoderOpts = new TransformerEncoderOptions(layerOpts, numLayers);
        this.transformerEncoder = new TransformerEncoderImpl(encoderOpts);

        register_module("transformerEncoder", transformerEncoder);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 稀疏转稠密
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, 0.0f);
        Tensor denseX = denseData[0]; // [N, MaxDeg, F]
        Tensor maskBool = denseData[1]; // [N, MaxDeg]

        long maxDeg = denseX.size(1);
        if (maxDeg == 0) {
            return zeros(new long[]{dimSize, x.size(1)}, x.options());
        }

        // 2. 构造 Padding Mask
        Tensor paddingMask = maskBool.logical_not();

        // 3. 维度转置 [B, S, F] -> [S, B, F]
        Tensor src = denseX.transpose(0, 1).contiguous();

        // 4. Transformer Forward
        // 对于全 Padding 的行，这里极易产生 NaN
        Tensor transformed = transformerEncoder.forward(src, null, paddingMask);

        // 5. 维度转回 [B, S, F]
        transformed = transformed.transpose(0, 1).contiguous();

        // 6. --- 核心修复：处理 NaN ---
        // 将产生的 NaN 替换为 0，这样后续 mul(0) 才能得到 0
        transformed = nan_to_num(transformed, new DoubleOptional(0.0), new DoubleOptional(0.0), new DoubleOptional(0.0));

        // 7. Masking & Aggregation (Sum)
        Tensor validMask = maskBool.unsqueeze(2).expand_as(transformed).to(x.dtype());

        // 执行掩码过滤并求和
        Tensor sum = transformed.multiply(validMask).sum(new long[]{1}, false, new ScalarTypeOptional());

        return sum;
    }

    //    @Override 
    public Tensor forward3(Tensor x, Tensor index, long dimSize) {
        // 1. 稀疏转稠密
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, 0.0f);
        Tensor denseX = denseData[0]; // [N, MaxDeg, F]
        Tensor maskBool = denseData[1]; // [N, MaxDeg]

        long maxDeg = denseX.size(1);
        if (maxDeg == 0) {
            return zeros(new long[]{dimSize, x.size(1)}, x.options());
        }

        // 2. 构造 Padding Mask
        Tensor paddingMask = maskBool.logical_not(); // [N, MaxDeg]

        // 3. 维度转置 (核心修复)
        // 从 [Batch, Seq, Feat] 转为 [Seq, Batch, Feat]
        Tensor src = denseX.transpose(0, 1).contiguous();

        // 4. Transformer Forward
        // 注意：src_key_padding_mask 始终期望 [Batch, Seq]
        Tensor transformed = transformerEncoder.forward(src, null, paddingMask);

        // 5. 维度转回 [Batch, Seq, Feat]
        transformed = transformed.transpose(0, 1).contiguous();

        // 6. Masking & Aggregation (Sum)
        Tensor validMask = maskBool.unsqueeze(2).expand_as(transformed).to(torch.ScalarType.Float);
        Tensor sum = transformed.mul(validMask).sum(new long[]{1}, false, new ScalarTypeOptional());

        return sum;
    }

    //    public PatchTransformerAggregation(long channels, long numHeads, long numLayers) {
//
//        // 1. 定义 Encoder Layer
//        TransformerEncoderLayerOptions layerOpts = new TransformerEncoderLayerOptions(channels, numHeads);
//        layerOpts.dim_feedforward().put(channels * 2);

    /// /        layerOpts.batch_first().put(true); // 关键: 输入为 [//Batch, Seq, Feat]
//
//        TransformerEncoderLayerImpl encoderLayer = new TransformerEncoderLayerImpl(layerOpts);
//
//        // 2. 定义 Encoder (堆叠多层)
//        TransformerEncoderOptions encoderOpts = new TransformerEncoderOptions(layerOpts,numLayers);
//        encoderOpts.num_layers().put(numLayers);
//        this.transformerEncoder = new TransformerEncoderImpl(encoderOpts); //encoderLayer
//
//        register_module("transformerEncoder", transformerEncoder);
//    }

//    @Override
    public Tensor forward2(Tensor x, Tensor index, long dimSize) {
        // 1. 稀疏转稠密
        // 填充 0.0
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, 0.0f);
        Tensor denseX = denseData[0]; // [N, MaxDeg, F]
        Tensor maskBool = denseData[1]; // [N, MaxDeg], True=Valid

        long maxDeg = denseX.size(1);
        if (maxDeg == 0) {
            return zeros(new long[]{dimSize, x.size(1)}, x.options());
        }

        // 2. 构造 Padding Mask
        // PyTorch Transformer 的 src_key_padding_mask 定义：
        // True 表示 Padding (被忽略)，False 表示 Valid (保留)。
        // 我们的 maskBool 是 True=Valid。
        // 所以我们需要取反: ~maskBool
        Tensor paddingMask = maskBool.logical_not();

        // 3. Transformer Forward
        // denseX: [N, MaxDeg, F]
        // mask: [N, MaxDeg]
        // 注意：forward 签名通常是 (src, mask, src_key_padding_mask)
        // src_mask 用于 causal masking (这里不需要，填 null)
        // src_key_padding_mask 用于 padding masking

        Tensor transformed = transformerEncoder.forward(denseX, null, paddingMask);

        // 4. org.bytedeco.pytorch.geometric.aggr.Aggregation (Sum or Mean)
        // 因为 transformed 包含了 padding 位置的脏数据 (Transformer可能会在padding位输出非0值)
        // 所以必须先 Mask 再 Sum

        // maskBool: [N, MaxDeg] -> [N, MaxDeg, 1] -> [N, MaxDeg, F]
        Tensor validMask = maskBool.unsqueeze(2).expand_as(transformed).to(torch.ScalarType.Float);

        // Sum Valid Tokens
        Tensor sum = transformed.mul(validMask).sum(new long[]{1}, false, new ScalarTypeOptional()); // Sum over MaxDeg dim

        // 如果需要 Mean:
        // Tensor counts = denseData[2].clamp_min(1).unsqueeze(1);
        // return sum.div(counts);

        return sum;
    }
}