package org.bytedeco.pytorch.geometric.utils;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingFromPretrainedOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;

public class EmbeddingUtils {

    /**
     * 手动实现 Embedding::from_pretrained
     * * @param embeddings 预训练的权重 Tensor [num_embeddings, embedding_dim]
     * @param freeze 是否冻结权重
     * @param paddingIdx 可选的填充索引
     * @return 初始化的 Embedding 实例
     */
    public static EmbeddingImpl from_pretrained(Tensor embeddings, boolean freeze, Long paddingIdx,double max_norm,double norm_type , boolean sparse, boolean scale_grad_by_freq) {
        // 1. 获取维数信息
        long numEmbeddings = embeddings.size(0);
        long embeddingDim = embeddings.size(1);

        // 2. 构造 EmbeddingOptions
        EmbeddingOptions options = new EmbeddingOptions(numEmbeddings, embeddingDim);
        
        if (paddingIdx != null) {
            options.padding_idx().put(new LongOptional(paddingIdx));
        }
        options.scale_grad_by_freq().put(scale_grad_by_freq);
        options.sparse().put(sparse);
        options.norm_type().put(norm_type);
        options.max_norm().put(max_norm);
        // 3. 创建 Embedding 实例 (这会触发默认的初始化)
        EmbeddingImpl embedding = new EmbeddingImpl(options);

        // 4. 关键：使用预训练权重覆盖默认权重
        // 注意：embedding.ptr() 指向 EmbeddingImpl
        embedding.weight(embeddings);

        // 5. 处理冻结逻辑
        if (freeze) {
            embeddings.requires_grad_(false);
        }

        return embedding;
    }
    public static EmbeddingImpl from_pretrained(Tensor embeddings, boolean freeze, Long paddingIdx, boolean sparse, boolean scale_grad_by_freq) {
        // 1. 获取维数信息
        long numEmbeddings = embeddings.size(0);
        long embeddingDim = embeddings.size(1);

        // 2. 构造 EmbeddingOptions
        EmbeddingOptions options = new EmbeddingOptions(numEmbeddings, embeddingDim);
        if (paddingIdx != null) {
            options.padding_idx().put(new LongOptional(paddingIdx));
        }
        options.scale_grad_by_freq().put(scale_grad_by_freq);
        options.sparse().put(sparse);
        if (options.max_norm().has_value()) {
            options.max_norm().put(options.max_norm());
        }

        // 处理 norm_type (DoublePointer)
        if (!options.norm_type().isNull()) {
            options.norm_type().put(options.norm_type().get());
        }
//
//        // 处理 scale_grad_by_freq (BoolPointer)
//        if (!options.scale_grad_by_freq().isNull()) {
//            options.scale_grad_by_freq().put(scale_grad_by_freq);
//        }
//
//        // 处理 sparse (BoolPointer)
//        if (!options.sparse().isNull()) {
//            options.sparse().put(sparse);
//        }

        // 3. 创建 Embedding 实例 (这会触发默认的初始化)
        EmbeddingImpl embedding = new EmbeddingImpl(options);

        // 4. 关键：使用预训练权重覆盖默认权重
        // 注意：embedding.ptr() 指向 EmbeddingImpl
        embedding.weight(embeddings);

        // 5. 处理冻结逻辑
        if (freeze) {
            embeddings.requires_grad_(false);
        }

        return embedding;
    }
    public static EmbeddingImpl from_pretrained(Tensor embeddings, boolean freeze, Long paddingIdx) {
        // 1. 获取维数信息
        long numEmbeddings = embeddings.size(0);
        long embeddingDim = embeddings.size(1);

        // 2. 构造 EmbeddingOptions
        EmbeddingOptions options = new EmbeddingOptions(numEmbeddings, embeddingDim);
        if (paddingIdx != null) {
            options.padding_idx().put(new LongOptional(paddingIdx));
        }

        // 3. 创建 Embedding 实例 (这会触发默认的初始化)
        EmbeddingImpl embedding = new EmbeddingImpl(options);

        // 4. 关键：使用预训练权重覆盖默认权重
        // 注意：embedding.ptr() 指向 EmbeddingImpl
        embedding.weight(embeddings);

        // 5. 处理冻结逻辑
        if (freeze) {
            embeddings.requires_grad_(false);
        }

        return embedding;
    }

    public static EmbeddingImpl from_pretrained(Tensor embeddings, EmbeddingFromPretrainedOptions options) {
        // 1. 基础维度校验
        long numEmbeddings = embeddings.size(0);
        long embeddingDim = embeddings.size(1);

        // 2. 映射 EmbeddingFromPretrainedOptions 到 EmbeddingOptions
        EmbeddingOptions embOptions = new EmbeddingOptions(numEmbeddings, embeddingDim);

        // 处理 padding_idx
        if (options.padding_idx().has_value()) {
            embOptions.padding_idx().put(options.padding_idx());
        }

        // 处理 max_norm
        if (options.max_norm().has_value()) {
            embOptions.max_norm().put(options.max_norm());
        }

        // 处理 norm_type (DoublePointer)
        if (!options.norm_type().isNull()) {
            embOptions.norm_type().put(options.norm_type().get());
        }

        // 处理 scale_grad_by_freq (BoolPointer)
        if (!options.scale_grad_by_freq().isNull()) {
            embOptions.scale_grad_by_freq().put(options.scale_grad_by_freq().get());
        }

        // 处理 sparse (BoolPointer)
        if (!options.sparse().isNull()) {
            embOptions.sparse().put(options.sparse().get());
        }

        // 3. 创建实例并覆盖权重
        EmbeddingImpl embedding = new EmbeddingImpl(embOptions);

        // 关键：将预训练的 Tensor 设置为 weight
        // 在 C++ 底层这会替代默认初始化的变量
        embedding.weight(embeddings);

        // 4. 处理 freeze 逻辑
        if (!options.freeze().isNull() && options.freeze().get()) {
            embedding.weight().requires_grad_(false);
        }

        return embedding;
    }
    // 重载方法，匹配官方更多参数
    public static EmbeddingImpl from_pretrained2(Tensor embeddings, EmbeddingFromPretrainedOptions options) {
        // 从 EmbeddingFromPretrainedOptions 提取参数
        boolean freeze = options.freeze().get();
        Long pIdx = options.padding_idx().has_value() ? options.padding_idx().get() : null;

        double maxNorm = options.max_norm().has_value() ? options.max_norm().get() : 0.0;
        double normType = !options.norm_type().isNull() ? options.norm_type().get() : 2.0;
        boolean scaleGradByFreq = !options.scale_grad_by_freq().isNull() ? options.scale_grad_by_freq().get() : false;
        boolean sparse = !options.sparse().isNull() ? options.sparse().get() : false;
        return from_pretrained(embeddings, freeze, pIdx);
    }
}

//  public native @ByRef @NoException(true) DoubleOptional max_norm();
//  public native @ByRef @NoException(true) DoublePointer norm_type();
//  public native @Cast("bool*") @ByRef @NoException(true) BoolPointer scale_grad_by_freq();
//  public native @Cast("bool*") @ByRef @NoException(true) BoolPointer sparse();