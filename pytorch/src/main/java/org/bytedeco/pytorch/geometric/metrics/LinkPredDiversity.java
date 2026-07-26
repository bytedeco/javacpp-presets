package org.bytedeco.pytorch.geometric.metrics;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorVector;

import static org.bytedeco.pytorch.global.torch.*;

public class LinkPredDiversity extends LinkPredMetric {
    private Tensor itemCategories; // [num_items] 存储每个 Item 的类别 ID

    public LinkPredDiversity(int k, Tensor itemCategories) {
        super(k);
        this.itemCategories = itemCategories;
    }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            // 1. 获取 Top-K 索引 [Batch, K]
            Tensor indices = yPred.topk(k, 1, true, true).get1();

            // 2. 将 Item 索引映射为类别 ID [Batch, K]
            // 使用 index_select 的一维平铺版，然后 reshape 回来
            Tensor recommendedCats = itemCategories.index_select(0, indices.view(-1)).view(indices.sizes());

            // 3. 计算每个 Batch 内部唯一类别的数量
            // 在 LibTorch 中，没有直接按行求 unique 的算子，通常使用循环或 bitset 逻辑
            TensorVector diversityResults = new TensorVector();
            for (int i = 0; i < recommendedCats.size(0); i++) {
                long numUniqueCats = unique_consecutive(recommendedCats.index(new TensorIndexVector(new TensorIndex(tensor(i))))).get0().size(0);
                diversityResults.push_back(tensor((double) numUniqueCats / k).to(kFloat()));
            }

            return cat(diversityResults, 0).detach();
//        }
    }
}
