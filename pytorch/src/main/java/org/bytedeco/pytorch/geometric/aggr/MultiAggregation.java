package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;


/**
 * 1. org.bytedeco.pytorch.geometric.aggr.MultiAggregation
 * 将多个聚合器的结果拼接 (Concat)
 */
public class MultiAggregation extends Aggregation {
    private List<Aggregation> aggrs;

    public MultiAggregation(List<Aggregation> aggrs) {
        this.aggrs = aggrs;
        for (int i = 0; i < aggrs.size(); i++) {
            register_module("aggr_" + i, aggrs.get(i));
        }
    }

    // 辅助构造函数，支持可变参数
    public MultiAggregation(Aggregation... aggrs) {
        this(java.util.Arrays.asList(aggrs));
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        List<Tensor> results = new ArrayList<>();
//        var refs =  new TensorArrayRef();
//        TensorVector outputs = new TensorVector();
        for (Aggregation aggr : aggrs) {
            Tensor res = aggr.forward(x, index, dimSize);
//            refs.
            results.add(res);
//            outputs.put(res);
        }
//        System.out.println("MultiAggregation Structure initialized.");
        // 在特征维度 (dim=1) 拼接
        Tensor cats = torch.cat(new TensorVector(results.toArray(new Tensor[0])), 1);
//        System.out.println("MultiAggregation Structure  get cat.");
        return cats;
    }
}



