package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Multi-aggregation: concatenates outputs of several {@link Aggregation}s on the
 * feature dimension.
 *
 * <pre>
 *   y = [ Aggr_1(x); Aggr_2(x); … ; Aggr_k(x) ]   // concat on dim=1
 * </pre>
 */
public class MultiAggregation extends Aggregation {

    private final List<Aggregation> aggrs;

    public MultiAggregation(List<Aggregation> aggrs) {
        super();
        if (aggrs == null || aggrs.isEmpty()) {
            throw new IllegalArgumentException("MultiAggregation requires ≥1 aggregator");
        }
        this.aggrs = new ArrayList<>(aggrs);
        for (int i = 0; i < this.aggrs.size(); i++) {
            if (this.aggrs.get(i) == null) {
                throw new IllegalArgumentException("aggregator[" + i + "] is null");
            }
            register_module("aggr_" + i, this.aggrs.get(i));
        }
    }

    public MultiAggregation(Aggregation... aggrs) {
        this(Arrays.asList(aggrs));
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        if (x == null || index == null) {
            throw new NullPointerException("x and index must not be null");
        }
        List<Tensor> results = new ArrayList<>(aggrs.size());
        for (Aggregation aggr : aggrs) {
            results.add(aggr.forward(x, index, dimSize));
        }
        return torch.cat(new TensorVector(results.toArray(new Tensor[0])), 1);
    }

    public List<Aggregation> getAggregators() {
        return Collections.unmodifiableList(aggrs);
    }

    public int size() {
        return aggrs.size();
    }
}
