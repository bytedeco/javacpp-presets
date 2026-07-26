package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

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
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * Equilibrium org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 迭代更新 z = f(z, x) 直到收敛或达到最大迭代次数。
 */
public class EquilibriumAggregation extends Aggregation {
    private Module f; // Update function: (z, x) -> z_new
    private int maxIter;
    private double tol;

    public EquilibriumAggregation(int channels, int maxIter, double tol) {
        this.maxIter = maxIter;
        this.tol = tol;

        // f 简单实现为 MLP: Cat(z, x) -> Linear -> Tanh -> Linear
        SequentialImpl mlp = new SequentialImpl();
        mlp.push_back(new LinearImpl(2 * channels, channels));
        mlp.push_back(new TanhImpl());
        this.f = mlp;
        register_module("f", f);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // Init Z as 0
        Tensor z = torch.zeros(new long[]{dimSize, x.size(1)}, x.options());

        // 聚合输入的 X (作为 context)
        // 注意：Equilibrium 通常是 Global Pooling，这里适配为 Local org.bytedeco.pytorch.geometric.aggr.Aggregation
        // Context 我们可以取 x 的 Mean 作为初始猜测或条件
        Tensor xAgg = AggrUtils.scatter(x, index, dimSize, "mean");

        for (int i = 0; i < maxIter; i++) {
            Tensor zPrev = z;

            // f(z, x_agg)
            // Cat: [Batch, 2*C]
            Tensor inp = torch.cat(new TensorVector(z, xAgg), 1);
            z = ((SequentialImpl)f).forward(inp);

            // Check convergence
            Tensor diff = z.sub(zPrev).abs().max();
            if (diff.item().toDouble() < tol) {
                break;
            }
        }
        return z;
    }
}