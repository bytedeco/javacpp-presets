package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 9. org.bytedeco.pytorch.geometric.aggr.DegreeScalerAggregation
 * PNA 中的 Scaler: Identity, Amplification, Attenuation
 * Formula: agg * (log(deg + 1) / avg_log_deg)
 */
public class DegreeScalerAggregation extends Aggregation {
    private double avgLogDeg; // 训练集上的平均度数对数 (预计算常数)
    private List<String> scalers; // "identity", "amplification", "attenuation"
    private Aggregation baseAggr; // 基础聚合器 (如 Mean)

    public DegreeScalerAggregation(double avgDeg, List<String> scalers, Aggregation baseAggr) {
        // avgLogDeg 近似为 log(avgDeg + 1)
        this.avgLogDeg = Math.log(avgDeg + 1);
        this.scalers = scalers;
        this.baseAggr = baseAggr;
        register_module("baseAggr", baseAggr);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 获取基础聚合结果
        Tensor agg = baseAggr.forward(x, index, dimSize).contiguous();

        // 2. 计算度数缩放因子
        Tensor deg = AggrUtils.compute_degree(index, dimSize).to(x.dtype());
        Tensor logDeg = deg.add(new Scalar(1.0)).log();

        // 3. 准备存储容器：使用 ArrayList 临时存储，最后转为 Tensor[]
        List<Tensor> outputs = new ArrayList<>();
//        System.out.println("DegreeScalerAggregation Structure initialized. 111");
        for (String scaler : scalers) {
            if ("identity".equals(scaler)) {
                outputs.add(agg);
            } else if ("amplification".equals(scaler)) {
                Tensor scale = logDeg.divide(new Scalar(avgLogDeg)).unsqueeze(1);
                // 显式 expand 并连续化内存，防止广播计算中的指针异常
                outputs.add(agg.multiply(scale.expand_as(agg)).contiguous());
            } else if ("attenuation".equals(scaler)) {
                Tensor scale = torch.tensor(avgLogDeg).divide(logDeg.clamp_min(new Scalar(1e-5))).unsqueeze(1);
                outputs.add(agg.multiply(scale.expand_as(agg)).contiguous());
            }
        }
//        System.out.println("DegreeScalerAggregation Structure initialized. 222");
        // --- 核心修复：将 List 转为原生 Tensor 数组进行 cat ---
        Tensor[] tensorArray = outputs.toArray(new Tensor[0]);
//        System.out.println("DegreeScalerAggregation Structure initialized. 333");
        // 调用 TensorArrayRef 版本，这在 JavaCPP 中是最稳定的
        Tensor result = torch.cat(new TensorVector(tensorArray), 1);
//        System.out.println("DegreeScalerAggregation Structure initialized. 444");
        return result;
    }

    public Tensor forward5(Tensor x, Tensor index, long dimSize) {
        // 1. 获取基础聚合结果 [dimSize, total_features]
        Tensor agg = baseAggr.forward(x, index, dimSize);

        // 2. 计算度数
        // 注意：必须 .to(x.dtype()) 否则 cross-type 运算在底层可能崩溃
        Tensor deg = AggrUtils.compute_degree(index, dimSize).to(x.dtype());
        Tensor logDeg = deg.add(new Scalar(1.0)).log();

        // --- 核心修复：使用 List 保持 Java 强引用，防止 GC 导致 SIGSEGV ---
        List<Tensor> tensorRefs = new ArrayList<>();
        TensorVector outputs = new TensorVector();

        for (String scaler : scalers) {
            Tensor scaled;
            if ("identity".equals(scaler)) {
                scaled = agg;
            } else if ("amplification".equals(scaler)) {
                Tensor scale = logDeg.divide(new Scalar(avgLogDeg)).unsqueeze(1);
                scaled = agg.multiply(scale.expand_as(agg));
            } else if ("attenuation".equals(scaler)) {
                // 加上 1e-5 防止 logDeg 为 0 导致的 Inf/NaN 在后续运算中崩溃
                Tensor scale = torch.tensor(new Scalar(avgLogDeg)).divide(logDeg.clamp_min(new Scalar(1e-5))).unsqueeze(1);
                scaled = agg.multiply(scale.expand_as(agg));
            } else {
                continue;
            }

            // 同时保存到 Java List 和 TensorVector
            tensorRefs.add(scaled);
            outputs.put(scaled);
        }

        // 3. 执行 cat
        Tensor result = torch.cat(outputs, 1);

        // 显式清理，虽然不是必须，但有助于理解生命周期
//        outputs.clear();
//        tensorRefs.clear();

        return result;
    }

    //    @Override
    public Tensor forward4(Tensor x, Tensor index, long dimSize) {
        // 1. 获取基础聚合结果
        // 如果 baseAggr 是 MultiAggregation，它返回 [dimSize, num_aggrs * features]
        Tensor agg = baseAggr.forward(x, index, dimSize);

        // 2. 计算度数缩放因子 (严格对齐 PNA 论文公式)
        // deg 形状 [dimSize]
        Tensor deg = AggrUtils.compute_degree(index, dimSize).to(x.dtype());
        // log_deg = log(deg + 1)
        Tensor logDeg = deg.add(new Scalar(1.0)).log();

        TensorVector finalOutputs = new TensorVector();

        // 3. 对聚合后的 Tensor 进行切片处理，或者整体应用缩放后再拼接
        // 逻辑修正：s(x, d) = x * [log(d+1)/avg] 或 x * [avg/log(d+1)]
        for (String scaler : scalers) {
            Tensor scale;
            if ("identity".equals(scaler)) {
                finalOutputs.put(agg);
                continue;
            } else if ("amplification".equals(scaler)) {
                // scale = log(deg+1) / avgLogDeg
                scale = logDeg.divide(new Scalar(avgLogDeg));
            } else if ("attenuation".equals(scaler)) {
                // scale = avgLogDeg / log(deg+1)
                scale = torch.tensor(new Scalar(avgLogDeg)).divide(logDeg.clamp_min(new Scalar(1e-5)));
            } else {
                continue;
            }

            // scale 形状从 [dimSize] 变为 [dimSize, 1] 以便广播
            // agg 是 [dimSize, total_features]
            finalOutputs.put(agg.multiply(scale.unsqueeze(1).expand_as(agg)));
        }

        // 4. 最终拼接：[dimSize, num_scalers * (num_aggrs * features)]
        return torch.cat(finalOutputs, 1);
    }

    //    @Override
    public Tensor forward3(Tensor x, Tensor index, long dimSize) {
        // 1. 基础聚合
        Tensor agg = baseAggr.forward(x, index, dimSize);

        // 2. 计算度数
        Tensor deg = AggrUtils.compute_degree(index, dimSize);
        deg = deg.clamp_min(new Scalar(1.0));

        // 3. 计算 log(deg + 1)
        Tensor logDeg = deg.add(new Scalar(1.0)).log();

        // 4. 应用 Scalers
        TensorVector outputs = new TensorVector();

        for (String scaler : scalers) {
            if ("identity".equals(scaler)) {
                outputs.put(agg);
            } else if ("amplification".equals(scaler)) {
                // Formula: scale = log(deg+1) / avg
                // [优化] 改用乘法: scale = log(deg+1) * (1/avg)
                Tensor scale = logDeg.mul(new Scalar(1.0 / avgLogDeg));

                outputs.put(agg.mul(scale.unsqueeze(1)));
            } else if ("attenuation".equals(scaler)) {
                // Formula: scale = avg / log(deg+1)
                // ❌ 原崩溃代码: new Tensor(new Scalar(avg)).div(logDeg)
                // [修复] 改用倒数乘法: scale = (log(deg+1))^-1 * avg
                // reciprocal() 是求倒数 1/x
                Tensor scale = logDeg.reciprocal().mul(new Scalar(avgLogDeg));

                outputs.put(agg.mul(scale.unsqueeze(1)));
            }
        }

        return torch.cat(outputs, 1);
    }
//    @Override
    public Tensor forward2(Tensor x, Tensor index, long dimSize) {
        // 1. 基础聚合 (例如 Mean)
        Tensor agg = baseAggr.forward(x, index, dimSize); // [N, C]

        // 2. 计算度数
        Tensor deg = AggrUtils.compute_degree(index, dimSize); // [N]
        deg = deg.clamp_min(new Scalar(1.0)); // 避免 log(0)

        // 3. 计算 log(deg + 1)
        Tensor logDeg = deg.add(new Scalar(1.0)).log();

        // 4. 应用 Scalers 并拼接
        TensorVector outputs = new TensorVector();

        for (String scaler : scalers) {
            if ("identity".equals(scaler)) {
                outputs.put(agg);
            } else if ("amplification".equals(scaler)) {
                // scale = log(deg+1) / avg
                Tensor scale = logDeg.div(new Scalar(avgLogDeg));
                // 广播 scale: [N] -> [N, 1]
                outputs.put(agg.mul(scale.unsqueeze(1)));
            } else if ("attenuation".equals(scaler)) {
                // scale = avg / log(deg+1)
                Tensor scale = new Tensor(new Scalar(avgLogDeg)).div(logDeg);
                outputs.put(agg.mul(scale.unsqueeze(1)));
            }
        }

        return torch.cat(outputs, 1);
    }
}
