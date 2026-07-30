package samples.demo.trainer;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.enumtype.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.enumtype.kMean;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.HuberLossOptions;
import org.bytedeco.pytorch.nn.options.MSELossOptions;
import org.bytedeco.pytorch.geometric.utils.MetricUtils;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.options.AdamWOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.schedulers.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import static org.bytedeco.pytorch.global.torch.hasMPS;
import static org.bytedeco.pytorch.global.torch.kFloat;

public class StateGridTrainer {
    public static void main(String[] args) {
        Device device = hasMPS() ? new Device(torch.DeviceType.CPU) : new Device(torch.DeviceType.CPU);
        System.out.println("国家电网核心调度计算引擎启动... Device: " + device);

        long numNodes = 2000;
        long hiddenDim = 32; // KGE 维度
        GridKGETransformer model = new GridKGETransformer(numNodes, 5, hiddenDim, 60);
        model.to(device, kFloat(), false);

        Optimizer optimizer = new AdamW(model.parameters(), new AdamWOptions(1e-4));
        //     @ByRef Optimizer optimizer,
        //        SchedulerMode mode/*=torch::optim::ReduceLROnPlateauScheduler::min*/,
        //        float factor/*=0.1*/,
        //        int patience/*=10*/,
        //        double threshold/*=1e-4*/,
        //        ThresholdMode threshold_mode/*=torch::optim::ReduceLROnPlateauScheduler::rel*/,
        //        int cooldown/*=0*/,
        //        @StdVector float[] min_lr/*=std::vector<float>()*/,
        //        double eps/*=1e-8*/,
        //        @Cast("bool") boolean verbose/*=false*/
        ReduceLROnPlateauScheduler scheduler = new ReduceLROnPlateauScheduler(optimizer, ReduceLROnPlateauScheduler.SchedulerMode.max, 0.35f, 30, 1e-4f, ReduceLROnPlateauScheduler.ThresholdMode.abs, 0, new FloatVector(), 1e-8f, false);
        StepLR stepLr = new StepLR(optimizer, 500, 0.5);
        MSELossOptions options = new MSELossOptions();
        options.reduction().put(new kMean());
//        MSELossImpl mse = new MSELossImpl(options);

        HuberLossOptions huberLossOptions = new HuberLossOptions();
        huberLossOptions.delta().put(1.0f);
        HuberLossImpl mse = new HuberLossImpl(huberLossOptions);
        model.train(true);
        for (int epoch = 1; epoch <= 10000; epoch++) {
            try (PointerScope scope = new PointerScope()) {

                GridKGEData.PowerSnapshot batch = new GridKGEData.PowerSnapshot(numNodes, 8000, 5000);
                batch.to(device);

                optimizer.zero_grad();

                // 1. 计算负载预测损失
                Tensor pred = model.forward(batch);
                // 1. 设置权重 (针对低负载节点提高权重)

// 如果 target 很小，1/(target + eps) 就会很大
                Tensor weights = torch.ones_like(batch.y).divide(batch.y.add(new Scalar(1.0f)));

// 2. 计算 Raw Loss (此时由于 reduction=none，得到的是向量)
                Tensor rawLoss = mse.forward(pred, batch.y);

// 3. 应用权重并聚合为标量 (Scalar)
// 只有变成了标量，后续的 .backward() 才能运行
                Tensor loadLoss = rawLoss.multiply(weights).mean();

// 4. 合并 KG Loss 并调用 backward
                Tensor kgLoss = model.calculateKgeloss(batch);
                Tensor totalLoss = loadLoss.add(kgLoss.multiply(new Scalar(12.0f)));

//                Tensor loadLoss = mse.forward(pred, batch.y);
//
//                // 2. 计算知识图谱结构损失 (确保 KGE 表征具有语义意义)
//                Tensor kgLoss = model.calculateKgeloss(batch);
//
//                // 联合优化
//                Tensor totalLoss = loadLoss.add(kgLoss.multiply(new Scalar(12.0f)));


                totalLoss.backward();
                optimizer.step();
//                if (epoch % 10 == 0) {
//                    optimizer.step();
////                stepLr.step();
//                }
                if (epoch % 100 == 0) {
                    float mape = MetricUtils.calculateMAPE(pred.detach(), batch.y.detach());
                    float r2 = MetricUtils.calculateR2(pred.detach(), batch.y.detach());

                    System.out.printf("Epoch %d | Loss: %.4f | MAPE: %.2f%% | R2: %.4f\n",
                            epoch, totalLoss.item().toFloat(), mape, r2);
                    System.out.printf("Epoch %d | Total Loss: %.4f | KG Loss: %.4f\n",
                            epoch, totalLoss.item().toFloat(), kgLoss.item().toFloat());
                    scheduler.step(totalLoss.item().toFloat());
                }
                // 在 StateGridTrainer 的 main 循环内
                if (epoch % 100 == 0) {
                    try (PointerScope kgeScope = new PointerScope()) {
                        // --- 准备排名测试数据 ---
                        // 选取 batch 中的前 50 个三元组进行评估
                        Tensor h = batch.headIndices.slice(0, new LongOptional(0), new LongOptional(50), 1);
                        Tensor r = batch.relIndices.slice(0, new LongOptional(0), new LongOptional(50), 1);
                        Tensor t = batch.tailIndices.slice(0, new LongOptional(0), new LongOptional(50), 1);

                        // 正样本得分
                        Tensor posS = model.getKgeScore(h, r, t);

                        // 构造干扰项得分 [50, 100] (每个正样本对比 100 个随机节点)
                        long numEvalNodes = 50;
                        long numNegatives = 100;
                        Tensor negTails = torch.randint(0, 2000, new long[]{numEvalNodes, numNegatives}, t.options());

                        // 为了计算 [50, 100] 的得分，我们需要广播 head 和 relation
                        Tensor hExp = h.unsqueeze(1).expand(new long[]{numEvalNodes, numNegatives}, true).reshape(-1);
                        Tensor rExp = r.unsqueeze(1).expand(new long[]{numEvalNodes, numNegatives}, true).reshape(-1);
                        Tensor tNegFlat = negTails.reshape(-1);

                        Tensor negS = model.getKgeScore(hExp, rExp, tNegFlat).view(numEvalNodes, numNegatives);

                        // 计算指标
                        float[] kgeMetrics = MetricUtils.calculateKGEMetrics(posS.detach(), negS.detach());

                        System.out.printf(">>> [KGE 拓扑理解力] MRR: %.4f | Hits@1: %.2f%% | Hits@10: %.2f%%\n",
                                kgeMetrics[0], kgeMetrics[1] * 100, kgeMetrics[2] * 100);
                    }
                }

            }

        }

        // 预测未来负载
        predict(model, device);
    }

    private static void predict(GridKGETransformer model, Device device) {
        model.eval();
        GridKGEData.PowerSnapshot testData = new GridKGEData.PowerSnapshot(2000, 3000, 1000);
        testData.to(device);

//        try (NoGradGuard noGrad = new NoGradGuard()) {
//            Tensor forecast = model.forward(testData);
//            for (int i = 0; i < forecast.size(0); i++) {
//                System.out.println("节点 " + i + " 未来负荷预测值: " + forecast.index(new TensorIndexVector(new TensorIndex(i))).item().toFloat() + " MW");
//            }
//        }
        try (NoGradGuard noGrad = new NoGradGuard()) {
            Tensor forecast = model.forward(testData);
            Tensor actual = testData.y;

            System.out.println("\n--- 国家电网调度负荷预测精确度报告 ---");
            System.out.printf("%-10s | %-12s | %-12s | %-8s\n", "节点ID", "真实负荷(MW)", "预测负荷(MW)", "偏差率");
            System.out.println("------------------------------------------------------------");

            for (int i = 0; i < 50; i++) {
                float realVal = actual.index(new TensorIndexVector(new TensorIndex(i))).item().toFloat();
                float predVal = forecast.index(new TensorIndexVector(new TensorIndex(i))).item().toFloat();
                float error = Math.abs(realVal - predVal) / realVal * 100;

                System.out.printf("节点 #%-6d | %-12.2f | %-12.2f | %-7.2f%%\n",
                        i, realVal, predVal, error);
            }

            float finalMAPE = MetricUtils.calculateMAPE(forecast, actual);
            System.out.printf("\n全网平均预测精确度 (MAPE): %.2f%%\n", finalMAPE);
        }
    }
}