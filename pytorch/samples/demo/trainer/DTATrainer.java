package samples.demo.trainer;
import org.bytedeco.pytorch.nn.options.*;


import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.options.HuberLossOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.optim.schedulers.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.geometric.utils.MetricUtils;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.clip_grad_norm_;
//import org.bytedeco.pytorch.geometric.nn.model.GraphDTANet;

public class DTATrainer {
    public static void main(String[] args) {
        Device device = new Device(DeviceType.CPU);// hasMPS() ? new Device(DeviceType.MPS) : new Device(DeviceType.CPU);

        // 1. 初始化模型：药物原子特征16维，蛋白输入20维
        GraphDTANet model = new GraphDTANet(16, 20);
        model.to(device, false);

        AdamOptions adamOpts = new AdamOptions(1e-4);  // 提高学习率
        adamOpts.weight_decay().put(1e-5);
        Adam optimizer = new Adam(model.parameters(), adamOpts);
        ReduceLROnPlateauScheduler scheduler = new ReduceLROnPlateauScheduler(optimizer, ReduceLROnPlateauScheduler.SchedulerMode.max, 0.35f, 30, 1e-4f, ReduceLROnPlateauScheduler.ThresholdMode.abs, 0, new FloatVector(), 1e-8f, false);
        StepLR stepLr = new StepLR(optimizer, 100, 0.1);
        HuberLossOptions huberLossOptions = new HuberLossOptions();
        huberLossOptions.delta().put(1.0f);
        HuberLossImpl huber = new HuberLossImpl(huberLossOptions);
//        HuberLossImpl huber = new HuberLossImpl(new HuberLossOptions().delta(1.0));

        System.out.println("🧬 癌症靶向药研训系统启动 | 设备: " + device.type());

        for (int epoch = 1; epoch <= 5000; epoch++) {
            try (PointerScope scope = new PointerScope()) {
                model.train(true);
                optimizer.zero_grad();

                // 2. 获取模拟药研数据 (Ligand + Protein)
                CancerDrugData.DrugTargetPair batch = CancerDrugData.generateMockData();

                // 3. 前向传播：预测亲和力 pKd
                int batchSize = 16;
//                Tensor targets = rand(new long[]{batchSize}, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)).requires_grad(new BoolOptional(false))).multiply(new Scalar(5.0f)).add(new Scalar(5.0f)); // 5.0 到 10.0 之间的随机数
                Tensor prediction = model.forward(batch.drugX, batch.drugedge_index, batch.proteinSeq);
//                prediction.print();

                Tensor loss = huber.forward(prediction.view(-1), batch.affinity.view(-1));

                loss.backward();
                // 梯度裁剪：防止梯度过大导致 NaN
                clip_grad_norm_(model.parameters(), 0.9);
                optimizer.step();
//                scheduler.step(loss.item().toFloat());
//                stepLr.step();

                // 4. 定期评估多维度指标
                if (epoch % 20 == 0) {
                    scheduler.step(loss.item().toFloat());
                    evaluateMetrics(epoch, prediction.detach(), batch.affinity.detach(), loss.item().toFloat()); //batch.affinity.detach(),
                }
            }
        }
    }

    private static void evaluateMetrics(int epoch, Tensor pred, Tensor target, float loss) {
        // 回归指标

        float mape = MetricUtils.calculateMAPE(pred, target);
        float rmse = MetricUtils.calculateRMSE(pred, target);
        float r2 = MetricUtils.calculateR2(pred, target);
        float[] kgeMetrics = MetricUtils.calculateKGEMetrics(pred, target);
        float mr = kgeMetrics[0];
        float mrr = kgeMetrics[1];
        float h10 = kgeMetrics[2];
        // 药研关键指标：Pearson 相关系数 (衡量预测趋势的一致性)
//        float pearson = calculatePearson(pred, target);

        // 分类指标：将亲和力 > 7.0 定义为“高活性候选药”
        float auc = MetricUtils.calculateAUC(pred, target, 7.0f);
//        System.out.println(String.format("Epoch %d | Loss: %.4f | pred: %.4f | target: %.4f \r", epoch, loss, pred.item().toFloat(), target.item().toFloat()));
        System.out.printf("Epoch %d | Loss: %.4f | RMSE: %.4f | MAPE: %.2f%% | r2: %.4f | mr: %.3f | mrr: %.3f | h10: %.3f\n",
                epoch, loss, rmse, mape, r2, mr, mrr, h10);
    }


    //                System.out.println("DrugX NaN: " + isnan(batch.drugX).any().item().toBool());
//                System.out.println("ProteinSeq NaN: " + isnan(batch.proteinSeq).any().item().toBool());
//                batch.to(device,false);

//                batch.affinity.print();
//                batch.proteinSeq.print();
//                batch.drugX.print();
//                batch.drugedge_index.print();
//    private static float calculatePearson(Tensor x, Tensor y) {
//        Tensor vx = x.subtract(x.mean());
//        Tensor vy = y.subtract(y.mean());
//        Tensor corr = sum(vx.multiply(vy)).divide(
//                torcsqrt(torch.sum(vx.pow(new Scalar(2)))).multiply(torch.sqrt(torch.sum(vy.pow(new Scalar(2))))).add(1e-8)
//        );
//        return corr.item().toFloat();
//    }
}
