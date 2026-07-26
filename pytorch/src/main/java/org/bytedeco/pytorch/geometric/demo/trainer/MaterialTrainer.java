package org.bytedeco.pytorch.geometric.demo.trainer;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.clip_grad_norm_;
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
public class MaterialTrainer {
    public static void main(String[] args) {
        Device device = new Device(torch.DeviceType.CPU);
        MaterialGNN model = new MaterialGNN(109, 64, 64);
        model.to(device, true);

        Adam optimizer = new Adam(model.parameters(), new AdamOptions(5e-4));
        HuberLossImpl criterion = new HuberLossImpl();
        ReduceLROnPlateauScheduler scheduler = new ReduceLROnPlateauScheduler(optimizer, ReduceLROnPlateauScheduler.SchedulerMode.min, 0.5f, 10, 1e-4f, ReduceLROnPlateauScheduler.ThresholdMode.rel, 0, new FloatVector(), 1e-8f, false);

        System.out.println("🧪 化学新材料特性预测系统启动...");

        for (int epoch = 1; epoch <= 10000; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < 20; i++) {
                // 模拟不同规模的合成材料
                MaterialDataFactory.MaterialGraph material = new MaterialDataFactory.MaterialGraph(10 + i, 30 + i * 2);

                optimizer.zero_grad();
                Tensor pred = model.forward(material.x, material.edge_index);
                Tensor loss = criterion.forward(pred.view(-1), material.property.view(-1));

                loss.backward();
                clip_grad_norm_(model.parameters(), 1.0);
                optimizer.step();

                totalLoss += loss.item().toDouble();
            }
            if (epoch % 50 == 0) {
                scheduler.step((float) totalLoss / 20);
                System.out.printf("Epoch %d, 材料特性预测误差 (Loss): %.6f\n", epoch, totalLoss / 20);
            }
        }
    }
}
