package samples.demo.trainer;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
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
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.schedulers.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class ProteinTrainer {
    public static void main(String[] args) {
        // 1. 设置 MPS 设备
        Device device = new Device(DeviceType.MPS);
        if (hasMPS()) {
            System.out.println("MPS not available, using CPU");
            device = new Device(DeviceType.CPU);
        }
        System.out.println("Using Device: " + device);

        // 2. 初始化模型与数据集
        long hiddenDim = 32;
        long numClasses = 20;
        ProteinGNN model = new ProteinGNN(16, hiddenDim, numClasses);
        model.to(device, kFloat(), false);

        Optimizer optimizer = new Adam(model.parameters(), new AdamOptions(0.001));
        CrossEntropyLossImpl criterion = new CrossEntropyLossImpl();
        ReduceLROnPlateauScheduler scheduler = new ReduceLROnPlateauScheduler(optimizer, ReduceLROnPlateauScheduler.SchedulerMode.min, 0.5f, 10, 1e-4f, ReduceLROnPlateauScheduler.ThresholdMode.rel, 0, new FloatVector(), 1e-8f, false);

        // 3. 模拟训练循环
        model.train(true);
        for (int epoch = 1; epoch <= 5000; epoch++) {
            double totalLoss = 0;

            // 每次训练模拟 10 个蛋白质
            for (int i = 0; i < 10; i++) {
                ProteinDataFactory.ProteinGraph protein = new ProteinDataFactory.ProteinGraph(20 + i, 50, numClasses);
                protein.to(device);

                optimizer.zero_grad();
                Tensor pred = model.forward(protein.x, protein.edge_index);
                Tensor loss = criterion.forward(pred, protein.y);

                loss.backward();
                optimizer.step();
//                scheduler.step(loss.item().toFloat());
                totalLoss += loss.item().toDouble();
//                pred.print();
//                float r2 = MetricUtils.calculateR2(pred.view(-1).detach(), protein.y.view(-1).to(ScalarType.Float).detach());
//                System.out.printf("Epoch %d, Protein %d, Loss: %.4f, R2: %.4f\n", epoch, i, loss.item().toFloat(), r2);

            }

            if (epoch % 20 == 0) {
                scheduler.step((float) totalLoss);

//                System.out.printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss / 10);
            }
            if (epoch % 200 == 0) {

                System.out.printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss / 10);
            }
        }

        // 4. 预测演示
        predict(model, device);
    }

    public static void predict(ProteinGNN model, Device device) {
        model.eval();
        System.out.println("\n--- Starting Prediction ---");

        ProteinDataFactory.ProteinGraph testProtein = new ProteinDataFactory.ProteinGraph(30, 80, 5);
        testProtein.to(device);

        try (NoGradGuard noGrad = new NoGradGuard()) {
            Tensor logits = model.forward(testProtein.x, testProtein.edge_index);
            Tensor prob = softmax(logits, 1);
            long predictedClass = prob.argmax(new LongOptional(1), false).item().toLong();

            System.out.println("Predicted Function Class ID: " + predictedClass);
            System.out.println("Confidence: " + prob.index(new TensorIndexVector(new TensorIndex(0), new TensorIndex(predictedClass))).item().toFloat());
        }
    }
}