package org.bytedeco.pytorch.geometric.demo.trainer;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;


import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.geometric.nn.model.PNA;
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;
import org.bytedeco.pytorch.geometric.nn.pooling.GlobalPooling;

import static org.bytedeco.pytorch.global.torch.*;

public class GraphDTANet extends org.bytedeco.pytorch.nn.Module {
    // 药物支路
    private final PNA drugEncoder;
    // 蛋白支路
    private final SequentialImpl proteinEncoder;
    private final LayerNorm drugNorm;
    private final LayerNorm proteinNorm;
    // 融合层
    private final LinearImpl fc1;
    private final LinearImpl fc2;
    private final LinearImpl out;
    private GlobalPooling readout;

    public GraphDTANet(long drugInDim, long proteinInDim) {
        super();

        // 1. 药物编码器 (使用 PNA 捕获复杂化学环境)
        //        String[] aggrs = {"mean", "max", "sum", "std"};
//        String[] scalers = {"identity", "amplification"};
//        this.drugEncoder = new PNA(drugInDim, 128, 128, 3, aggrs, scalers, 2.0);

        String[] aggrs = {"mean", "max", "sum"};  // 减少聚合器数量，提高稳定性
        String[] scalers = {"identity"};  // 只使用identity缩放
        this.drugEncoder = new PNA(drugInDim, 64, 64, 2, aggrs, scalers, 2.0);

        // 2. 蛋白质编码器 (使用 1D-CNN 处理氨基酸序列)
        this.proteinEncoder = new SequentialImpl();
        Conv1dOptions convOpt = new Conv1dOptions(proteinInDim, 32, new LongPointer(8));
        convOpt.stride().put(1);
//        convOpt.padding().put(4);
        convOpt.kernel_size().put(8);
        this.proteinEncoder.push_back(new Conv1dImpl(convOpt));
        this.proteinEncoder.push_back(new ReLUImpl());
        Conv1dOptions convOpt2 = new Conv1dOptions(32, 64, new LongPointer(8));
        convOpt.stride().put(1);
//        convOpt.padding().put(4);
        convOpt2.kernel_size().put(8);
        this.proteinEncoder.push_back(new Conv1dImpl(convOpt2));
        this.proteinEncoder.push_back(new ReLUImpl());
//        this.readout = new GlobalPooling();
//        this.proteinEncoder.push_back(readout);
//        var maxOpt = new MaxPool1dOptions(new LongPointer(3));
//        maxOpt.kernel_size().put(3);
//        maxOpt.stride().put(3);
//        this.proteinEncoder.push_back(new MaxPool1dImpl(maxOpt));

        // 3. 融合预测层
//        this.fc1 = new LinearImpl(128 + 64, 512);
//        this.fc2 = new LinearImpl(512, 256);
//        this.out = new LinearImpl(256, 16); // 预测亲和力数值 (pKd/pKi)

        this.fc1 = new LinearImpl(64 + 64, 256);
        this.fc2 = new LinearImpl(256, 128);
        this.out = new LinearImpl(128, 1);  // 输出层维度改为1
        this.drugNorm = new LayerNorm(64, 1e-5, true);
        this.proteinNorm = new LayerNorm(64, 1e-5, true);

        xavier_uniform_(fc1.weight());
        zeros_(fc1.bias());
        xavier_uniform_(fc2.weight());
        zeros_(fc2.bias());
        xavier_uniform_(out.weight());
        zeros_(out.bias());
//        this.drugNorm = new LayerNorm(128,1e-5, true);
//        this.proteinNorm = new LayerNorm(64,1e-5, true);
        register_module("drugNorm", drugNorm);
        register_module("proteinNorm", proteinNorm);
        register_module("drugEncoder", drugEncoder);
        register_module("proteinEncoder", proteinEncoder);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("out", out);
    }

    public Tensor forward(Tensor drugX, Tensor drugedge_index, Tensor proteinSeq) {
        // 药物特征提取 + 全局池化 [Batch, 128]
//        System.out.println("xDrug 0 NaN: " + isnan(drugX).any().item().toBool());
//        Tensor xDrug = drugEncoder.forward(drugX, drugedge_index);
//        System.out.println("xDrug 1 NaN: " + isnan(xDrug).any().item().toBool());
//        xDrug = xDrug.mean(new long[]{0}, true,new ScalarTypeOptional(kFloat()));//.nan_to_num(); // 简化版 Readout
//        System.out.println("xDrug 2 NaN: " + isnan(xDrug).any().item().toBool());
//        xDrug = drugNorm.forward(xDrug);
        Tensor xDrug = drugEncoder.forward(drugX, drugedge_index);
        Tensor batch = zeros(new long[]{drugX.size(0)}, drugX.options().dtype(new ScalarTypeOptional(kLong())));
        xDrug = GlobalPooling.pool(xDrug, batch, "mean");

        Tensor xProtein = proteinNorm.forward(proteinEncoder.forward(proteinSeq).mean(new long[]{-1}, false, new ScalarTypeOptional(kFloat()))).nan_to_num();
        Tensor combined = cat(new TensorVector(xDrug, xProtein), 1);
        combined = combined.clamp(new ScalarOptional(new Scalar(-5.0)), new ScalarOptional(new Scalar(5.0)));
        Tensor h = relu(fc1.forward(combined));
        h = dropout(relu(fc2.forward(h)), 0.2, is_training());

        Tensor output = out.forward(h);
        output = output.mul(new Scalar(1.0)).add(new Scalar(7.5));  // 调整到7.5为中心
        output = output.clamp(new ScalarOptional(new Scalar(5.0)), new ScalarOptional(new Scalar(10.0)));

        return output;
//        return out.forward(this.fc2.forward(relu(fc1.forward(combined))));
    }
}


//        System.out.println("xDrug 3 NaN: " + isnan(xDrug).any().item().toBool());

//        // 蛋白序列提取 [Batch, 64]
//        Tensor xProtein = proteinNorm.forward(proteinEncoder.forward(proteinSeq).mean(new long[]{-1}, false,new ScalarTypeOptional(kFloat()))).nan_to_num();
//
//        // 假装蛋白支路输出全 0
/// /        Tensor xProtein = rand(new long[]{1, 64}, xDrug.options());
//        // 拼接融合
//        Tensor combined = cat(new TensorVector(xDrug, xProtein), 1);
/// / 增加一个小技巧：对拼接后的特征做 clamp
//        combined = combined.clamp(new ScalarOptional(new Scalar(-10.0)), new ScalarOptional(new Scalar(10.0)));
//        Tensor h = relu(fc1.forward(combined));
//        h = dropout(relu(fc2.forward(h)), 0.2, is_training());
//        return out.forward(h);
//        Tensor xProtein = rand(new long[]{1, 64}, drugX.options()); // 模拟

//        Tensor xDrug = rand(new long[]{1, 128}, drugX.options()); // 模拟