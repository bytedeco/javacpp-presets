package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.data.datasets.*;
import org.bytedeco.pytorch.data.options.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.Module;
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
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.*;
import static org.bytedeco.pytorch.global.torch.arange;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

public class NeuralFingerprint extends Module {
    private ModuleListImpl convs;
    private ModuleListImpl writeouts; // Readout layers
    private long fingerprintDim;
    private int numLayers;
    private GCNConv[] convArray;

    //class NeuralFingerprint(in_channels: int, hidden_channels: int, out_channels: int, num_layers: int,
    public NeuralFingerprint(long inChannels, long hiddenChannels, long out_channels, int numLayers) {
        this.fingerprintDim = out_channels;
        this.numLayers = numLayers;
        this.convs = new ModuleListImpl();
        this.writeouts = new ModuleListImpl();
        this.convArray = new GCNConv[numLayers];
        for (int i = 0; i < numLayers; i++) {
            long dim = (i==0) ? inChannels : hiddenChannels;
            // Simplified GCN update: H = sigma(H @ W + A @ H @ W)
            // Here we assume standard org.bytedeco.pytorch.geometric.nn.conv.GCNConv
            // 注册卷积层
            GCNConv conv = new GCNConv(dim, hiddenChannels);
            convs.push_back(conv);
            convArray[i] = conv; // 永久保存 Java 侧的强类型引用
            // 注册 Readout 线性层 (映射到指纹空间)
            LinearImpl out = new LinearImpl(dim, fingerprintDim);
            writeouts.push_back(out);
//            convs.register_module(String.valueOf(i), new GCNConv(dim, hiddenChannels));
//
//            // Writeout: H -> Fingerprint Bit Score
//            // Mapping to fingerprint space
//            LinearImpl out = new LinearImpl(dim, fingerprintDim);
//            writeouts.register_module(String.valueOf(i), out);
        }
        register_module("convs", convs);
        register_module("writeouts", writeouts);
    }

    public Tensor forward(Tensor x, Tensor edge_index, Tensor batch) {
        long batchSize = (batch == null) ? 1 : batch.max().item().toLong() + 1;
        Tensor fingerprint = torch.zeros(new long[]{batchSize, fingerprintDim}, x.options());

        // Neural FP Iteration //convs.size()
        for (int i = 0; i < numLayers; i++) {
            // 1. Writeout (Update Fingerprint)
            // i-th layer representation
            LinearImpl writer = writeouts.get(i).asLinear();//String.valueOf(i));
            // 2. Update Atom Features
//            var conv = convs.get(i);//String.valueOf(i));
            GCNConv conv = convArray[i];// new GCNConv(convArray[i].get(i));
            Tensor score = writer.forward(x).softmax(1); // Softmax over bits

            // Sum over atoms in a molecule
            Tensor graphScore = AggrUtils.scatter(score, batch, batchSize, "sum");
            fingerprint = fingerprint.add(graphScore);


            x = conv.forward(x, edge_index).relu();
        }

        return fingerprint;
    }
}
