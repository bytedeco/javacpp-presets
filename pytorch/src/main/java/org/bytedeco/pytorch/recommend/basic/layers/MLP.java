/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/MLP.scala
 *
 * Multi-Layer Perceptron using SequentialImpl (matching PyTorch's nn.Sequential).
 * Note: Scala did not register sequential as a submodule; we register it so
 * parameters() / optimizers see the Linear weights.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.nn.functional.FunctionalDropout;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.GELUImpl;
import org.bytedeco.pytorch.nn.modules.IdentityImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LeakyReLUImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.PReLUImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.SiLUImpl;
import org.bytedeco.pytorch.nn.modules.SigmoidImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import static org.bytedeco.pytorch.global.torch.kaiming_uniform_;

/**
 * Multi-Layer Perceptron using SequentialImpl (matching PyTorch's nn.Sequential)
 *
 * <p>Parameters
 * <ul>
 *   <li>inputDim — Input dimension.</li>
 *   <li>hiddenDims — Hidden layer sizes.</li>
 *   <li>outputDim — Output dimension (default=1).</li>
 *   <li>activation — Activation function (sigmoid, relu, prelu, dice, softmax, ...).</li>
 *   <li>dropout — Dropout probability.</li>
 *   <li>useBatchNorm — Whether to use batch norm.</li>
 *   <li>useLayerNorm — Whether to use layer norm.</li>
 *   <li>outputLayer — Whether to append a final Linear(*, outputDim) (default=true).</li>
 *   <li>device — Device for computation.</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MLP extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final ModuleListImpl sequential;
    private int moduleCounter = 0;
    private float dropout;
    private boolean useLinearOutFunc = false;
    private Parameter linearOutWeight;

    private void addModule(Module m) {
        String name = "layer_" + moduleCounter++;
        sequential.push_back(m);
        register_module(name, m);
    }

    private void addModule(String name, Module m) {
        sequential.push_back( m);
        register_module(name, m);
    }

    public MLP(long inputDim, long[] hiddenDims) {
        this(inputDim, hiddenDims, 1L, "relu", 0.0f, false, false, true, DeviceSupport.backend());
    }

    public MLP(long inputDim, long[] hiddenDims, long outputDim, String activation,
               float dropout, boolean useBatchNorm, String device) {
        this(inputDim, hiddenDims, outputDim, activation, dropout, useBatchNorm, false, true, device);
    }

    public MLP(long inputDim, long[] hiddenDims, long outputDim, String activation,
               float dropout, boolean useBatchNorm, boolean useLayerNorm,
               boolean outputLayer, String device) {
        super("MLP");
        this.sequential = new ModuleListImpl();
        long prevDim = inputDim;

        if (hiddenDims != null) {
            for (long dim : hiddenDims) {
            addModule(new LinearImpl(prevDim, dim));

                if (useLayerNorm) {
                    LongVector vec = new LongVector(1);
                    vec.put(0, dim);
                addModule(new LayerNormImpl(vec));
                } else if (useBatchNorm && !"relu".equals(activation)) {
                addModule(new BatchNorm1dImpl(new BatchNormOptions(dim)));
                }

                String act = activation == null ? "relu" : activation.toLowerCase();
                switch (act) {
                    case "relu":
                    addModule(new ReLUImpl());
                        break;
                    case "sigmoid":
                    addModule(new SigmoidImpl());
                        break;
                    case "tanh":
                    addModule(new TanhImpl());
                        break;
                    case "silu":
                    case "swish":
                    addModule(new SiLUImpl());
                        break;
                    case "gelu":
                    addModule(new GELUImpl());
                        break;
                    case "prelu":
                    addModule(new PReLUImpl());
                        break;
                    case "leaky_relu":
                    case "leakyrelu":
                    addModule(new LeakyReLUImpl());
                        break;
                    case "none":
                    case "identity":
                    addModule(new IdentityImpl());
                        break;
                    default:
                    addModule(new ReLUImpl());
                        break;
                }

                if (dropout > 0) {
                    this.dropout = dropout;
                    // Use functional wrapper to avoid ambiguity between DropoutImpl forward overloads
                    // (e.g., forward(Tensor) vs forward(Tensor, boolean)) which can cause native crashes.
                addModule(new FunctionalDropout(dropout));
                }
                prevDim = dim;
            }
        }
        addModule(new LinearImpl(prevDim, outputDim));
        // Register so parameters() sees Linear weights (Scala omitted this).
        register_module("sequential", sequential);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            sequential.to(dev, false);
            this.to(dev, false);
        }
    }

    /** DNN alias factory (Scala object DNN). */
    public static MLP dnn(long inputDim, long[] hiddenDims, long outputDim, String activation,
                          float dropout, boolean useBatchNorm, String device) {
        return new MLP(inputDim, hiddenDims, outputDim, activation, dropout, useBatchNorm,
                false, true, device);
    }

    @Override
    public Tensor forward(Tensor x) {
        try {
            if (x == null) {
                throw new IllegalArgumentException("Input tensor cannot be null");
            }
            if (x.numel() == 0) {
                throw new IllegalArgumentException("Input tensor has no elements");
            }
            // Execute modules in Java to avoid native SequentialImpl dispatch issues
            Tensor out = x;
            for (int i = 0; i < sequential.children().size(); i++) {   // or use the registry if you want
                out = sequential.children().get(i).forward(out);                   // this uses the typed Java wrappers
            }

            return out;
        } catch (RuntimeException e) {
            System.err.println("[MLP] Forward pass failed: " + e.getMessage());
            try {
                System.err.println("[MLP] Input shape: " + java.util.Arrays.toString(x.shape())
                        + ", dtype: " + x.dtype());
            } catch (Throwable ignored) {
            }
            e.printStackTrace();
            throw e;
        }
    }
}



//            out = sequential.forward(out);
//            if (dropout > 0){
//                out = torch.dropout(out, dropout, this.is_training());
//            }
//            if(useLinearOutFunc){
////                Tensor castedOut = out.to(torch.kFloat());
//                System.out.println("MLP Layer CastedOut final shape: " + java.util.Arrays.toString(out.shape())+ "out type: " +out.dtype().toScalarType().name()+" MLP Layer final linearOutWeight shape: " + java.util.Arrays.toString(linearOutWeight.shape()) +" weight type: " + linearOutWeight.dtype().toScalarType().name());
//                out = torch.linear(out, linearOutWeight.toType(out.dtype().toScalarType()));
//            }


//        if (outputLayer && dropout <= 0) {
//            this.useLinearOutFunc = false;
//            addModule(new LinearImpl(prevDim, outputDim));
//        }else if(outputLayer && dropout > 0){
//            this.useLinearOutFunc = true;
//            long finalDim = hiddenDims != null && hiddenDims.length > 0
//                    ? hiddenDims[hiddenDims.length - 1]
//                    : outputDim;
//            var weight = torch.empty(new long[]{finalDim, prevDim},new TensorOptions()
////                    .dtype(new ScalarTypeOptional(torch.kFloat()))
//                            .device(new DeviceOptional(new Device(device))),
////                            .requires_grad(new BoolOptional(true)),
//                    new MemoryFormatOptional());
//            kaiming_uniform_(weight, Math.sqrt(5.0),new FanModeType(new kFanOut()),new Nonlinearity(new kLeakyReLU()));
//            this.linearOutWeight = new Parameter(weight);
//        }else{
//            this.useLinearOutFunc = false;
//        }