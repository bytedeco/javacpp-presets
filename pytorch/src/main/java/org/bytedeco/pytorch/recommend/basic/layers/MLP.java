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
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.functional.FunctionalDropout;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
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
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

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

    private final SequentialImpl sequential;

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
        this.sequential = new SequentialImpl();
        long prevDim = inputDim;
        int dropoutCount = 0;

        if (hiddenDims != null) {
            for (long dim : hiddenDims) {
                sequential.push_back(new LinearImpl(prevDim, dim));

                if (useLayerNorm) {
                    LongVector vec = new LongVector(1);
                    vec.put(0, dim);
                    sequential.push_back(new LayerNormImpl(vec));
                } else if (useBatchNorm && !"relu".equals(activation)) {
                    sequential.push_back(new BatchNorm1dImpl(new BatchNormOptions(dim)));
                }

                String act = activation == null ? "relu" : activation.toLowerCase();
                switch (act) {
                    case "relu":
                        sequential.push_back(new ReLUImpl());
                        break;
                    case "sigmoid":
                        sequential.push_back(new SigmoidImpl());
                        break;
                    case "tanh":
                        sequential.push_back(new TanhImpl());
                        break;
                    case "silu":
                    case "swish":
                        sequential.push_back(new SiLUImpl());
                        break;
                    case "gelu":
                        sequential.push_back(new GELUImpl());
                        break;
                    case "prelu":
                        sequential.push_back(new PReLUImpl());
                        break;
                    case "leaky_relu":
                    case "leakyrelu":
                        sequential.push_back(new LeakyReLUImpl());
                        break;
                    case "none":
                    case "identity":
                        sequential.push_back(new IdentityImpl());
                        break;
                    default:
                        sequential.push_back(new ReLUImpl());
                        break;
                }

                if (dropout > 0) {
                    // Use functional wrapper to avoid ambiguity between DropoutImpl forward overloads
                    // (e.g., forward(Tensor) vs forward(Tensor, boolean)) which can cause native crashes.
                    sequential.push_back(new FunctionalDropout(dropout));
                }

                prevDim = dim;
            }
        }

        if (outputLayer) {
            sequential.push_back(new LinearImpl(prevDim, outputDim));
        }

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
            return sequential.forward(x);
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