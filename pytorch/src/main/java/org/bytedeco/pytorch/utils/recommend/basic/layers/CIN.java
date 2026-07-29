/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CIN.scala
 *
 * Compressed Interaction Network (xDeepFM).
 * Input: (batch, numFields, embedDim) → Output: (batch, 1)
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CIN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int inputDim;
    private final int[] cinSize;
    private final boolean splitHalf;
    private final int numLayers;
    private final List<Conv1dImpl> convLayers = new ArrayList<>();
    private final LinearImpl fc;

    public CIN(int inputDim, int[] cinSize) {
        this(inputDim, cinSize, true, DeviceSupport.backend());
    }

    public CIN(int inputDim, int[] cinSize, boolean splitHalf, String device) {
        super("CIN");
        if (inputDim <= 0) {
            throw new IllegalArgumentException("CIN: inputDim must be > 0, got " + inputDim);
        }
        if (cinSize == null || cinSize.length == 0) {
            throw new IllegalArgumentException("CIN: cinSize must not be empty");
        }
        this.inputDim = inputDim;
        this.cinSize = cinSize.clone();
        this.splitHalf = splitHalf;
        this.numLayers = cinSize.length;

        int prevDim = inputDim;
        int fcInputDim = 0;
        for (int i = 0; i < numLayers; i++) {
            int fullSize = cinSize[i];
            LongPointer kernel = new LongPointer(new long[]{1L});
            Conv1dOptions opt = new Conv1dOptions(
                    (long) inputDim * prevDim, fullSize, kernel);
            Conv1dImpl conv = new Conv1dImpl(opt);
            register_module("conv_" + i, conv);
            convLayers.add(conv);

            boolean splitLayer = splitHalf && i != numLayers - 1;
            int sizeForNext = splitLayer ? fullSize / 2 : fullSize;
            prevDim = sizeForNext;
            fcInputDim += sizeForNext;
        }

        this.fc = new LinearImpl(fcInputDim, 1L);
        register_module("fc", fc);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, inputDim, embedDim)
        long batchSize = x.size(0);
        long embedDim = x.size(2);

        Tensor x0 = x.unsqueeze(2);              // (batch, inputDim, 1, embedDim)
        Tensor h = x;                             // updated each layer
        List<Tensor> xs = new ArrayList<>();

        for (int i = 0; i < numLayers; i++) {
            Tensor outer = x0.mul(h.unsqueeze(1));  // (batch, inputDim, prevDim, embedDim)
            Tensor reshaped = outer.view(batchSize, inputDim * h.size(1), embedDim);

            Tensor convOut = convLayers.get(i).forward(reshaped).relu();

            if (splitHalf && i != numLayers - 1) {
                long halfSize = convOut.size(1) / 2;
                Tensor first = convOut.narrow(1, 0L, halfSize);
                Tensor second = convOut.narrow(1, halfSize, halfSize);
                xs.add(first);
                h = second;
            } else {
                xs.add(convOut);
                h = convOut;
            }
        }

        TensorVector tensorVec = new TensorVector();
        for (Tensor t : xs) tensorVec.push_back(t);
        Tensor stacked = torch.cat(tensorVec, 1L);  // (batch, fcInputDim, embedDim)
        Tensor summed = stacked.sum(2L);              // (batch, fcInputDim)
        return fc.forward(summed);
    }
}
