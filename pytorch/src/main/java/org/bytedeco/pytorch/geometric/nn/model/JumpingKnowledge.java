package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.options.LSTMOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
/**
 * Jumping Knowledge: 融合多层 GNN 的输出
 * mode: "cat", "max", "lstm"
 */
public class JumpingKnowledge extends Module {
    private String mode;
    private LSTMImpl lstm;

    public JumpingKnowledge(String mode, long channels, int numLayers) {
        this.mode = mode;
        if ("lstm".equals(mode)) {
            // Bi-LSTM
            LSTMOptions opts = new LSTMOptions(channels, channels / 2);
            opts.bidirectional().put(true);
            opts.batch_first().put(true);
            this.lstm = new LSTMImpl(opts);
            register_module("lstm", lstm);
        }
    }

    public Tensor forward(List<Tensor> xs) {
        // xs: [x_0, x_1, ..., x_k]
        if ("cat".equals(mode)) {
            TensorVector vec = new TensorVector();
            for(Tensor t : xs) vec.put(t);
            return torch.cat(vec, 1);
        } else if ("max".equals(mode)) {
            Tensor stack = torch.stack(new TensorVector(xs.toArray(new Tensor[0])), 0);
            return stack.max(0).get0();
        } else if ("lstm".equals(mode)) {
            // [NumLayers, N, C] -> permute -> [N, NumLayers, C]
            Tensor stack = torch.stack(new TensorVector(xs.toArray(new Tensor[0])), 0);
            stack = stack.permute(1, 0, 2);
            T_TensorT_TensorTensor_T_T ret = lstm.forwardT_TensorT_TensorTensor_T_T(stack);
            Tensor out = ret.get0(); // [N, NumLayers, C]
            // Take last step? JK usually takes the representation learned by LSTM
            // PyG: max of LSTM outputs or last step
            return out.select(1, -1);
        }
        return xs.get(xs.size()-1);
    }
}

