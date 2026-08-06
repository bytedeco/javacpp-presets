package example;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.options.*;

//import org.bytedeco.pytorch.ASMoutput;

public class RNNTest {
    public static void main(String[] args) {
        Loader.load(torch.class);
        try {
            RNNImpl rnn = new RNNImpl(new RNNOptions(2L, 4L).num_layers(1));
            Tensor x = torch.randn(new long[]{1L, 1L, 2L});
            T_TensorTensor_T rnnOut = rnn.forwardT_TensorTensor_T(x);
            System.out.println("[rnn] OK, out0 sum=" + rnnOut.get0().sum().item_double() + ", out1 sum=" + rnnOut.get1().sum().item_double());
        } catch (Throwable t) {
            System.out.println("[rnn] FAIL: " + t);
        }
        try {
            GRUImpl gru = new GRUImpl(new GRUOptions(2L, 4L).num_layers(1));
            Tensor x = torch.randn(new long[]{1L, 1L, 2L});
            T_TensorTensor_T gruOut = gru.forwardT_TensorTensor_T(x);
            System.out.println("[gru] OK, out0 sum=" + gruOut.get0().sum().item_double());
        } catch (Throwable t) {
            System.out.println("[gru] FAIL: " + t);
        }
        try {
            LSTMImpl lstm = new LSTMImpl(new LSTMOptions(2L, 4L).num_layers(1));
            Tensor x = torch.randn(new long[]{1L, 1L, 2L});
            T_TensorT_TensorTensor_T_T lstmOut = lstm.forwardT_TensorT_TensorTensor_T_T(x);
            System.out.println("[lstm] OK, out0 sum=" + lstmOut.get0().sum().item_double());
        } catch (Throwable t) {
            System.out.println("[lstm] FAIL: " + t);
        }
        try {
            LSTMCellImpl cell = new LSTMCellImpl(new LSTMCellOptions(2L, 4L));
            Tensor input = torch.randn(new long[]{1L, 2L});
            Tensor hx = torch.randn(new long[]{1L, 4L});
            Tensor cx = torch.randn(new long[]{1L, 4L});
            // C++ forward(input, std::optional<Tensor> hx={}, std::optional<Tensor> cx={})
            // collapses 3 args + 2 defaults to forward(input, T_TensorTensor_T).
            T_TensorTensor_T out = cell.forward(input, new T_TensorTensor_T(hx, cx));
            System.out.println("[lstmcell] OK, h=" + out.get0().sum().item_double() + ", c=" + out.get1().sum().item_double());
        } catch (Throwable t) {
            System.out.println("[lstmcell] FAIL: " + t);
        }
        try {
            MultiheadAttentionImpl mha = new MultiheadAttentionImpl(new MultiheadAttentionOptions(8L, 1));
            Tensor q = torch.randn(new long[]{1L, 1L, 8L});
            T_TensorTensor_T mhaOut = mha.forwardT_TensorTensor_T(q, q, q);
            System.out.println("[mha] OK, attn sum=" + mhaOut.get0().sum().item_double());
        } catch (Throwable t) {
            System.out.println("[mha] FAIL: " + t);
        }
//        try {
//            LongVector cutoffs = new LongVector(3L);
//            cutoffs.push_back(2L);
//            cutoffs.push_back(2L);
//            cutoffs.push_back(2L);
//            AdaptiveLogSoftmaxWithLossImpl asl = new AdaptiveLogSoftmaxWithLossImpl(
//                new AdaptiveLogSoftmaxWithLossOptions(4L, 3L, cutoffs));
//            Tensor input = torch.randn(new long[]{2L, 4L});
//            Tensor target = torch.randint(0L, 3L, new long[]{2L}, new TensorOptions().dtype(new ScalarTypeOptional(torch.kLong())));
//            ASMoutput out = asl.forwardASMoutput(input, target);
//            System.out.println("[alsm] OK, output sum=" + out.output().sum().item_double());
//        } catch (Throwable t) {
//            System.out.println("[alsm] FAIL: " + t);
//            t.printStackTrace();
//        }
        try {
            SequentialImpl seq = new SequentialImpl();
            seq.push_back(new LinearImpl(8L, 4L));
            seq.push_back(new DropoutImpl(0.5));
            seq.push_back(new LinearImpl(4L, 2L));
            Tensor x = torch.randn(new long[]{1L, 8L});
            Tensor y = seq.forward(x);
            System.out.println("[seq+dropout] OK, sum=" + y.sum().item_double());
        } catch (Throwable t) {
            System.out.println("[seq+dropout] FAIL: " + t);
        }
    }
}
