package org.bytedeco.pytorch.geometric.utils;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.TensorExampleVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public class ExampleToolkit {

    // 将 Vector 中的特征张量堆叠为一个批次张量
    public static Tensor stackData(ExampleVector batch) {
        TensorVector tv = new TensorVector();
        for (long i = 0; i < batch.size(); i++) {
            tv.push_back(batch.get(i).data());
        }
        return torch.stack(tv);
    }

    // 将 Vector 中的标签张量堆叠为一个批次张量
    public static Tensor stackTarget(ExampleVector batch) {
        TensorVector tv = new TensorVector();
        for (long i = 0; i < batch.size(); i++) {
            tv.push_back(batch.get(i).target());
        }
        return torch.stack(tv);
    }

    public static Tensor stackExampleData(ExampleVector batch) {
        TensorVector tensors = new TensorVector();
        for (long i = 0; i < batch.size(); i++) {
            tensors.push_back(batch.get(i).data());
        }
        return torch.stack(tensors);
    }

    public static Tensor stackTensorExampleData(TensorExampleVector batch) {
        TensorVector tensors = new TensorVector();
        for (long i = 0; i < batch.size(); i++) {
            tensors.push_back(batch.get(i).data());
        }
        return torch.stack(tensors);
    }

    public static Tensor stackExampleTarget(ExampleVector batch) {
        TensorVector tensors = new TensorVector();
        for (long i = 0; i < batch.size(); i++) {
            tensors.push_back(batch.get(i).target());
        }
        return torch.stack(tensors);
    }
    
    
}
