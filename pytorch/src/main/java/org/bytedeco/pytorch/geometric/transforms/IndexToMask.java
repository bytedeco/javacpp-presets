package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kBool;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * IndexToMask: convert index tensors to boolean masks.
 *
 * <p>Reads optional {@code train_indices} / {@code val_indices} / {@code test_indices}
 * and writes {@code train_mask} / {@code val_mask} / {@code test_mask}.
 *
 * <p>Bool fills use {@code index_fill_(..., Scalar(1))} — {@code tensor(true, boolOpts)}
 * is not implemented for Bool dtype on CPU in libtorch Java bindings.
 */
public class IndexToMask implements BaseTransform {
    private final long size;

    public IndexToMask(long size) {
        if (size <= 0) {
            throw new IllegalArgumentException("size must be > 0, got " + size);
        }
        this.size = size;
    }

    @Override
    public GraphData apply(GraphData data) {
        TransformUtils.requireData(data);
        convert(data, "train_indices", "train_mask");
        convert(data, "val_indices", "val_mask");
        convert(data, "test_indices", "test_mask");
        return data;
    }

    private void convert(GraphData data, String idxKey, String maskKey) {
        Tensor indices = data.get(idxKey);
        if (indices == null || !indices.defined()) {
            return;
        }
        Tensor ref = data.x != null ? data.x : indices;
        Tensor mask = zeros(new long[]{size},
                ref.options().dtype(new ScalarTypeOptional(kBool())));
        mask.index_fill_(0,
                indices.to(org.bytedeco.pytorch.global.torch.ScalarType.Long),
                new Scalar(1));
        data.put(maskKey, mask);
    }
}
