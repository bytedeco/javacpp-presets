/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/InputMask.scala
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Return input masks from features.
 *
 * <p>Shape
 * <ul>
 *   <li>Input x: map feature_name → feature_value; sequence {@code (B, L)}, sparse/dense {@code (B,)}</li>
 *   <li>features: only SparseFeature or SequenceFeature</li>
 *   <li>Output Sparse: {@code (B, num_features)}; Sequence: {@code (B, num_seq, seq_length)}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class InputMask extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public InputMask() {
        super("InputMask");
    }

    public Tensor forward(Map<String, Tensor> x, List<? extends Feature> features) {
        List<Tensor> mask = new ArrayList<>();

        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                SparseFeature sf = (SparseFeature) f;
                Long paddingIdx = sf.paddingIdx();
                Tensor feaMask;
                if (paddingIdx != null && paddingIdx >= 0) {
                    feaMask = x.get(sf.name()).ne(new Scalar(paddingIdx));
                } else {
                    feaMask = x.get(sf.name()).ne(new Scalar(-1L));
                }
                mask.add(feaMask.unsqueeze(1).toType(ScalarType.Float));
            } else if (f instanceof SequenceFeature) {
                SequenceFeature seqf = (SequenceFeature) f;
                long paddingIdx = seqf.paddingIdx();
                Tensor feaMask;
                if (paddingIdx >= 0) {
                    feaMask = x.get(seqf.name()).ne(new Scalar(paddingIdx));
                } else {
                    feaMask = x.get(seqf.name()).ne(new Scalar(-1L));
                }
                mask.add(feaMask.unsqueeze(1).toType(ScalarType.Float));
            } else {
                throw new IllegalArgumentException(
                        "Only SparseFeature or SequenceFeature support to get mask.");
            }
        }

        TensorVector vec = new TensorVector();
        for (Tensor t : mask) {
            vec.push_back(t);
        }
        return torch.cat(vec, 1L);
    }
}
