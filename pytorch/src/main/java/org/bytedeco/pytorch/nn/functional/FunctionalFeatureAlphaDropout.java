package org.bytedeco.pytorch.nn.functional;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FunctionalFeatureAlphaDropout extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float p;

    public FunctionalFeatureAlphaDropout(float p) {
        super("FunctionalFeatureAlphaDropout");
        this.p = p;
    }

    @Override
    public Tensor forward(Tensor x) {
        if (x == null) throw new IllegalArgumentException("Input tensor cannot be null");
        // Use functional torch.alpha_dropout to avoid native AlphaDropoutImpl overload dispatch issues.
        return torch.feature_alpha_dropout(x, p, this.is_training());
    }
}
