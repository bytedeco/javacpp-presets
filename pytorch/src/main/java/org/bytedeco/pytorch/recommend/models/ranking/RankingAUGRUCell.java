/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DIEN.scala
 * (AUGRUCell + AUGRU ranking-local variants — different from basic.layers.AUGRU)
 *
 * Parameter-matrix AUGRU cell with xavier-uniform init (Python AUGRU_Cell mirror).
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;

/**
 * AUGRU gate cell. Mirrors Python's AUGRU_Cell with xavier-uniform parameters.
 * Named RankingAUGRUCell to avoid clash with basic.layers.AUGRU_Cell.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RankingAUGRUCell extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Tensor Wu, Uu, bu, Wr, Ur, br, Wh, Uh, bh;

    public RankingAUGRUCell(int embedDim) {
        super("AUGRUCell");
        this.Wu = initXavier(new long[]{embedDim, embedDim});
        register_parameter("Wu", Wu);
        this.Uu = initXavier(new long[]{embedDim, embedDim});
        register_parameter("Uu", Uu);
        this.bu = initXavier(new long[]{1L, embedDim});
        register_parameter("bu", bu);
        this.Wr = initXavier(new long[]{embedDim, embedDim});
        register_parameter("Wr", Wr);
        this.Ur = initXavier(new long[]{embedDim, embedDim});
        register_parameter("Ur", Ur);
        this.br = initXavier(new long[]{1L, embedDim});
        register_parameter("br", br);
        this.Wh = initXavier(new long[]{embedDim, embedDim});
        register_parameter("Wh", Wh);
        this.Uh = initXavier(new long[]{embedDim, embedDim});
        register_parameter("Uh", Uh);
        this.bh = initXavier(new long[]{1L, embedDim});
        register_parameter("bh", bh);
    }

    private static Tensor initXavier(long[] shape) {
        Tensor t = torch.zeros(shape,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        torch.xavier_uniform_(t);
        return t;
    }

    public Tensor forward(Tensor x, Tensor h1, Tensor a) {
        Tensor u = torch.sigmoid(x.matmul(Wu).add(h1.matmul(Uu)).add(bu));
        Tensor r = torch.sigmoid(x.matmul(Wr).add(h1.matmul(Ur)).add(br));
        Tensor hHat = torch.tanh(x.matmul(Wh).add(r.mul(h1.matmul(Uh))).add(bh));
        Tensor uHat = a.mul(u);
        return uHat.neg().add(new Scalar(1.0f)).mul(h1).add(uHat.mul(hHat));
    }
}
