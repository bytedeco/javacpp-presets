package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Dense (batched) Graph Isomorphism Network convolution.
 *
 * <pre>
 *   x' = MLP( (1 + ε) X + A X )
 * </pre>
 * Inputs {@code x [B,N,F]}, {@code adj [B,N,N]}. ε may be trainable.
 */
public class DenseGINConv extends MessagePassing {

    private final Module mlp;
    private final double epsInit;
    private final boolean trainEps;
    private final Parameter epsParam;

    public DenseGINConv(Module mlp) {
        this(mlp, 0.0, false);
    }

    public DenseGINConv(Module mlp, boolean trainEps) {
        this(mlp, 0.0, trainEps);
    }

    public DenseGINConv(Module mlp, double eps, boolean trainEps) {
        super("sum");
        if (mlp == null) {
            throw new IllegalArgumentException("mlp must not be null");
        }
        this.mlp = register_module("mlp", mlp);
        this.epsInit = eps;
        this.trainEps = trainEps;

        if (trainEps) {
            Tensor e = torch.tensor(new float[]{(float) eps},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.epsParam = new Parameter(e.clone().requires_grad_(true), true);
            register_parameter("eps", this.epsParam);
        } else {
            this.epsParam = null;
        }
    }

    /** Convenience for SequentialImpl demos. */
    public DenseGINConv(SequentialImpl mlp, boolean trainEps) {
        this((Module) mlp, 0.0, trainEps);
    }

    public DenseGINConv(SequentialImpl mlp, double eps, boolean trainEps) {
        this((Module) mlp, eps, trainEps);
    }

    /**
     * Dense forward. Second arg is adjacency {@code [B,N,N]}.
     * @param x   [B, N, F]
     * @param adj [B, N, N]
     */
    @Override
    public Tensor forward(Tensor x, Tensor adj) {
        if (x == null || adj == null) {
            throw new NullPointerException("x and adj must not be null");
        }
        if (x.dim() != 3 || adj.dim() != 3) {
            throw new IllegalArgumentException("x must be [B,N,C], adj [B,N,N]");
        }

        double epsVal;
        if (trainEps && epsParam != null) {
            epsVal = 1.0 + epsParam.item_double();
        } else {
            epsVal = 1.0 + epsInit;
        }

        Tensor out = x.mul(new Scalar(epsVal)).add(adj.matmul(x));
        return forwardMlp(out);
    }

    private Tensor forwardMlp(Tensor in) {
        if (mlp instanceof SequentialImpl) {
            return ((SequentialImpl) mlp).forward(in);
        }
        if (mlp instanceof LinearImpl) {
            return ((LinearImpl) mlp).forward(in);
        }
        return mlp.asSequential().forward(in);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    public Parameter getEpsParam() {
        return epsParam;
    }

    public Module getMlp() {
        return mlp;
    }

    /** Demo compatibility. */
    public SequentialImpl getMlpSequential() {
        return mlp instanceof SequentialImpl ? (SequentialImpl) mlp : null;
    }
}
