package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

/**
 * Base class for aggregation operators used by MessagePassing.
 *
 * <p>Isolated-node policy (PyG-aligned, implemented by {@code AggrUtils}):
 * <ul>
 *   <li>sum / mean → 0</li>
 *   <li>max / min → 0 after reduce</li>
 *   <li>mul / prod → 1</li>
 * </ul>
 *
 * <p>Implement {@link #forward(Tensor, Tensor, long)}. The CSR {@code ptr} overload
 * defaults to ignoring {@code ptr} and calling the index-based path.
 */
public abstract class Aggregation extends Module {

    public Aggregation() {
        super();
    }

    /**
     * @param x       edge/message features [E, F...]
     * @param index   target node index per edge [E]
     * @param dimSize number of target nodes N
     * @return aggregated features [N, F...]
     */
    public abstract Tensor forward(Tensor x, Tensor index, long dimSize);

    /**
     * CSR-style aggregation. Default ignores {@code ptr} and uses {@code index}.
     * Subclasses may override for segment_reduce backends.
     */
    public Tensor forward(Tensor x, Tensor index, Tensor ptr, long dimSize) {
        return forward(x, index, dimSize);
    }
}
