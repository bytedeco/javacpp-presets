package org.bytedeco.pytorch.rl.critic;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.distribution.Normal;


import static org.bytedeco.pytorch.global.torch.*;

/**
 * Continuous-action Actor-Critic (Normal policy).
 *
 * <p>Previously mislabeled as {@code DISCRETE} while returning {@link Normal} —
 * that broke A2C/GRPO simplified constructors that branch on action space type.
 */
public class ActorCritic extends AbstractActorCritic {
    private final LinearImpl actor;
    private final LinearImpl critic;
    /** Learnable log-std; keep original handle (do not store register_parameter ByRef). */
    private final Tensor logStd;

    public ActorCritic(long stateDim, long actionDim) {
        super(stateDim, actionDim, ActionSpaceType.CONTINUOUS);
        actor = register_module("actor", new LinearImpl(stateDim, actionDim));
        critic = register_module("critic", new LinearImpl(stateDim, 1));
        // Own storage, then register for Module.parameters() discovery only.
        Tensor logStdInit = zeros(new long[]{actionDim},
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())))
                .contiguous().clone();
        logStdInit.requires_grad_(true);
        register_parameter("log_std", logStdInit, true);
        this.logStd = logStdInit;
    }

    @Override
    public Distribution getDistribution(Tensor state) {
        Tensor mu = tanh(actor.forward(state)); // mean in [-1, 1]
        Tensor std = exp(logStd);
        return new Normal(mu, std);
    }

    @Override
    public Tensor getValue(Tensor state) {
        return critic.forward(state);
    }

    @Override
    public void close() {
        // Modules/parameters owned by Module graph; no extra native handles to free.
    }
}