package org.bytedeco.pytorch.rl.critic;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.distribution.Normal;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Continuous Actor-Critic with a shared feature layer (64-d MLP).
 * Action space is {@link ActionSpaceType#CONTINUOUS} (Normal policy).
 */
public class ActorCriticNetwork extends AbstractActorCritic {
    private final LinearImpl fc1;
    private final LinearImpl actor;
    private final LinearImpl critic;
    /** Own storage — never store {@code register_parameter} ByRef return. */
    private final Tensor logStd;

    public ActorCriticNetwork(long stateDim, long actionDim) {
        super(stateDim, actionDim, ActionSpaceType.CONTINUOUS);
        fc1 = register_module("fc1", new LinearImpl(stateDim, 64));
        actor = register_module("actor", new LinearImpl(64, actionDim));
        critic = register_module("critic", new LinearImpl(64, 1));

        Tensor logStdInit = zeros(new long[]{actionDim},
                new TensorOptions().dtype(new ScalarTypeOptional(kFloat())))
                .contiguous().clone();
        logStdInit.requires_grad_(true);
        register_parameter("logStd", logStdInit, true);
        this.logStd = logStdInit;
    }

    @Override
    public Tensor getValue(Tensor state) {
        return forward_value(state);
    }

    @Override
    public Distribution getDistribution(Tensor state) {
        return forward_policy(state);
    }

    public Tensor evaluateValue(Tensor state) {
        return forward_value(state);
    }

    public Distribution forward_policy(Tensor state) {
        Tensor x = relu(fc1.forward(state));
        Tensor mu = actor.forward(x);
        return new Normal(mu, exp(logStd));
    }

    public Tensor forward_value(Tensor state) {
        Tensor x = relu(fc1.forward(state));
        return critic.forward(x);
    }

    @Override
    public void close() {
        // Module-owned resources.
    }
}
