package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;

/**
 * Explicit name for the <em>guided-reward</em> PPO agent formerly only called
 * {@link GRPOAgent}.
 *
 * <p>This is <b>not</b> DeepSeek Group-Relative Policy Optimization — see
 * {@link GroupRelativePPOAgent} / {@link org.bytedeco.pytorch.llm.trl.GRPOTrainer}.
 *
 * <p>Binary-compatible: subclasses {@link GRPOAgent} with identical behaviour.
 */
public class GuidedRewardPPOAgent extends GRPOAgent {

    public GuidedRewardPPOAgent(AbstractActorCritic model,
                                Optimizer optimizer,
                                ReplayBuffer replayBuffer,
                                float clipEps,
                                float gamma,
                                float gaeLambda,
                                float guideRewardWeight,
                                float entropyCoeff) {
        super(model, optimizer, replayBuffer, clipEps, gamma, gaeLambda,
                guideRewardWeight, entropyCoeff);
    }

    public GuidedRewardPPOAgent(long stateDim, long actionDim,
                                AbstractActorCritic.ActionSpaceType actionSpaceType) {
        super(stateDim, actionDim, actionSpaceType);
    }

    public static GuidedRewardPPOAgent create(long stateDim, long actionDim,
                                              AbstractActorCritic.ActionSpaceType actionSpaceType) {
        return new GuidedRewardPPOAgent(stateDim, actionDim, actionSpaceType);
    }
}
