package org.bytedeco.pytorch.rl;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;
import org.bytedeco.pytorch.rl.env.SimpleTradingEnv;
import org.bytedeco.pytorch.rl.trainer.GRPOTrainer;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Minimal trading demo: {@link SimpleTradingEnv} + discrete policy +
 * group-relative {@link GRPOTrainer}.
 *
 * <p>Raw step rewards are stored in the buffer <em>return</em> slot so
 * {@link GRPOTrainer#trainBatch} can treat the episode as one group and
 * normalize advantages in-group (DeepSeek-style GRPO).
 */
public class TradingSystem {
    public static void main(String[] args) {
        // 1. Mock prices (sine + noise)
        float[] mockPrices = new float[200];
        for (int i = 0; i < 200; i++) {
            mockPrices[i] = (float) (100 + 10 * Math.sin(i * 0.1) + Math.random() * 2);
        }

        SimpleTradingEnv env = new SimpleTradingEnv(mockPrices);
        // 5-day window observation, 3 discrete actions
        CartPoleActorCritic model = new CartPoleActorCritic(5, 3);
        ReplayBuffer buffer = new ReplayBuffer();
        // groupSize = episode length is unknown a priori → trainer falls back to
        // whole-batch group when numel % groupSize != 0
        GRPOTrainer trainer = new GRPOTrainer(model, 1e-4, 0.2, 4);

        int episodes = args != null && args.length > 0 ? Integer.parseInt(args[0]) : 20;
        for (int episode = 0; episode < episodes; episode++) {
            float[] s = env.reset();
            try (PointerScope scope = new PointerScope()) {
                while (true) {
                    Tensor stateTensor = tensor(s).reshape(1, 5);
                    Categorical dist = model.forward_policy(stateTensor);
                    Tensor actionTensor = dist.sample();
                    Tensor logp = dist.log_prob(actionTensor);

                    StepResult res = env.step((int) actionTensor.item_long());
                    // push(s, a, lp, adv, ret): adv unused by GRPOTrainer (uses returns
                    // as group scores); store reward in the return slot.
                    buffer.push(
                            stateTensor.squeeze(0),
                            actionTensor.reshape(-1),
                            logp.reshape(-1),
                            zeros(new long[]{1}),
                            scalar_tensor(new Scalar(res.reward),
                                    new TensorOptions().dtype(new ScalarTypeOptional(kFloat())))
                                    .reshape(1));

                    s = res.nextState;
                    if (res.done) break;
                }
                trainer.trainBatch(buffer);
                buffer.clear();
                System.out.println("Episode " + episode + " trading train step done");
            }
        }
    }
}
