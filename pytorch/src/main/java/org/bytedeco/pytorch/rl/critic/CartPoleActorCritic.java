package org.bytedeco.pytorch.rl.critic;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Distribution;

import static org.bytedeco.pytorch.global.torch.relu;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * CartPole 专用 ActorCritic（离散动作空间）适配抽象父类
 */
public class CartPoleActorCritic extends AbstractActorCritic {
    private final LinearImpl fc, actor, critic;

    public CartPoleActorCritic(long stateDim, long actionDim) {
        // 调用抽象父类构造函数（离散动作空间）
        super(stateDim, actionDim, ActionSpaceType.DISCRETE);

        fc = register_module("fc", new LinearImpl(stateDim, 128));
        actor = register_module("actor", new LinearImpl(128, actionDim));
        critic = register_module("critic", new LinearImpl(128, 1));
    }

    @Override
    public Distribution getDistribution(Tensor state) {
        return forward_policy(state);
    }

    @Override
    public Tensor getValue(Tensor state) {
        return forward_value(state);
    }

    // 原有 forward_policy 方法
    public Categorical forward_policy(Tensor state) {
        Tensor x = relu(fc.forward(state));
        Tensor logits = actor.forward(x);
        // Numerically stable: amax over last dim + clamp to avoid overflow after bad updates.
        Tensor stable = logits.sub(logits.amax(new long[]{-1}, true))
                .clamp(new ScalarOptional(new Scalar(-20.0)), new ScalarOptional(new Scalar(20.0)));
        return new Categorical(softmax(stable, -1));
    }

    // 原有 forward_value 方法
    public Tensor forward_value(Tensor state) {
        Tensor x = relu(fc.forward(state));
        return critic.forward(x);
    }

    @Override
    public void close() {
//        super.close();
        if (fc != null) fc.close();
        if (actor != null) actor.close();
        if (critic != null) critic.close();
    }
}
//public class CartPoleActorCritic extends Module {
//    
//    private final LinearImpl fc, actor, critic;
//
//    public CartPoleActorCritic(long stateDim, long actionDim) {
//        fc = register_module("fc", new LinearImpl(stateDim, 128));
//        actor = register_module("actor", new LinearImpl(128, actionDim));
//        critic = register_module("critic",new LinearImpl(128, 1));
//    }
//
//    // 返回动作分布
//    public Categorical forward_policy(Tensor state) {
//        Tensor x = relu(fc.forward(state));
//        Tensor logits = actor.forward(x);
//        // 使用 Softmax 转化为概率，传给我们在前面章节实现的 Categorical 类
//        return new Categorical(softmax(logits, -1));
//    }
//
//    // 返回状态价值 V(s)
//    public Tensor forward_value(Tensor state) {
//        Tensor x = relu(fc.forward(state));
//        return critic.forward(x);
//    }
//}