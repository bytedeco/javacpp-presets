package org.bytedeco.pytorch.rl.critic;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.distribution.Normal;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 大模型微调专用ActorCritic（支持离散/连续动作空间）
 * 离散：文本生成（token级别的Categorical分布）
 * 连续：嵌入空间优化（Normal分布）
 */
public class LMActorCritic extends AbstractActorCritic {
    // 大模型主干（可替换为LLM的Transformer层）
    private final LinearImpl lmBackbone;
    // 策略头（Actor）：离散=token分类，连续=嵌入优化
    private final LinearImpl actorHead;
    // 价值头（Critic）：奖励/偏好评分
    private final LinearImpl criticHead;
    // 连续动作空间的标准差（可学习）
    private final Tensor logStd;
    // 动作空间类型：discrete/categorical 或 continuous/normal
    private final String actionSpaceType;

    /**
     * 构造函数
     * @param hiddenDim 大模型隐藏层维度
     * @param actionDim 动作维度（离散=vocab_size，连续=embedding_dim）
     * @param actionSpaceType 动作空间类型："discrete" 或 "continuous"
     */
    public LMActorCritic(long hiddenDim, long actionDim, String actionSpaceType) {
        super(hiddenDim, actionDim,
                "continuous".equalsIgnoreCase(actionSpaceType)
                        ? ActionSpaceType.CONTINUOUS
                        : ActionSpaceType.DISCRETE);
        this.actionSpaceType = actionSpaceType == null ? "discrete" : actionSpaceType.toLowerCase();

        // 大模型主干（简化版，实际替换为Transformer）
        this.lmBackbone = register_module("lm_backbone", new LinearImpl(hiddenDim, hiddenDim));
        // 策略头
        this.actorHead = register_module("actor_head", new LinearImpl(hiddenDim, actionDim));
        // 价值头
        this.criticHead = register_module("critic_head", new LinearImpl(hiddenDim, 1));

        // 连续动作空间初始化标准差（own storage; never store register_parameter ByRef）
        if ("continuous".equals(this.actionSpaceType)) {
            Tensor logStdInit = zeros(new long[]{actionDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(kFloat())))
                    .contiguous().clone();
            logStdInit.requires_grad_(true);
            register_parameter("log_std", logStdInit, true);
            this.logStd = logStdInit;
        } else {
            this.logStd = null;
        }
    }

    /**
     * 获取动作分布（核心：使用已实现的Categorical/Normal分布）
     * @param hiddenStates 大模型隐藏状态 [batch, seq_len, hidden_dim]
     * @return 动作分布
     */
    @Override
    public Distribution getDistribution(Tensor hiddenStates) {
        // 1. 大模型前向传播
        Tensor x = relu(lmBackbone.forward(hiddenStates));

        // 2. 策略头输出
        Tensor logits = actorHead.forward(x);

        // 3. 根据动作空间返回对应分布
        if ("discrete".equals(actionSpaceType)) {
            // 离散：文本生成，Categorical分布（token概率）
            return new Categorical(softmax(logits, -1));
        } else if ("continuous".equals(actionSpaceType)) {
            // 连续：嵌入优化，Normal分布（均值+标准差）
            Tensor mu = tanh(logits); // 限制均值在[-1,1]
            Tensor std = exp(logStd);
            return new Normal(mu, std);
        } else {
            throw new IllegalArgumentException("不支持的动作空间类型：" + actionSpaceType);
        }
    }

    /**
     * 获取价值评分（Critic）
     * @param hiddenStates 大模型隐藏状态
     * @return 价值评分 [batch, 1]
     */
    public Tensor getValue(Tensor hiddenStates) {
        Tensor x = relu(lmBackbone.forward(hiddenStates));
        return criticHead.forward(x);
    }
    

    /**
     * 大模型前向传播（获取隐藏状态）
     * @param inputEmbeds 输入嵌入 [batch, seq_len, hidden_dim]
     * @return 隐藏状态
     */
    public Tensor forwardLM(Tensor inputEmbeds) {
        return relu(lmBackbone.forward(inputEmbeds));
    }

    // 资源释放
    @Override
    public void close() {
//        super.close();
        if (logStd != null && !logStd.isNull()) {
            logStd.close();
        }
    }
}