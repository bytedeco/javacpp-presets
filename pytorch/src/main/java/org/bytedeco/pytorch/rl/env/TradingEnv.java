package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.rl.StepResult2;

import static org.bytedeco.pytorch.global.torch.tensor;

public class TradingEnv {
    private final float[] prices;       // 原始价格序列 (Close)
    private final float[][] features;   // 预处理后的特征 (如 RSI, ROC)
    private int currentIndex;
    private final int maxSteps;
    private double entryPrice = 0.0; // <--- 必须在这里定义！
    private double balance = 10000.0;   // 初始资金
    private double shares = 0;          // 持仓数
    private final double feeRate = 0.0001; // 万一手续费

    public Tensor getCurrentStateTensor() {
        float[] rawObs = features[currentIndex];
        float[] obsWithState = new float[rawObs.length + 1];
        System.arraycopy(rawObs, 0, obsWithState, 0, rawObs.length);
        obsWithState[rawObs.length] = (shares > 0) ? 1.0f : -1.0f;
        return tensor(obsWithState).unsqueeze(0);
    }

    public Tensor getCurrentStateTensor2() {
        // 这里的 Tensor.fromArray 是基础算子，不会产生递归
        // 返回 [1, FeatureDim]
        return tensor(this.features[currentIndex]).unsqueeze(0);
    }
    public TradingEnv(float[] prices, float[][] features) {
        this.prices = prices;
        this.features = features;
        this.maxSteps = prices.length - 1;
        this.currentIndex = 20;
    }

    /**
     * 执行交易环境的一步。
     * @param action 模型的预测值（通常在 -2.0 到 2.0 之间）
     * @return 包含下一步观察值、奖励和结束标志的结果对象
     */
    public StepResult2 step(float action) {
        // 1. 获取当前和下一步的价格数据
        double currentPrice = prices[currentIndex];
        double nextPrice = prices[currentIndex + 1];

        // 2. 记录操作前的账户价值
        double prevNetWorth = balance + (shares * currentPrice);

        // 3. 交易决策标志
        boolean isTradeExecuted = false;

        // --- 4. 核心交易逻辑（带阈值和持仓检查） ---
        // 动作 > 0.5: 强烈看多 | 动作 < -0.5: 强烈看空 | 其余: 观望或持仓不动
        if (action > 0.5) {
            if (shares == 0) { // 只有在空仓时才允许买入（开仓）
                double fee = balance * feeRate;
                shares = (balance - fee) / currentPrice;
                balance = 0;
                this.entryPrice = currentPrice; // 记录买入成本
                isTradeExecuted = true;
            }
        } else if (action < -0.5) {
            if (shares > 0) { // 只有在持仓时才允许卖出（平仓）
                double revenue = shares * currentPrice;
                double fee = revenue * feeRate;
                balance = revenue - fee;
                shares = 0;
                this.entryPrice = 0; // 平仓后重置成本
                isTradeExecuted = true;
            }
        }

        // --- 5. 环境状态更新 ---
        currentIndex++;
        boolean done = (currentIndex >= prices.length - 1);

        // --- 6. 奖励函数设计 (Reward Shaping) ---
        // 计算新的账户净值（基于下一步的价格波动）
        double nextNetWorth = balance + (shares * nextPrice);
        double stepReturn = (nextNetWorth - prevNetWorth) / prevNetWorth;

        // a. 基础奖励：净值增长的百分比，放大 250 倍让信号更明显
        double reward = stepReturn * 250.0;

        // b. 交易惩罚：解决“多动症”的关键。每次开平仓固定扣分，过滤掉微小波动的诱惑
        if (isTradeExecuted) {
            reward -= 0.1;
        }

        // c. 止损诱导惩罚：如果持仓且价格低于入场价，直接给负反馈
        if (shares > 0 && nextPrice < entryPrice) {
            double lossRate = (nextPrice - entryPrice) / entryPrice;
            reward += lossRate * 100.0; // 亏损越多罚得越重
        }

        // d. 时间惩罚：如果持仓超过很久价格没动，每步扣除微小分数，防止 Agent 变懒
        if (shares > 0 && Math.abs(nextPrice - currentPrice) < 0.0001) {
            reward -= 0.005;
        }

        // --- 7. 构造 6 维观察值 (Observation) ---
        float[] rawObs = features[currentIndex];
        float[] obsWithAccount = new float[6];

        // 前 4 维：原始价格特征（RSI, MA 等）
        System.arraycopy(rawObs, 0, obsWithAccount, 0, 4);

        // 第 5 维：当前持仓状态（1.0 有货, -1.0 没货）
        obsWithAccount[4] = (shares > 0) ? 1.0f : -1.0f;

        // 第 6 维：浮动盈亏 (PnL)。Agent 会以此学习“亏了得跑，赚了要拿”
        double currentPnL = (shares > 0) ? (currentPrice - entryPrice) / entryPrice : 0;
        obsWithAccount[5] = (float) currentPnL * 10.0f; // 放大 10 倍增强感官

        // --- 8. 返回结果 ---
        Tensor observation = tensor(obsWithAccount).unsqueeze(0);
        return new StepResult2(observation, (float) reward, done);
    }
    public StepResult2 step7(float action) {
        // --- 1. 基础状态准备 ---
        double currentPrice = prices[currentIndex];
        double nextPrice = prices[currentIndex + 1]; // 预看下一步用于算奖励
        double prevNetWorth = balance + (shares * currentPrice);

        // 记录入场价格（建议在类成员变量中定义 double entryPrice）
        // 如果没有持仓，entryPrice 应为 0

        boolean tradeExecuted = false;

        // --- 2. 交易触发逻辑 (降低阈值并增加对称性) ---
        // 动作: > 0.4 买入, < -0.4 卖出, 之间维持
        if (action > 0.4) {
            if (shares == 0) { // 开仓
                double fee = balance * feeRate;
                shares = (balance - fee) / currentPrice;
                balance = 0;
                this.entryPrice = currentPrice; // 记录入场价
                tradeExecuted = true;
            }
        } else if (action < -0.4) {
            if (shares > 0) { // 平仓
                double revenue = shares * currentPrice;
                double fee = revenue * feeRate;
                balance = revenue - fee;
                shares = 0;
                this.entryPrice = 0; // 重置入场价
                tradeExecuted = true;
            }
        }

        // --- 3. 时间推进 ---
        currentIndex++;
        boolean done = (currentIndex >= prices.length - 1);

        // --- 4. 奖励函数设计 (关键在于止损诱导) ---
        double currentNetWorth = balance + (shares * nextPrice);
        double stepReturn = (currentNetWorth - prevNetWorth) / prevNetWorth;

        // a. 收益奖励 (适当放大)
        double reward = stepReturn * 250.0;

        // b. 交易惩罚 (稍微降低，避免 Agent 过于胆小)
        if (tradeExecuted) {
            reward -= 0.05;
        }

        // c. 止损惩罚 (如果持仓且价格低于入场价，额外给负分)
        if (shares > 0 && nextPrice < entryPrice) {
            double lossFromEntry = (nextPrice - entryPrice) / entryPrice;
            reward += lossFromEntry * 50.0; // 亏损越多，罚得越狠
        }

        // d. 空仓奖励 (如果行情下跌 Agent 没买，给大奖)
        if (shares == 0 && nextPrice < currentPrice) {
            reward += 0.02;
        }

        // --- 5. 构造 6 维观察值 (Observation) ---
        float[] rawObs = features[currentIndex];
        float[] obsWithState = new float[6]; // 增加到 6 维
        System.arraycopy(rawObs, 0, obsWithState, 0, 4);

        // 第 5 维：持仓状态
        obsWithState[4] = (shares > 0) ? 1.0f : -1.0f;

        // 第 6 维：当前浮动盈亏 (PnL)
        // 这是 Agent 学会止损的“眼睛”
        double currentPnL = (shares > 0) ? (currentPrice - entryPrice) / entryPrice : 0;
        obsWithState[5] = (float) currentPnL * 10.0f; // 放大 10 倍，增强神经网络感知力

        // --- 6. 返回结果 ---
        return new StepResult2(
                tensor(obsWithState).unsqueeze(0),
                (float) reward,
                done
        );
    }
    public StepResult2 step6(float action) {
        // --- 1. 初始化当前步数据 ---
        double currentPrice = prices[currentIndex];
        double prevNetWorth = balance + (shares * currentPrice);
        boolean isTradeExecuted = false;

        // --- 2. 执行交易逻辑 (引入阈值与手续费) ---
        // 动作定义: > 0.6 买入并持有, < -0.6 卖出并空仓, 之间则维持现状
        if (action > 0.6) {
            if (shares == 0) { // 开多仓
                double fee = balance * feeRate;
                shares = (balance - fee) / currentPrice;
                balance = 0;
                isTradeExecuted = true;
            }
        } else if (action < -0.6) {
            if (shares > 0) { // 平仓
                double revenue = shares * currentPrice;
                double fee = revenue * feeRate;
                balance = revenue - fee;
                shares = 0;
                isTradeExecuted = true;
            }
        }

        // --- 3. 时间推进 ---
        currentIndex++;
        boolean done = (currentIndex >= prices.length - 1);
        double nextPrice = prices[currentIndex];

        // --- 4. 计算即时收益与奖励 (Reward Shaping) ---
        double nextNetWorth = balance + (shares * nextPrice);

        // a. 基础奖励：净值变动的百分比 (放大 200 倍让梯度更明显)
        double stepReturn = (nextNetWorth - prevNetWorth) / prevNetWorth;
        double reward = stepReturn * 200.0;

        // b. 惩罚项：抑制过度交易 (多动症的解药)
        if (isTradeExecuted) {
            // 每次交易扣除 0.1 分。模型必须认为后续涨幅能覆盖这 0.1 才会交易
            reward -= 0.1;
        }

        // c. 惩罚项：抑制盲目持仓 (长拿不放的解药)
        if (shares > 0 && nextPrice <= currentPrice) {
            // 如果持仓但价格不涨或下跌，额外扣一点分，鼓励 Agent 寻找更有爆发力的入场点
            reward -= 0.02;
        }

        // d. 奖励项：鼓励空仓避险
        if (shares == 0 && nextPrice < currentPrice) {
            // 如果行情下跌而 Agent 选择了空仓，给予正奖励
            reward += 0.01;
        }

        // --- 5. 构造 5 维观察值 (Observation) ---
        // 特征：[原有4个指标] + [当前持仓状态]
        float[] rawObs = features[currentIndex];
        float[] obsWithState = new float[5];
        System.arraycopy(rawObs, 0, obsWithState, 0, 4);

        // 持仓状态：1.0 代表持仓，-1.0 代表空仓。这能帮模型学会“我有货，该考虑卖了”
        obsWithState[4] = (shares > 0) ? 1.0f : -1.0f;

        // --- 6. 封装结果 ---
        Tensor observation = tensor(obsWithState).unsqueeze(0);

        return new StepResult2(
                observation,
                (float) reward,
                done
        );
    }
    public StepResult2 step5(float action) {
        double currentPrice = prices[currentIndex];
        double prevNetWorth = balance + (shares * currentPrice);

        // 标记本步是否发生了实际的仓位变动
        boolean isTradeExecuted = false;

        // 1. 执行交易逻辑
        if (action > 0.8) { // 强烈看多门槛
            if (shares == 0) { // 只有当前空仓才买入
                double cost = balance * feeRate;
                shares = (balance - cost) / currentPrice;
                balance = 0;
                isTradeExecuted = true; // 触发买入交易
            }
        } else if (action < -0.8) { // 强烈看空门槛
            if (shares > 0) { // 只有当前持仓才卖出
                double revenue = shares * currentPrice;
                double cost = revenue * feeRate;
                balance = revenue - cost;
                shares = 0;
                isTradeExecuted = true; // 触发卖出交易
            }
        }

        // 2. 时间步进
        currentIndex++;
        boolean done = (currentIndex >= prices.length - 1);
        double nextPrice = prices[currentIndex];

        // 3. 计算操作后的净值及收益
        double nextNetWorth = balance + (shares * nextPrice);
        double profitRate = (nextNetWorth - prevNetWorth) / prevNetWorth;

        // 4. 核心奖励函数设计 (Reward Shaping)
        // 基础收益奖励 (放大 100 倍)
        double reward = profitRate * 100.0;

        // --- 惩罚项：治疗“多动症”的关键 ---
        if (isTradeExecuted) {
            // 每次动作扣除 0.05 的奖励分
            // 这样模型只有在预期收益远大于 0.05 时才会选择交易
            reward -= 0.05;
        }

        // --- 惩罚项：治疗“盲目持仓” ---
        if (shares > 0 && nextPrice < currentPrice) {
            // 持仓时价格下跌，给予额外负反馈
            reward -= 0.01;
        }

        // 5. 封装观察值 (5维：4指标 + 1持仓状态)
        float[] rawObs = features[currentIndex];
        float[] obsWithState = new float[5];
        System.arraycopy(rawObs, 0, obsWithState, 0, 4);
        obsWithState[4] = (shares > 0) ? 1.0f : -1.0f;

        return new StepResult2(
                tensor(obsWithState).unsqueeze(0),
                (float) reward,
                done
        );
    }
    public StepResult2 step4(float action) {
        // 1. 计算操作前的净值
        double currentPrice = prices[currentIndex];
        double prevNetWorth = balance + (shares * currentPrice);

        // 2. 执行动作 (逻辑：> 0.5 买入, < -0.5 卖出, 其余持有/空仓)
        // 注意：这里为了让模型更容易学习，采用了简单的全仓逻辑
        if (action > 0.5) {
            // 买入信号
            if (shares == 0) {
                double cost = balance * feeRate;
                shares = (balance - cost) / currentPrice;
                balance = 0;
            }
        } else if (action < -0.5) {
            // 卖出信号
            if (shares > 0) {
                double revenue = shares * currentPrice;
                double cost = revenue * feeRate;
                balance = revenue - cost;
                shares = 0;
            }
        }

        // 3. 时间前进
        currentIndex++;
        boolean done = (currentIndex >= prices.length - 1);
        double nextPrice = prices[currentIndex];

        // 4. 计算操作后的净值及收益率
        double nextNetWorth = balance + (shares * nextPrice);
        double profitRate = (nextNetWorth - prevNetWorth) / prevNetWorth;

        // 5. 构造奖励 (Reward Design) - 极其重要！
        // 放大收益，并对“持仓不涨”或“空仓大涨”给予隐形反馈
        double reward = profitRate * 100.0;
// 关键逻辑：如果是换仓动作，额外扣分
   
        // 惩罚项：鼓励模型避开回撤
        if (shares > 0 && nextPrice < currentPrice) {
            reward -= 0.01; // 额外的负反馈
        }

        // 6. 构造观察值 (加上账户持仓状态)
        // 现在的输入特征是：[原有的4个技术指标 + 1个持仓状态]
        float[] rawObs = features[currentIndex];
        float[] obsWithState = new float[rawObs.length + 1];
        System.arraycopy(rawObs, 0, obsWithState, 0, rawObs.length);
        obsWithState[rawObs.length] = (shares > 0) ? 1.0f : -1.0f; // 告诉模型你手里有没有货

        return new StepResult2(
                tensor(obsWithState).unsqueeze(0), // 观察值 Tensor
                (float) reward,
                done
        );
    }
    public StepResult2 step2(int action) {
        float currentPrice = prices[currentIndex];
        double prevNetWorth = balance + shares * currentPrice;

        // 执行动作: 0=卖出(平仓), 1=观望, 2=买入(满仓)
        if (action == 0 && shares > 0) {
            balance += shares * currentPrice * (1 - feeRate);
            shares = 0;
        } else if (action == 2 && balance > 0) {
            shares = (balance * (1 - feeRate)) / currentPrice;
            balance = 0;
        }

        currentIndex++;
        float nextPrice = prices[currentIndex];
        double currentNetWorth = balance + shares * nextPrice;

        // 奖励函数：使用对数收益率
        double reward = Math.log(currentNetWorth / prevNetWorth);
        boolean done = (currentIndex >= maxSteps - 1);

        // 构造状态 Tensor [1, FeatureSize]
        Tensor obs = tensor(features[currentIndex]).unsqueeze(0);

        return new StepResult2(obs, reward, done);
    }

    public void reset() {
        this.currentIndex = 20;
        this.balance = 10000.0;
        this.shares = 0;

    }
}