package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.rl.StepResult;

public class SimpleTradingEnv {
    private float[] prices; // 模拟价格序列
    private int currentIndex;
    private int windowSize = 5;
    private float balance = 1000f;
    private float shares = 0;

    public SimpleTradingEnv(float[] prices) {
        this.prices = prices;
        reset();
    }

    public float[] reset() {
        currentIndex = windowSize;
        balance = 1000f;
        shares = 0;
        return getObservation();
    }

    private float[] getObservation() {
        float[] obs = new float[windowSize];
        for (int i = 0; i < windowSize; i++) {
            // 观测值为价格的变化率
            obs[i] = (prices[currentIndex - windowSize + i + 1] / prices[currentIndex - windowSize + i]) - 1.0f;
        }
        return obs;
    }

    public StepResult step(int action) {
        float price = prices[currentIndex];
        float prevValue = balance + shares * price;

        if (action == 2 && balance >= price) { // 买入
            shares += 1;
            balance -= price;
        } else if (action == 0 && shares > 0) { // 卖出
            balance += price;
            shares -= 1;
        }

        currentIndex++;
        float currentValue = balance + shares * prices[currentIndex];
        float reward = (currentValue / prevValue) - 1.0f; // 收益率作为奖励

        boolean done = currentIndex >= prices.length - 1;
        return new StepResult(getObservation(), reward, done);
    }
}