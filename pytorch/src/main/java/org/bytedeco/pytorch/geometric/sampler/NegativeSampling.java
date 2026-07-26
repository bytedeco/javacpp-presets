package org.bytedeco.pytorch.geometric.sampler;

public class NegativeSampling {
    public enum Strategy { RANDOM, TRIPLET }

    private Strategy strategy;
    private int amount; // 每个正样本对应的负样本数

    public NegativeSampling(Strategy strategy, int amount) {
        this.strategy = strategy;
        this.amount = amount;
    }

    public int getAmount() { return amount; }
    public Strategy getStrategy() { return strategy; }
}
