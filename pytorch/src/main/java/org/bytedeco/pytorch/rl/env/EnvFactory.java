package org.bytedeco.pytorch.rl.env;

import java.io.Serializable;

//public class EnvFactory {
//    public static TradingEnv createTradingEnv(float[] prices, float[][] features) {
//        return new TradingEnv(prices, features);
//    }
//}
@FunctionalInterface
public interface EnvFactory extends Serializable {
    TradingEnv create();
    public static TradingEnv createTradingEnv(float[] prices, float[][] features) {
        return new TradingEnv(prices, features);
    }
}