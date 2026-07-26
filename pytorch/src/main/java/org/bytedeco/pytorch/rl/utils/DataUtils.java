package org.bytedeco.pytorch.rl.utils;
import org.bytedeco.pytorch.jit.*;

//public class DataUtils {
//    public static float[] loadPrices(String path) {
//        // 从 CSV 读取收盘价列
//        return new float[]{100.1f, 100.5f, 99.8f, 101.2f};
//    }
//
//    public static float[][] computeFeatures(float[] prices) {
//        float[][] features = new float[prices.length][4];
//        for (int i = 1; i < prices.length; i++) {
//            features[i][0] = (prices[i] - prices[i-1]) / prices[i-1]; // 收益率
//            // ... 添加均线、RSI 等特征
//        }
//        return features;
//    }
//}


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataUtils {

    /**
     * 加载 CSV 中的收盘价
     * 假设 CSV 格式: Date, Open, High, Low, Close, Volume
     * 我们通常取第 4 列 (索引为 4)
     */
    public static float[] loadPrices(String path) {
        List<Float> priceList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean isHeader = true;

            while ((line = br.readLine()) != null) {
                if (isHeader) { // 跳过表头
                    isHeader = false;
                    continue;
                }

                String[] values = line.split(",");
                // 确保列数足够，取 Close 列 (根据你的 CSV 结构调整索引)
                if (values.length > 4) {
                    priceList.add(Float.parseFloat(values[4]));
                }
            }
        } catch (IOException | NumberFormatException e) {
            System.err.println("数据加载失败: " + e.getMessage());
            // 返回一个假数据或抛出异常，防止程序崩溃
            return new float[]{100.0f, 101.0f, 102.0f};
        }

        // 转换为原生数组以适配 Tensor 构造
        float[] prices = new float[priceList.size()];
        for (int i = 0; i < priceList.size(); i++) {
            prices[i] = priceList.get(i);
        }

        System.out.println("成功加载数据点: " + prices.length);
        return prices;
    }

    /**
     * 计算特征矩阵 [TotalSteps][FeatureDim]
     * 特征通常需要归一化，否则神经网络难以收敛
     */
    public static float[][] computeFeatures(float[] prices) {
        // 定义特征维度，例如 4 个特征：[收益率, 5日均线偏差, 20日均线偏差, 波动率]
        int featureDim = 4;
        float[][] features = new float[prices.length][featureDim];

        for (int i = 20; i < prices.length; i++) {
            // 1. 收益率 (Log Return)
            features[i][0] = (float) Math.log(prices[i] / prices[i - 1]);

            // 2. 5日均线偏差 (Price / MA5 - 1)
            float ma5 = getMA(prices, i, 5);
            features[i][1] = (prices[i] / ma5) - 1;

            // 3. 20日均线偏差 (Price / MA20 - 1)
            float ma20 = getMA(prices, i, 20);
            features[i][2] = (prices[i] / ma20) - 1;

            // 4. 波动率 (简单实现：过去 5 天的极差)
            features[i][3] = (getMax(prices, i, 5) - getMin(prices, i, 5)) / prices[i];
        }
        return features;
    }

    private static float getMA(float[] p, int curr, int n) {
        float sum = 0;
        for (int i = curr - n + 1; i <= curr; i++) sum += p[i];
        return sum / n;
    }

    private static float getMax(float[] p, int curr, int n) {
        float max = -Float.MAX_VALUE;
        for (int i = curr - n + 1; i <= curr; i++) max = Math.max(max, p[i]);
        return max;
    }

    private static float getMin(float[] p, int curr, int n) {
        float min = Float.MAX_VALUE;
        for (int i = curr - n + 1; i <= curr; i++) min = Math.min(min, p[i]);
        return min;
    }
}