package org.bytedeco.pytorch.data.dataframe.feature.pipeline;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;

import java.io.Serializable;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;

/**
 * 目标变换回归器（对应 sklearn TransformedTargetRegressor）
 * 先对 y 做变换（如 log），再训练回归器，预测时做逆变换
 *
 * <pre>
 * TransformedTargetRegressor ttr = new TransformedTargetRegressor(
 *     new LinearRegression(),
 *     Math::log,   // func
 *     Math::exp    // inverse_func
 * );
 * ttr.fit(X, y);
 * double[] pred = ttr.predict(X);  // 返回原始尺度
 * </pre>
 */
public class TransformedTargetRegressor extends BaseRegressor implements Serializable {
    private static final long serialVersionUID = 1L;

    private final BaseRegressor regressor;
    private final DoubleUnaryOperator func;
    private final DoubleUnaryOperator inverseFunc;

    public TransformedTargetRegressor(BaseRegressor regressor,
                                       DoubleUnaryOperator func,
                                       DoubleUnaryOperator inverseFunc) {
        this.regressor   = regressor;
        this.func        = func;
        this.inverseFunc = inverseFunc;
    }

    /** Convenience: log / exp transform */
    public static TransformedTargetRegressor withLogTransform(BaseRegressor regressor) {
        return new TransformedTargetRegressor(regressor, Math::log1p, Math::expm1);
    }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        double[] yT = new double[y.length];
        for (int i = 0; i < y.length; i++) yT[i] = func.applyAsDouble(y[i]);
        regressor.fit(X, yT);
        fitted = true;
        return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] yT = regressor.predict(X);
        double[] result = new double[yT.length];
        for (int i = 0; i < yT.length; i++) result[i] = inverseFunc.applyAsDouble(yT[i]);
        return result;
    }

    public BaseRegressor getRegressor() { return regressor; }

    @Override
    public Map<String, Object> getParams() { return regressor.getParams(); }

    @Override
    public void setParams(Map<String, Object> params) { regressor.setParams(params); }
}

