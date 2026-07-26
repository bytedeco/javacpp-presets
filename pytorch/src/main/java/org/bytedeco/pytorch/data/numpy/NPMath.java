package org.bytedeco.pytorch.data.numpy;

/**
 * NumPy-style elementwise math, trigonometric, comparison and logical ufuncs.
 * All binary ops support broadcasting via {@link NPArrayUtil}.
 */
public final class NPMath {
    private NPMath() {}

    // ---- unary math ---------------------------------------------------------

    public static NDArray abs(NDArray a) {
        if (a.isComplex()) return NPComplex.absolute(a);
        return NPArrayUtil.unary(a, Math::abs);
    }
    public static NDArray fabs(NDArray a) { return abs(a); }
    public static NDArray sqrt(NDArray a) { return NPArrayUtil.unary(a, Math::sqrt); }
    public static NDArray square(NDArray a) { return NPArrayUtil.unary(a, x -> x * x); }
    public static NDArray cbrt(NDArray a) { return NPArrayUtil.unary(a, Math::cbrt); }
    public static NDArray exp(NDArray a) { return NPArrayUtil.unary(a, Math::exp); }
    public static NDArray exp2(NDArray a) { return NPArrayUtil.unary(a, x -> Math.pow(2.0, x)); }
    public static NDArray expm1(NDArray a) { return NPArrayUtil.unary(a, Math::expm1); }
    public static NDArray log(NDArray a) { return NPArrayUtil.unary(a, Math::log); }
    public static NDArray log2(NDArray a) { return NPArrayUtil.unary(a, x -> Math.log(x) / Math.log(2)); }
    public static NDArray log10(NDArray a) { return NPArrayUtil.unary(a, Math::log10); }
    public static NDArray log1p(NDArray a) { return NPArrayUtil.unary(a, Math::log1p); }
    public static NDArray sign(NDArray a) {
        return NPArrayUtil.unary(a, x -> x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0));
    }
    public static NDArray ceil(NDArray a) { return NPArrayUtil.unary(a, Math::ceil); }
    public static NDArray floor(NDArray a) { return NPArrayUtil.unary(a, Math::floor); }
    public static NDArray trunc(NDArray a) { return NPArrayUtil.unary(a, x -> x < 0 ? Math.ceil(x) : Math.floor(x)); }
    public static NDArray rint(NDArray a) { return NPArrayUtil.unary(a, x -> (double) Math.round(x)); }
    public static NDArray round(NDArray a) { return rint(a); }
    public static NDArray negative(NDArray a) { return NPArrayUtil.unary(a, x -> -x); }
    public static NDArray neg(NDArray a) { return negative(a); }
    public static NDArray reciprocal(NDArray a) { return NPArrayUtil.unary(a, x -> 1.0 / x); }
    public static NDArray positive(NDArray a) { return NP.copy(a); }

    public static NDArray isfinite(NDArray a) { return NPArrayUtil.unaryBool(a, Double::isFinite); }
    public static NDArray isinf(NDArray a) { return NPArrayUtil.unaryBool(a, v -> Double.isInfinite(v)); }
    public static NDArray isnan(NDArray a) { return NPArrayUtil.unaryBool(a, Double::isNaN); }
    public static NDArray isreal(NDArray a) { return NPComplex.isrealArray(a); }
    public static NDArray imag(NDArray a) { return NPComplex.imag(a); }
    public static NDArray real(NDArray a) { return NPComplex.real(a); }
    public static NDArray conj(NDArray a) { return NPComplex.conj(a); }
    public static NDArray signbit(NDArray a) { return NPArrayUtil.unaryBool(a, x -> x < 0 || (x == 0 && 1 / x < 0)); }

    public static NDArray relu(NDArray a) { return NPArrayUtil.unary(a, x -> Math.max(0.0, x)); }
    public static NDArray leaky_relu(NDArray a, double alpha) {
        return NPArrayUtil.unary(a, x -> x >= 0 ? x : alpha * x);
    }
    public static NDArray sigmoid(NDArray a) {
        return NPArrayUtil.unary(a, x -> 1.0 / (1.0 + Math.exp(-x)));
    }

    // ---- trig ---------------------------------------------------------------

    public static NDArray sin(NDArray a) { return NPArrayUtil.unary(a, Math::sin); }
    public static NDArray cos(NDArray a) { return NPArrayUtil.unary(a, Math::cos); }
    public static NDArray tan(NDArray a) { return NPArrayUtil.unary(a, Math::tan); }
    public static NDArray arcsin(NDArray a) { return NPArrayUtil.unary(a, Math::asin); }
    public static NDArray asin(NDArray a) { return arcsin(a); }
    public static NDArray arccos(NDArray a) { return NPArrayUtil.unary(a, Math::acos); }
    public static NDArray acos(NDArray a) { return arccos(a); }
    public static NDArray arctan(NDArray a) { return NPArrayUtil.unary(a, Math::atan); }
    public static NDArray atan(NDArray a) { return arctan(a); }
    public static NDArray sinh(NDArray a) { return NPArrayUtil.unary(a, Math::sinh); }
    public static NDArray cosh(NDArray a) { return NPArrayUtil.unary(a, Math::cosh); }
    public static NDArray tanh(NDArray a) { return NPArrayUtil.unary(a, Math::tanh); }
    public static NDArray arcsinh(NDArray a) {
        return NPArrayUtil.unary(a, x -> Math.log(x + Math.sqrt(x * x + 1.0)));
    }
    public static NDArray asinh(NDArray a) { return arcsinh(a); }
    public static NDArray arccosh(NDArray a) {
        return NPArrayUtil.unary(a, x -> Math.log(x + Math.sqrt(x * x - 1.0)));
    }
    public static NDArray acosh(NDArray a) { return arccosh(a); }
    public static NDArray arctanh(NDArray a) {
        return NPArrayUtil.unary(a, x -> 0.5 * Math.log((1.0 + x) / (1.0 - x)));
    }
    public static NDArray atanh(NDArray a) { return arctanh(a); }
    public static NDArray radians(NDArray a) { return NPArrayUtil.unary(a, Math::toRadians); }
    public static NDArray degrees(NDArray a) { return NPArrayUtil.unary(a, Math::toDegrees); }
    public static NDArray deg2rad(NDArray a) { return radians(a); }
    public static NDArray rad2deg(NDArray a) { return degrees(a); }
    public static NDArray arctan2(NDArray y, NDArray x) { return NPArrayUtil.binary(y, x, Math::atan2); }
    public static NDArray atan2(NDArray y, NDArray x) { return arctan2(y, x); }

    // ---- binary -------------------------------------------------------------

    public static NDArray add(NDArray x1, NDArray x2) {
        if (x1.isComplex() || x2.isComplex()) return NPComplex.add(x1, x2);
        return NPArrayUtil.binary(x1, x2, Double::sum);
    }
    public static NDArray add(NDArray a, double s) { return NPArrayUtil.unary(a, x -> x + s); }
    public static NDArray subtract(NDArray x1, NDArray x2) {
        if (x1.isComplex() || x2.isComplex()) return NPComplex.subtract(x1, x2);
        return NPArrayUtil.binary(x1, x2, (x, y) -> x - y);
    }
    public static NDArray sub(NDArray x1, NDArray x2) { return subtract(x1, x2); }
    public static NDArray subtract(NDArray a, double s) { return NPArrayUtil.unary(a, x -> x - s); }
    public static NDArray sub(NDArray a, double s) { return subtract(a, s); }
    public static NDArray multiply(NDArray x1, NDArray x2) {
        if (x1.isComplex() || x2.isComplex()) return NPComplex.multiply(x1, x2);
        return NPArrayUtil.binary(x1, x2, (x, y) -> x * y);
    }
    public static NDArray mul(NDArray x1, NDArray x2) { return multiply(x1, x2); }
    public static NDArray multiply(NDArray a, double s) { return NPArrayUtil.unary(a, x -> x * s); }
    public static NDArray mul(NDArray a, double s) { return multiply(a, s); }
    public static NDArray divide(NDArray x1, NDArray x2) {
        if (x1.isComplex() || x2.isComplex()) return NPComplex.divide(x1, x2);
        return NPArrayUtil.binary(x1, x2, (x, y) -> x / y);
    }
    public static NDArray div(NDArray x1, NDArray x2) { return divide(x1, x2); }
    public static NDArray divide(NDArray a, double s) { return NPArrayUtil.unary(a, x -> x / s); }
    public static NDArray div(NDArray a, double s) { return divide(a, s); }
    public static NDArray true_divide(NDArray x1, NDArray x2) { return divide(x1, x2); }
    public static NDArray floor_divide(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, (x, y) -> Math.floor(x / y));
    }
    public static NDArray power(NDArray x1, NDArray x2) { return NPArrayUtil.binary(x1, x2, Math::pow); }
    public static NDArray power(NDArray a, double exp) { return NPArrayUtil.unary(a, x -> Math.pow(x, exp)); }
    public static NDArray pow(NDArray x1, NDArray x2) { return power(x1, x2); }
    public static NDArray mod(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, (x, y) -> {
            double r = x % y;
            if (r != 0 && Math.signum(x) != Math.signum(y)) r += y;
            return r;
        });
    }
    public static NDArray remainder(NDArray x1, NDArray x2) { return mod(x1, x2); }
    public static NDArray fmod(NDArray x1, NDArray x2) { return NPArrayUtil.binary(x1, x2, (x, y) -> x % y); }
    public static NDArray maximum(NDArray x1, NDArray x2) { return NPArrayUtil.binary(x1, x2, Math::max); }
    public static NDArray maximum(NDArray a, double s) { return NPArrayUtil.unary(a, x -> Math.max(x, s)); }
    public static NDArray minimum(NDArray x1, NDArray x2) { return NPArrayUtil.binary(x1, x2, Math::min); }
    public static NDArray minimum(NDArray a, double s) { return NPArrayUtil.unary(a, x -> Math.min(x, s)); }
    public static NDArray fmax(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, (x, y) -> Double.isNaN(x) ? y : (Double.isNaN(y) ? x : Math.max(x, y)));
    }
    public static NDArray fmin(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, (x, y) -> Double.isNaN(x) ? y : (Double.isNaN(y) ? x : Math.min(x, y)));
    }
    public static NDArray hypot(NDArray x1, NDArray x2) { return NPArrayUtil.binary(x1, x2, Math::hypot); }
    public static NDArray copysign(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, (x, y) -> Math.copySign(x, y));
    }
    public static NDArray gcd(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, DType.INT64, (x, y) -> gcdLong(Math.round(x), Math.round(y)));
    }
    public static NDArray lcm(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, DType.INT64, (x, y) -> {
            long a = Math.round(x), b = Math.round(y);
            if (a == 0 || b == 0) return 0;
            return Math.abs(a / gcdLong(a, b) * b);
        });
    }

    private static long gcdLong(long a, long b) {
        a = Math.abs(a); b = Math.abs(b);
        while (b != 0) { long t = b; b = a % b; a = t; }
        return a;
    }

    // ---- comparison / logic -------------------------------------------------

    public static NDArray equal(NDArray x1, NDArray x2) { return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a == b); }
    public static NDArray not_equal(NDArray x1, NDArray x2) { return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a != b); }
    public static NDArray greater(NDArray x1, NDArray x2) { return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a > b); }
    public static NDArray greater_equal(NDArray x1, NDArray x2) { return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a >= b); }
    public static NDArray less(NDArray x1, NDArray x2) { return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a < b); }
    public static NDArray less_equal(NDArray x1, NDArray x2) { return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a <= b); }
    public static NDArray logical_and(NDArray x1, NDArray x2) {
        return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a != 0 && b != 0);
    }
    public static NDArray logical_or(NDArray x1, NDArray x2) {
        return NPArrayUtil.binaryBool(x1, x2, (a, b) -> a != 0 || b != 0);
    }
    public static NDArray logical_xor(NDArray x1, NDArray x2) {
        return NPArrayUtil.binaryBool(x1, x2, (a, b) -> (a != 0) ^ (b != 0));
    }
    public static NDArray logical_not(NDArray x) { return NPArrayUtil.unaryBool(x, a -> a == 0); }

    // ---- special ------------------------------------------------------------

    public static NDArray clip(NDArray a, double min, double max) {
        return NPArrayUtil.unary(a, x -> Math.max(min, Math.min(max, x)));
    }
    public static NDArray clip(NDArray a, NDArray min, NDArray max) {
        return minimum(maximum(a, min), max);
    }

    public static NDArray where(NDArray cond, NDArray x, NDArray y) {
        // broadcast all three to common shape
        long[] s1 = NPArrayUtil.broadcastShapes(cond.shape, x.shape);
        long[] shape = NPArrayUtil.broadcastShapes(s1, y.shape);
        long[] cSt = NPArrayUtil.stridesOf(cond.shape);
        long[] xSt = NPArrayUtil.stridesOf(x.shape);
        long[] ySt = NPArrayUtil.stridesOf(y.shape);
        long[] oSt = NPArrayUtil.stridesOf(shape);
        NDArray out = new NDArray(NPArrayUtil.promote(x.dtype, y.dtype), shape);
        int[] idx = new int[shape.length];
        for (int flat = 0; flat < out.size; flat++) {
            NPArrayUtil.fillMultiIndex(flat, shape, oSt, idx);
            double c = cond.getDouble(NPArrayUtil.broadcastIndex(idx, cond.shape, cSt));
            double xv = x.getDouble(NPArrayUtil.broadcastIndex(idx, x.shape, xSt));
            double yv = y.getDouble(NPArrayUtil.broadcastIndex(idx, y.shape, ySt));
            out.setDouble(flat, c != 0 ? xv : yv);
        }
        return out;
    }

    public static NDArray heaviside(NDArray x1, NDArray x2) {
        return NPArrayUtil.binary(x1, x2, (x, h0) -> x < 0 ? 0 : (x == 0 ? h0 : 1));
    }

    public static NDArray nan_to_num(NDArray a) {
        return NPArrayUtil.unary(a, x -> {
            if (Double.isNaN(x)) return 0;
            if (x == Double.POSITIVE_INFINITY) return Double.MAX_VALUE;
            if (x == Double.NEGATIVE_INFINITY) return -Double.MAX_VALUE;
            return x;
        });
    }

    }
