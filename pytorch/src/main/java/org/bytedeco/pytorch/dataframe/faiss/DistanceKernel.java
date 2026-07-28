package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Hot-path distance kernels with 8-wide unrolling — used by HNSW / Flat fallback.
 */
public final class DistanceKernel {
    private DistanceKernel() {}

    /** Squared L2 between two dense vectors. */
    public static float l2(float[] a, float[] b, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            float d0 = a[i] - b[i];
            float d1 = a[i + 1] - b[i + 1];
            float d2 = a[i + 2] - b[i + 2];
            float d3 = a[i + 3] - b[i + 3];
            float d4 = a[i + 4] - b[i + 4];
            float d5 = a[i + 5] - b[i + 5];
            float d6 = a[i + 6] - b[i + 6];
            float d7 = a[i + 7] - b[i + 7];
            s0 += d0 * d0 + d1 * d1;
            s1 += d2 * d2 + d3 * d3;
            s2 += d4 * d4 + d5 * d5;
            s3 += d6 * d6 + d7 * d7;
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    /** Squared L2: query vs row-major matrix row. */
    public static float l2Row(float[] q, float[] matrix, int rowBase, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            float d0 = q[i] - matrix[rowBase + i];
            float d1 = q[i + 1] - matrix[rowBase + i + 1];
            float d2 = q[i + 2] - matrix[rowBase + i + 2];
            float d3 = q[i + 3] - matrix[rowBase + i + 3];
            float d4 = q[i + 4] - matrix[rowBase + i + 4];
            float d5 = q[i + 5] - matrix[rowBase + i + 5];
            float d6 = q[i + 6] - matrix[rowBase + i + 6];
            float d7 = q[i + 7] - matrix[rowBase + i + 7];
            s0 += d0 * d0 + d1 * d1;
            s1 += d2 * d2 + d3 * d3;
            s2 += d4 * d4 + d5 * d5;
            s3 += d6 * d6 + d7 * d7;
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) {
            float d = q[i] - matrix[rowBase + i];
            s += d * d;
        }
        return s;
    }

    /** Inner product. */
    public static float ip(float[] a, float[] b, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            s0 += a[i] * b[i] + a[i + 1] * b[i + 1];
            s1 += a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            s2 += a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5];
            s3 += a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) s += a[i] * b[i];
        return s;
    }

    public static float ipRow(float[] q, float[] matrix, int rowBase, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            s0 += q[i] * matrix[rowBase + i] + q[i + 1] * matrix[rowBase + i + 1];
            s1 += q[i + 2] * matrix[rowBase + i + 2] + q[i + 3] * matrix[rowBase + i + 3];
            s2 += q[i + 4] * matrix[rowBase + i + 4] + q[i + 5] * matrix[rowBase + i + 5];
            s3 += q[i + 6] * matrix[rowBase + i + 6] + q[i + 7] * matrix[rowBase + i + 7];
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) s += q[i] * matrix[rowBase + i];
        return s;
    }

    /** Squared L2 with query offset into a packed query matrix. */
    public static float l2Row(float[] queries, int qOff, float[] matrix, int rowBase, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            float d0 = queries[qOff + i] - matrix[rowBase + i];
            float d1 = queries[qOff + i + 1] - matrix[rowBase + i + 1];
            float d2 = queries[qOff + i + 2] - matrix[rowBase + i + 2];
            float d3 = queries[qOff + i + 3] - matrix[rowBase + i + 3];
            float d4 = queries[qOff + i + 4] - matrix[rowBase + i + 4];
            float d5 = queries[qOff + i + 5] - matrix[rowBase + i + 5];
            float d6 = queries[qOff + i + 6] - matrix[rowBase + i + 6];
            float d7 = queries[qOff + i + 7] - matrix[rowBase + i + 7];
            s0 += d0 * d0 + d1 * d1;
            s1 += d2 * d2 + d3 * d3;
            s2 += d4 * d4 + d5 * d5;
            s3 += d6 * d6 + d7 * d7;
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) {
            float dv = queries[qOff + i] - matrix[rowBase + i];
            s += dv * dv;
        }
        return s;
    }

    /** Inner product with query offset into a packed query matrix. */
    public static float ipRow(float[] queries, int qOff, float[] matrix, int rowBase, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            s0 += queries[qOff + i] * matrix[rowBase + i]
                + queries[qOff + i + 1] * matrix[rowBase + i + 1];
            s1 += queries[qOff + i + 2] * matrix[rowBase + i + 2]
                + queries[qOff + i + 3] * matrix[rowBase + i + 3];
            s2 += queries[qOff + i + 4] * matrix[rowBase + i + 4]
                + queries[qOff + i + 5] * matrix[rowBase + i + 5];
            s3 += queries[qOff + i + 6] * matrix[rowBase + i + 6]
                + queries[qOff + i + 7] * matrix[rowBase + i + 7];
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) s += queries[qOff + i] * matrix[rowBase + i];
        return s;
    }

    /** Squared L2 norm of one vector. */
    public static float sqNorm(float[] v, int off, int dim) {
        float s = 0;
        int end = off + dim;
        for (int i = off; i < end; i++) s += v[i] * v[i];
        return s;
    }

    /** In-place L2 normalize rows of row-major matrix. */
    public static void normalizeL2(float[] x, int n, int d) {
        for (int i = 0; i < n; i++) {
            int base = i * d;
            float sum = 0f;
            for (int j = 0; j < d; j++) sum += x[base + j] * x[base + j];
            if (sum > 0f) {
                float inv = (float) (1.0 / Math.sqrt(sum));
                for (int j = 0; j < d; j++) x[base + j] *= inv;
            }
        }
    }

    /** Copy+normalize into dst. */
    public static void normalizeL2Copy(float[] src, float[] dst, int n, int d) {
        for (int i = 0; i < n; i++) {
            int base = i * d;
            float sum = 0f;
            for (int j = 0; j < d; j++) sum += src[base + j] * src[base + j];
            float inv = sum > 0f ? (float) (1.0 / Math.sqrt(sum)) : 1f;
            for (int j = 0; j < d; j++) dst[base + j] = src[base + j] * inv;
        }
    }
}
