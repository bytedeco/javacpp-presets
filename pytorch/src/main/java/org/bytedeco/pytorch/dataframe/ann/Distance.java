package org.bytedeco.pytorch.dataframe.ann;

/**
 * Vector distance / similarity spaces for HNSW.
 * For IP and COSINE, larger similarity is better — search ranks by ascending
 * distance where distance = -similarity (so top-k min-heap still works).
 */
public enum Distance {
    /** Squared L2 (Euclidean) distance — lower is better. */
    L2 {
        @Override public float distance(float[] a, float[] b) {
            float s = 0f;
            int n = a.length;
            for (int i = 0; i < n; i++) {
                float d = a[i] - b[i];
                s += d * d;
            }
            return s;
        }
        @Override public boolean lowerIsBetter() { return true; }
    },
    /** Inner product — higher is better; stored as distance = -dot. */
    IP {
        @Override public float distance(float[] a, float[] b) {
            float s = 0f;
            int n = a.length;
            for (int i = 0; i < n; i++) s += a[i] * b[i];
            return -s;
        }
        @Override public boolean lowerIsBetter() { return true; } // after negation
    },
    /** Cosine distance = 1 - cos_sim (assumes optional pre-normalization). */
    COSINE {
        @Override public float distance(float[] a, float[] b) {
            float dot = 0f, na = 0f, nb = 0f;
            int n = a.length;
            for (int i = 0; i < n; i++) {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            if (na == 0f || nb == 0f) return 1f;
            float cos = dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
            // clamp numerical noise
            if (cos > 1f) cos = 1f;
            if (cos < -1f) cos = -1f;
            return 1f - cos;
        }
        @Override public boolean lowerIsBetter() { return true; }
    };

    public abstract float distance(float[] a, float[] b);
    public abstract boolean lowerIsBetter();

    /** L2 against a contiguous row-major matrix row. */
    public float distance(float[] query, float[] matrix, int row, int dim) {
        float s = 0f;
        int base = row * dim;
        switch (this) {
            case L2: {
                for (int i = 0; i < dim; i++) {
                    float d = query[i] - matrix[base + i];
                    s += d * d;
                }
                return s;
            }
            case IP: {
                for (int i = 0; i < dim; i++) s += query[i] * matrix[base + i];
                return -s;
            }
            case COSINE: {
                float dot = 0f, na = 0f, nb = 0f;
                for (int i = 0; i < dim; i++) {
                    float av = query[i], bv = matrix[base + i];
                    dot += av * bv;
                    na += av * av;
                    nb += bv * bv;
                }
                if (na == 0f || nb == 0f) return 1f;
                float cos = dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
                if (cos > 1f) cos = 1f;
                if (cos < -1f) cos = -1f;
                return 1f - cos;
            }
            default: return Float.MAX_VALUE;
        }
    }
}
