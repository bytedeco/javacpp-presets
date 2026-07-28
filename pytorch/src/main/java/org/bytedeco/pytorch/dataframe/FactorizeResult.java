package org.bytedeco.pytorch.dataframe;

/**
 * Result of {@link DataFrame#factorize(String)}: integer codes + unique labels (order of appearance).
 */
public final class FactorizeResult {
    private final int[] codes;
    private final String[] uniques;

    public FactorizeResult(int[] codes, String[] uniques) {
        this.codes = codes;
        this.uniques = uniques;
    }

    public int[] codes() { return codes; }
    public String[] uniques() { return uniques; }

    /** Number of distinct labels. */
    public int nUnique() { return uniques.length; }

    @Override
    public String toString() {
        return "FactorizeResult{n=" + codes.length + ", uniques=" + uniques.length + "}";
    }
}
