package org.bytedeco.pytorch.data.dataframe.io;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Shared null-token sets aligned with pandas / common scientific formats.
 */
public final class IoNullTokens {
    private IoNullTokens() {}

    /** Empty, NA/N/A, null/NULL, NaN — default for CSV/JSON/Excel. */
    public static final Set<String> PANDAS_DEFAULT = unmodifiable(
        "", "NA", "N/A", "null", "Null", "NULL", "NaN", "nan", "#N/A", "#NA", "-NaN", "-nan", "<NA>"
    );

    /** {@link #PANDAS_DEFAULT} plus SQL/IMDb-style {@code \N}. */
    public static final Set<String> TSV_EXTENDED = unmodifiable(
        "", "NA", "N/A", "null", "Null", "NULL", "NaN", "nan", "\\N", "#N/A", "<NA>"
    );

    public static boolean isNull(String s, Set<String> tokens) {
        if (s == null) return true;
        if (tokens == null || tokens.isEmpty()) return s.isEmpty();
        return tokens.contains(s) || tokens.contains(s.trim());
    }

    public static boolean isNull(String s) {
        return isNull(s, PANDAS_DEFAULT);
    }

    private static Set<String> unmodifiable(String... tokens) {
        return Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(tokens)));
    }
}
