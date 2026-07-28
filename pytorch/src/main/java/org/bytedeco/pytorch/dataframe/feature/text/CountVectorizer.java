package org.bytedeco.pytorch.dataframe.feature.text;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * CountVectorizer (sklearn-compatible core parameters).
 *
 * <p>Supports {@code max_features}, {@code min_df}, {@code max_df}, {@code stop_words},
 * {@code ngram_range}, lowercase tokenization.
 */
public class CountVectorizer extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private String textColumn;
    private int maxFeatures = Integer.MAX_VALUE;
    /** Absolute count, or fraction if in (0,1). */
    private double minDf = 1.0;
    /** Absolute count, or fraction if in (0,1]. */
    private double maxDf = 1.0;
    private int ngramMin = 1;
    private int ngramMax = 1;
    private final Set<String> stopWords = new HashSet<>();
    private boolean lowercase = true;
    private final Pattern tokenPattern = Pattern.compile("\\b\\w\\w+\\b");

    private List<String> vocabulary = new ArrayList<>();
    private Map<String, Integer> vocabIndex = new LinkedHashMap<>();

    public CountVectorizer(String textColumn) {
        super(textColumn);
        this.textColumn = textColumn;
    }

    public CountVectorizer(String textColumn, int maxFeatures) {
        this(textColumn);
        this.maxFeatures = maxFeatures <= 0 ? Integer.MAX_VALUE : maxFeatures;
    }

    public CountVectorizer setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures <= 0 ? Integer.MAX_VALUE : maxFeatures;
        return this;
    }

    public CountVectorizer setMinDf(double minDf) {
        this.minDf = minDf;
        return this;
    }

    public CountVectorizer setMaxDf(double maxDf) {
        this.maxDf = maxDf;
        return this;
    }

    public CountVectorizer setNgramRange(int minN, int maxN) {
        if (minN < 1 || maxN < minN) throw new IllegalArgumentException("invalid ngram_range");
        this.ngramMin = minN;
        this.ngramMax = maxN;
        return this;
    }

    public CountVectorizer setStopWords(String... words) {
        stopWords.clear();
        if (words != null) {
            for (String w : words) {
                if (w != null) stopWords.add(lowercase ? w.toLowerCase(Locale.ROOT) : w);
            }
        }
        return this;
    }

    public CountVectorizer setStopWords(Iterable<String> words) {
        stopWords.clear();
        if (words != null) {
            for (String w : words) {
                if (w != null) stopWords.add(lowercase ? w.toLowerCase(Locale.ROOT) : w);
            }
        }
        return this;
    }

    public CountVectorizer setLowercase(boolean lowercase) {
        this.lowercase = lowercase;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (textColumn == null) {
            if (!columns.isEmpty()) textColumn = columns.get(0);
            else throw new IllegalStateException("CountVectorizer requires textColumn");
        }
        Column col = X.column(textColumn);
        int nDocs = X.rowCount();

        Map<String, Integer> df = new HashMap<>();
        Map<String, Integer> tfTotal = new HashMap<>();

        for (int i = 0; i < nDocs; i++) {
            Object raw = col.get(i);
            Object unwrapped = raw == null ? null : DataValues.unwrap(raw);
            String text = unwrapped == null ? "" : unwrapped.toString();
            List<String> ngrams = analyze(text);
            Set<String> seen = new HashSet<>();
            for (String g : ngrams) {
                tfTotal.merge(g, 1, Integer::sum);
                if (seen.add(g)) df.merge(g, 1, Integer::sum);
            }
        }

        int minCount = resolveDfThreshold(minDf, nDocs, true);
        int maxCount = resolveDfThreshold(maxDf, nDocs, false);

        List<Map.Entry<String, Integer>> candidates = new ArrayList<>();
        for (Map.Entry<String, Integer> e : df.entrySet()) {
            int dfi = e.getValue();
            if (dfi < minCount) continue;
            if (dfi > maxCount) continue;
            candidates.add(Map.entry(e.getKey(), tfTotal.getOrDefault(e.getKey(), dfi)));
        }
        candidates.sort((a, b) -> {
            int cmp = Integer.compare(b.getValue(), a.getValue());
            return cmp != 0 ? cmp : a.getKey().compareTo(b.getKey());
        });

        vocabulary = new ArrayList<>();
        vocabIndex = new LinkedHashMap<>();
        int limit = Math.min(maxFeatures, candidates.size());
        for (int i = 0; i < limit; i++) {
            String term = candidates.get(i).getKey();
            vocabIndex.put(term, vocabulary.size());
            vocabulary.add(term);
        }
        fitted = true;
        return this;
    }

    /**
     * sklearn-compatible df threshold on double-only API:
     * <ul>
     *   <li>min_df: {@code (0,1)} → fraction of docs; {@code >=1} → absolute count
     *       (so default {@code 1.0} means absolute 1, like sklearn {@code min_df=1})</li>
     *   <li>max_df: {@code (0,1]} → fraction of docs (so default {@code 1.0} = 100%);
     *       {@code >1} → absolute count</li>
     * </ul>
     */
    private int resolveDfThreshold(double dfParam, int nDocs, boolean isMin) {
        if (dfParam <= 0) return isMin ? 1 : nDocs;
        boolean asFraction = isMin
            ? (dfParam > 0 && dfParam < 1.0)
            : (dfParam > 0 && dfParam <= 1.0);
        if (asFraction) {
            int v = (int) Math.ceil(dfParam * nDocs);
            if (isMin) return Math.max(1, v);
            return Math.min(nDocs, Math.max(1, v));
        }
        int v = (int) Math.round(dfParam);
        if (isMin) return Math.max(1, v);
        return Math.min(nDocs, Math.max(1, v));
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        Column col = X.column(textColumn);
        int nDocs = X.rowCount();
        DataFrame result = X.copy();

        List<String> outNames = new ArrayList<>();
        for (String term : vocabulary) {
            String name = FeatureMatrices.uniqueName(result, "count_" + sanitize(term));
            result.addColumn(name, Column.DType.INT32);
            Column c = result.column(name);
            while (c.size() < nDocs) c.add(0);
            for (int i = 0; i < nDocs; i++) c.set(i, 0);
            outNames.add(name);
        }

        for (int i = 0; i < nDocs; i++) {
            Object raw = col.get(i);
            Object unwrapped = raw == null ? null : DataValues.unwrap(raw);
            String text = unwrapped == null ? "" : unwrapped.toString();
            List<String> ngrams = analyze(text);
            int[] counts = new int[vocabulary.size()];
            for (String g : ngrams) {
                Integer idx = vocabIndex.get(g);
                if (idx != null) counts[idx]++;
            }
            for (int t = 0; t < vocabulary.size(); t++) {
                result.set(i, outNames.get(t), counts[t]);
            }
        }
        return result;
    }

    private List<String> analyze(String text) {
        if (text == null) text = "";
        String s = lowercase ? text.toLowerCase(Locale.ROOT) : text;
        List<String> tokens = new ArrayList<>();
        var m = tokenPattern.matcher(s);
        while (m.find()) {
            String tok = m.group();
            if (stopWords.contains(tok)) continue;
            tokens.add(tok);
        }
        if (ngramMin == 1 && ngramMax == 1) return tokens;

        List<String> grams = new ArrayList<>();
        int T = tokens.size();
        for (int n = ngramMin; n <= ngramMax; n++) {
            for (int i = 0; i + n <= T; i++) {
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < n; k++) {
                    if (k > 0) sb.append(' ');
                    sb.append(tokens.get(i + k));
                }
                grams.add(sb.toString());
            }
        }
        return grams;
    }

    private static String sanitize(String term) {
        return term.replaceAll("[^A-Za-z0-9_]+", "_");
    }

    public List<String> getVocabulary() { return Collections.unmodifiableList(vocabulary); }
    public Map<String, Integer> getVocabularyIndex() { return Collections.unmodifiableMap(vocabIndex); }
    public int getFeatureCount() { return vocabulary.size(); }
}
