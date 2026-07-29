/*
 * Davis / KIBA drug-target affinity datasets for DeepDTA / MolTrans / DrugBAN.
 *
 * References:
 *   - Davis et al., "Comprehensive analysis of kinase inhibitor selectivity",
 *     Nature Biotech 2011 (Kd affinities)
 *   - Tang et al., "Making sense of large-scale kinase inhibitor bioactivity data
 *     sets" (KIBA score)
 *   - DeepDTA (Öztürk et al., Bioinformatics 2018) popularized these benchmarks
 *
 * Official processed dumps are commonly mirrored with DeepDTA / GraphDTA repos.
 * This loader:
 *   1. Tries DatasetDownloader from known raw mirrors / local cache
 *   2. Falls back to synthetic SMILES-char / AA-token pairs with continuous labels
 *
 * Tensor contract:
 *   drugTokens    [N, Ld] long
 *   proteinTokens [N, Lp] long
 *   affinity      [N] float   (regression target; higher = stronger for pKd-style)
 *
 * Cache: ~/.torchrec-datasets/davis_kiba/
 */
package org.bytedeco.pytorch.utils.recommend.data.industry;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.data.DatasetDownloader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DavisKibaDataset {

    public static final int DEFAULT_DRUG_LEN = 64;
    public static final int DEFAULT_PROT_LEN = 128;
    /** SMILES char vocab incl. pad/unk — DeepDTA ~64. */
    public static final int DEFAULT_DRUG_VOCAB = 64;
    /** Amino-acid vocab incl. pad/unk — DeepDTA ~25. */
    public static final int DEFAULT_PROT_VOCAB = 25;

    public enum Source { DAVIS, KIBA, AUTO }

    private DavisKibaDataset() {}

    public static final class Split {
        public final Tensor drugTokens;
        public final Tensor proteinTokens;
        public final Tensor affinity;
        public final int drugVocabSize;
        public final int proteinVocabSize;
        public final boolean synthetic;
        public final String name;

        public Split(Tensor drugTokens, Tensor proteinTokens, Tensor affinity,
                     int drugVocabSize, int proteinVocabSize, boolean synthetic, String name) {
            this.drugTokens = drugTokens;
            this.proteinTokens = proteinTokens;
            this.affinity = affinity;
            this.drugVocabSize = drugVocabSize;
            this.proteinVocabSize = proteinVocabSize;
            this.synthetic = synthetic;
            this.name = name;
        }

        public long size() {
            return drugTokens.size(0);
        }

        /** Train/val/test index ranges as [trainEnd, valEnd) on shuffled order. */
        public SubSplits split(float trainRatio, float valRatio, int seed) {
            int n = (int) size();
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Random rng = new Random(seed);
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
            int nTrain = Math.max(1, (int) (n * trainRatio));
            int nVal = Math.max(1, (int) (n * valRatio));
            if (nTrain + nVal >= n) {
                nVal = Math.max(1, (n - nTrain) / 2);
            }
            int nTest = n - nTrain - nVal;
            return new SubSplits(
                    select(idx, 0, nTrain),
                    select(idx, nTrain, nTrain + nVal),
                    select(idx, nTrain + nVal, nTrain + nVal + nTest));
        }

        private Split select(int[] idx, int from, int to) {
            int m = Math.max(0, to - from);
            long Ld = drugTokens.size(1);
            long Lp = proteinTokens.size(1);
            long[] d = new long[m * (int) Ld];
            long[] p = new long[m * (int) Lp];
            float[] a = new float[m];
            // gather via CPU loops on cloned arrays would need data_ptr; use index_select
            long[] indexArr = new long[m];
            for (int i = 0; i < m; i++) indexArr[i] = idx[from + i];
            Tensor index = torch.tensor(indexArr, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            Tensor dSel = drugTokens.index_select(0L, index);
            Tensor pSel = proteinTokens.index_select(0L, index);
            Tensor aSel = affinity.index_select(0L, index);
            return new Split(dSel, pSel, aSel, drugVocabSize, proteinVocabSize, synthetic, name);
        }
    }

    public static final class SubSplits {
        public final Split train, val, test;
        public SubSplits(Split train, Split val, Split test) {
            this.train = train; this.val = val; this.test = test;
        }
    }

    public static Split load() {
        return load(Source.AUTO, 4_000, 42);
    }

    public static Split load(Source source, int maxSamples, int seed) {
        System.out.println("============================================================");
        System.out.println("Davis/KIBA Dataset Loading (" + source + ")");
        System.out.println("============================================================");
        File dir = tryLocate(source);
        if (dir != null) {
            try {
                Split s = parseDir(dir, source, maxSamples);
                if (s != null && s.size() > 0) {
                    System.out.println("  [OK] Loaded " + s.size() + " pairs from " + dir);
                    System.out.println("============================================================");
                    return s;
                }
            } catch (Throwable t) {
                System.out.println("  [Warn] parse failed: " + t.getMessage());
            }
        }
        System.out.println("  [Fallback] Synthetic DTI pairs...");
        String name = source == Source.KIBA ? "kiba-synthetic" : "davis-synthetic";
        Split syn = generateSynthetic(maxSamples, seed, name);
        System.out.println("  [OK] Synthetic: " + syn.size());
        System.out.println("============================================================");
        return syn;
    }

    private static File tryLocate(Source source) {
        String env = System.getenv("DTI_DATA_DIR");
        if (env != null && !env.isEmpty()) {
            File f = new File(env);
            if (f.isDirectory()) return f;
        }
        File base = new File(DatasetDownloader.cacheDir(), "davis_kiba");
        if (base.isDirectory()) return base;
        File davis = new File(DatasetDownloader.cacheDir(), "davis");
        if (davis.isDirectory()) return davis;
        File kiba = new File(DatasetDownloader.cacheDir(), "kiba");
        if (kiba.isDirectory()) return kiba;
        System.out.println("  [Info] Place Davis/KIBA CSVs under " + base.getAbsolutePath());
        System.out.println("         Expected columns: drug_smiles,protein_seq,affinity  OR");
        System.out.println("         DeepDTA-style ligands/proteins/Y matrices.");
        System.out.println("         Or set DTI_DATA_DIR.");
        //noinspection ResultOfMethodCallIgnored
        base.mkdirs();
        return null;
    }

    /**
     * Parse simple CSV: smiles,sequence,affinity
     * (header optional).
     */
    private static Split parseDir(File dir, Source source, int maxSamples) throws Exception {
        File csv = null;
        for (String name : new String[]{"pairs.csv", "davis.csv", "kiba.csv", "data.csv"}) {
            File f = new File(dir, name);
            if (f.isFile() && f.length() > 0) { csv = f; break; }
        }
        if (csv == null) return null;

        Map<Character, Integer> drugVocab = defaultDrugVocab();
        Map<Character, Integer> protVocab = defaultProtVocab();
        List<long[]> drugs = new ArrayList<>();
        List<long[]> prots = new ArrayList<>();
        List<Float> affs = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csv))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null && drugs.size() < maxSamples) {
                if (first) {
                    first = false;
                    if (line.toLowerCase().contains("smiles") || line.toLowerCase().contains("affinity")) {
                        continue; // header
                    }
                }
                String[] p = line.split(",", -1);
                if (p.length < 3) p = line.split("\t", -1);
                if (p.length < 3) continue;
                String smiles = p[0].trim();
                String seq = p[1].trim();
                float y;
                try { y = Float.parseFloat(p[2].trim()); } catch (Exception e) { continue; }
                drugs.add(encodeChars(smiles, drugVocab, DEFAULT_DRUG_LEN));
                prots.add(encodeChars(seq, protVocab, DEFAULT_PROT_LEN));
                affs.add(y);
            }
        }
        if (drugs.isEmpty()) return null;
        return toSplit(drugs, prots, affs, drugVocab.size(), protVocab.size(), false,
                source == Source.KIBA ? "kiba" : "davis");
    }

    public static Split generateSynthetic(int numSamples, int seed, String name) {
        Random rng = new Random(seed);
        int n = Math.max(numSamples, 16);
        List<long[]> drugs = new ArrayList<>(n);
        List<long[]> prots = new ArrayList<>(n);
        List<Float> affs = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            long[] d = new long[DEFAULT_DRUG_LEN];
            long[] p = new long[DEFAULT_PROT_LEN];
            int dLen = 8 + rng.nextInt(DEFAULT_DRUG_LEN - 8);
            int pLen = 16 + rng.nextInt(DEFAULT_PROT_LEN - 16);
            for (int t = 0; t < dLen; t++) d[t] = 1 + rng.nextInt(DEFAULT_DRUG_VOCAB - 1);
            for (int t = 0; t < pLen; t++) p[t] = 1 + rng.nextInt(DEFAULT_PROT_VOCAB - 1);
            // affinity correlated with simple hash of tokens for learnable signal
            double s = 0;
            for (int t = 0; t < 8; t++) s += d[t] + p[t];
            float y = (float) (5.0 + (s % 50) / 10.0 + rng.nextGaussian() * 0.1);
            drugs.add(d);
            prots.add(p);
            affs.add(y);
        }
        return toSplit(drugs, prots, affs, DEFAULT_DRUG_VOCAB, DEFAULT_PROT_VOCAB, true, name);
    }

    private static Split toSplit(List<long[]> drugs, List<long[]> prots, List<Float> affs,
                                 int dVocab, int pVocab, boolean synthetic, String name) {
        int n = drugs.size();
        int Ld = drugs.get(0).length;
        int Lp = prots.get(0).length;
        long[] dFlat = new long[n * Ld];
        long[] pFlat = new long[n * Lp];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(drugs.get(i), 0, dFlat, i * Ld, Ld);
            System.arraycopy(prots.get(i), 0, pFlat, i * Lp, Lp);
            y[i] = affs.get(i);
        }
        Tensor dT = torch.tensor(dFlat, optsLong()).view(n, Ld);
        Tensor pT = torch.tensor(pFlat, optsLong()).view(n, Lp);
        Tensor yT = torch.tensor(y, optsFloat());
        return new Split(dT, pT, yT, dVocab, pVocab, synthetic, name);
    }

    private static long[] encodeChars(String s, Map<Character, Integer> vocab, int maxLen) {
        long[] ids = new long[maxLen];
        if (s == null) return ids;
        int i = 0;
        for (int c = 0; c < s.length() && i < maxLen; c++) {
            char ch = s.charAt(c);
            Integer id = vocab.get(ch);
            if (id == null) id = vocab.getOrDefault('?', 1);
            ids[i++] = id;
        }
        return ids;
    }

    private static Map<Character, Integer> defaultDrugVocab() {
        Map<Character, Integer> m = new HashMap<>();
        m.put('\0', 0);
        m.put('?', 1);
        String chars = "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTUVWXYZ[\\]abcdefgilmnoprstuy";
        for (int i = 0; i < chars.length(); i++) {
            m.putIfAbsent(chars.charAt(i), m.size());
        }
        return m;
    }

    private static Map<Character, Integer> defaultProtVocab() {
        Map<Character, Integer> m = new HashMap<>();
        m.put('\0', 0);
        m.put('?', 1);
        String aa = "ACDEFGHIKLMNPQRSTVWY";
        for (int i = 0; i < aa.length(); i++) {
            m.put(aa.charAt(i), m.size());
        }
        return m;
    }

    private static TensorOptions optsLong() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
    }

    private static TensorOptions optsFloat() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
    }
}
