/*
 * MIND — Microsoft News Dataset loader for news recommend models (NRMS/NAML/LSTUR/NPA/DKN).
 *
 * Official: https://msnews.github.io/
 * Small demo split often distributed as MIND-small (train/dev behaviors + news.tsv).
 *
 * This loader:
 *   1. Tries DatasetDownloader for a compact public mirror / user-provided path
 *   2. Falls back to synthetic MIND-shaped tensors so smoke tests / benchmarks always run
 *
 * Tensor contract used by models.news.*:
 *   historyTokenIds   [N, H, L] long
 *   candidateTokenIds [N, C, L] long
 *   userIds           [N] long           (LSTUR/NPA)
 *   labels            [N, C] float       (multi-candidate click labels; or [N] if C=1)
 *
 * Cache: ~/.torchrec-datasets/mind/
 */
package org.bytedeco.pytorch.utils.recommend.data.industry;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
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
public final class MindDataset {

    public static final int DEFAULT_TITLE_LEN = 16;
    public static final int DEFAULT_HIST_LEN = 20;
    public static final int DEFAULT_NUM_CAND = 5;
    public static final int DEFAULT_VOCAB = 5000;

    private MindDataset() {}

    public static final class Split {
        public final Tensor historyTokenIds;   // [N, H, L]
        public final Tensor candidateTokenIds; // [N, C, L]
        public final Tensor userIds;           // [N]
        public final Tensor labels;            // [N, C]
        public final int vocabSize;
        public final int numUsers;
        public final boolean synthetic;

        public Split(Tensor historyTokenIds, Tensor candidateTokenIds, Tensor userIds,
                     Tensor labels, int vocabSize, int numUsers, boolean synthetic) {
            this.historyTokenIds = historyTokenIds;
            this.candidateTokenIds = candidateTokenIds;
            this.userIds = userIds;
            this.labels = labels;
            this.vocabSize = vocabSize;
            this.numUsers = numUsers;
            this.synthetic = synthetic;
        }

        public long size() {
            return historyTokenIds.size(0);
        }
    }

    /** Load with defaults; synthetic fallback if download/parse fails. */
    public static Split load() {
        return load(8_000, 42);
    }

    public static Split load(int maxSamples, int seed) {
        System.out.println("============================================================");
        System.out.println("MIND News Dataset Loading");
        System.out.println("============================================================");
        File dir = tryDownload();
        if (dir != null) {
            try {
                Split s = parseMindDir(dir, maxSamples, seed);
                if (s != null && s.size() > 0) {
                    System.out.println("  [OK] Real/partial MIND samples: " + s.size()
                            + " vocab=" + s.vocabSize);
                    System.out.println("============================================================");
                    return s;
                }
            } catch (Throwable t) {
                System.out.println("  [Warn] MIND parse failed: " + t.getMessage());
            }
        }
        System.out.println("  [Fallback] Generating synthetic MIND-shaped data...");
        Split syn = generateSynthetic(maxSamples, seed);
        System.out.println("  [OK] Synthetic samples: " + syn.size());
        System.out.println("============================================================");
        return syn;
    }

    /**
     * Attempt download. MIND official requires registration; we try a small public
     * sample if available, otherwise return null (caller uses synthetic).
     */
    private static File tryDownload() {
        // Prefer already-cached directory
        File cached = new File(DatasetDownloader.cacheDir(), "mind");
        if (cached.isDirectory()) {
            File news = new File(cached, "news.tsv");
            File behaviors = new File(cached, "behaviors.tsv");
            if (news.exists() || behaviors.exists()) {
                System.out.println("  [Cache] " + cached.getAbsolutePath());
                return cached;
            }
        }
        // Optional env override
        String env = System.getenv("MIND_DATA_DIR");
        if (env != null && !env.isEmpty()) {
            File f = new File(env);
            if (f.isDirectory()) return f;
        }
        // No reliable unauthenticated full dump — document and skip
        System.out.println("  [Info] Full MIND requires https://msnews.github.io/ registration.");
        System.out.println("  [Info] Set MIND_DATA_DIR to a local extract with news.tsv / behaviors.tsv");
        System.out.println("         or place files under " + cached.getAbsolutePath());
        return cached.isDirectory() ? cached : null;
    }

    /**
     * Minimal TSV parse if user placed files locally:
     *   news.tsv: news_id \t category \t subcategory \t title \t ...
     *   behaviors.tsv: impression_id \t user_id \t time \t history \t impressions
     * History/impressions: news_id-click pairs space-separated (N1-1 N2-0 ...).
     */
    private static Split parseMindDir(File dir, int maxSamples, int seed) throws Exception {
        File newsFile = firstExisting(dir, "news.tsv", "MINDsmall_train/news.tsv", "train/news.tsv");
        File behFile = firstExisting(dir, "behaviors.tsv", "MINDsmall_train/behaviors.tsv", "train/behaviors.tsv");
        if (newsFile == null || behFile == null) return null;

        Map<String, long[]> newsTitles = new HashMap<>();
        Map<String, Integer> wordVocab = new HashMap<>();
        wordVocab.put("<pad>", 0);
        wordVocab.put("<unk>", 1);
        try (BufferedReader br = new BufferedReader(new FileReader(newsFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split("\t", -1);
                if (p.length < 4) continue;
                String nid = p[0];
                String title = p[3];
                newsTitles.put(nid, tokenize(title, wordVocab, DEFAULT_TITLE_LEN));
            }
        }
        if (newsTitles.isEmpty()) return null;

        List<long[][]> histList = new ArrayList<>();
        List<long[][]> candList = new ArrayList<>();
        List<Long> userList = new ArrayList<>();
        List<float[]> labelList = new ArrayList<>();
        Map<String, Integer> userVocab = new HashMap<>();
        userVocab.put("<unk>", 0);

        try (BufferedReader br = new BufferedReader(new FileReader(behFile))) {
            String line;
            while ((line = br.readLine()) != null && histList.size() < maxSamples) {
                String[] p = line.split("\t", -1);
                if (p.length < 5) continue;
                String uid = p[1];
                String hist = p[3];
                String imps = p[4];
                int uIdx = userVocab.computeIfAbsent(uid, k -> userVocab.size());

                long[][] histTok = new long[DEFAULT_HIST_LEN][DEFAULT_TITLE_LEN];
                String[] hIds = hist == null || hist.isEmpty() ? new String[0] : hist.trim().split("\\s+");
                int hCount = 0;
                for (int i = Math.max(0, hIds.length - DEFAULT_HIST_LEN); i < hIds.length; i++) {
                    long[] title = newsTitles.getOrDefault(hIds[i], new long[DEFAULT_TITLE_LEN]);
                    histTok[hCount++] = title;
                }

                String[] impArr = imps.trim().split("\\s+");
                long[][] candTok = new long[DEFAULT_NUM_CAND][DEFAULT_TITLE_LEN];
                float[] labs = new float[DEFAULT_NUM_CAND];
                int cCount = 0;
                for (String imp : impArr) {
                    if (cCount >= DEFAULT_NUM_CAND) break;
                    int dash = imp.lastIndexOf('-');
                    if (dash <= 0) continue;
                    String nid = imp.substring(0, dash);
                    float lab = 0f;
                    try { lab = Float.parseFloat(imp.substring(dash + 1)); } catch (Exception ignored) {}
                    candTok[cCount] = newsTitles.getOrDefault(nid, new long[DEFAULT_TITLE_LEN]);
                    labs[cCount] = lab;
                    cCount++;
                }
                if (cCount == 0) continue;

                histList.add(histTok);
                candList.add(candTok);
                userList.add((long) uIdx);
                labelList.add(labs);
            }
        }
        if (histList.isEmpty()) return null;

        int n = histList.size();
        long[] histFlat = new long[n * DEFAULT_HIST_LEN * DEFAULT_TITLE_LEN];
        long[] candFlat = new long[n * DEFAULT_NUM_CAND * DEFAULT_TITLE_LEN];
        long[] users = new long[n];
        float[] labels = new float[n * DEFAULT_NUM_CAND];
        for (int i = 0; i < n; i++) {
            users[i] = userList.get(i);
            long[][] h = histList.get(i);
            long[][] c = candList.get(i);
            float[] lab = labelList.get(i);
            for (int a = 0; a < DEFAULT_HIST_LEN; a++) {
                System.arraycopy(h[a], 0, histFlat,
                        (i * DEFAULT_HIST_LEN + a) * DEFAULT_TITLE_LEN, DEFAULT_TITLE_LEN);
            }
            for (int a = 0; a < DEFAULT_NUM_CAND; a++) {
                System.arraycopy(c[a], 0, candFlat,
                        (i * DEFAULT_NUM_CAND + a) * DEFAULT_TITLE_LEN, DEFAULT_TITLE_LEN);
                labels[i * DEFAULT_NUM_CAND + a] = lab[a];
            }
        }

        Tensor histT = torch.tensor(histFlat, optsLong()).view(n, DEFAULT_HIST_LEN, DEFAULT_TITLE_LEN);
        Tensor candT = torch.tensor(candFlat, optsLong()).view(n, DEFAULT_NUM_CAND, DEFAULT_TITLE_LEN);
        Tensor userT = torch.tensor(users, optsLong());
        Tensor labT = torch.tensor(labels, optsFloat()).view(n, DEFAULT_NUM_CAND);
        return new Split(histT, candT, userT, labT, wordVocab.size(), userVocab.size(), false);
    }

    private static long[] tokenize(String title, Map<String, Integer> vocab, int maxLen) {
        long[] ids = new long[maxLen];
        if (title == null) return ids;
        String[] toks = title.toLowerCase().replaceAll("[^a-z0-9 ]", " ").trim().split("\\s+");
        int i = 0;
        for (String t : toks) {
            if (t.isEmpty()) continue;
            if (i >= maxLen) break;
            int id = vocab.computeIfAbsent(t, k -> vocab.size());
            ids[i++] = id;
        }
        return ids;
    }

    private static File firstExisting(File dir, String... rels) {
        for (String r : rels) {
            File f = new File(dir, r);
            if (f.isFile() && f.length() > 0) return f;
        }
        return null;
    }

    public static Split generateSynthetic(int numSamples, int seed) {
        return generateSynthetic(numSamples, seed, DEFAULT_VOCAB, 1000,
                DEFAULT_HIST_LEN, DEFAULT_NUM_CAND, DEFAULT_TITLE_LEN);
    }

    public static Split generateSynthetic(int numSamples, int seed, int vocabSize, int numUsers,
                                          int histLen, int numCand, int titleLen) {
        Random rng = new Random(seed);
        int n = Math.max(numSamples, 8);
        long[] hist = new long[n * histLen * titleLen];
        long[] cand = new long[n * numCand * titleLen];
        long[] users = new long[n];
        float[] labels = new float[n * numCand];
        for (int i = 0; i < n; i++) {
            users[i] = 1 + rng.nextInt(Math.max(numUsers - 1, 1));
            for (int h = 0; h < histLen; h++) {
                // leave some pads
                int real = 2 + rng.nextInt(Math.max(titleLen - 2, 1));
                for (int t = 0; t < real; t++) {
                    hist[(i * histLen + h) * titleLen + t] = 1 + rng.nextInt(vocabSize - 1);
                }
            }
            // one positive candidate among C
            int pos = rng.nextInt(numCand);
            for (int c = 0; c < numCand; c++) {
                int real = 2 + rng.nextInt(Math.max(titleLen - 2, 1));
                for (int t = 0; t < real; t++) {
                    cand[(i * numCand + c) * titleLen + t] = 1 + rng.nextInt(vocabSize - 1);
                }
                labels[i * numCand + c] = (c == pos) ? 1f : 0f;
            }
        }
        Tensor histT = torch.tensor(hist, optsLong()).view(n, histLen, titleLen);
        Tensor candT = torch.tensor(cand, optsLong()).view(n, numCand, titleLen);
        Tensor userT = torch.tensor(users, optsLong());
        Tensor labT = torch.tensor(labels, optsFloat()).view(n, numCand);
        return new Split(histT, candT, userT, labT, vocabSize, numUsers, true);
    }

    private static TensorOptions optsLong() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
    }

    private static TensorOptions optsFloat() {
        return new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
    }
}
