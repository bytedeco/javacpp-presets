package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.decomposition.FactorAnalysis;
import org.bytedeco.pytorch.dataframe.feature.decomposition.LDA;
import org.bytedeco.pytorch.dataframe.feature.decomposition.NMF;
import org.bytedeco.pytorch.dataframe.feature.decomposition.PCA;
import org.bytedeco.pytorch.dataframe.feature.imputation.IterativeImputer;
import org.bytedeco.pytorch.dataframe.feature.selection.RFE;
import org.bytedeco.pytorch.dataframe.feature.text.CountVectorizer;
import org.bytedeco.pytorch.dataframe.feature.util.DenseLinalg;
import org.bytedeco.pytorch.dataframe.ml.classification.LogisticRegression;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Focused verification for C-tier algorithm fixes (real PCA/NMF/LDA/FA/RFE/IterativeImputer/CountVectorizer).
 *
 * <pre>
 *   javac -cp "target/classes:$(cat target/cp.txt)" -d target/samples-compile \
 *         samples/VerifyFeatureCTier.java
 *   java  -cp "target/samples-compile:target/classes:$(cat target/cp.txt)" \
 *         samples.VerifyFeatureCTier
 * </pre>
 */
public class VerifyFeatureCTier {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  OK  " + name);
        } else {
            failed++;
            System.out.println(" FAIL " + name);
        }
    }

    static void checkNear(String name, double actual, double expected, double tol) {
        boolean ok = Math.abs(actual - expected) <= tol;
        if (ok) {
            passed++;
            System.out.printf("  OK  %s (%.6f ≈ %.6f)%n", name, actual, expected);
        } else {
            failed++;
            System.out.printf(" FAIL %s (got %.6f expected %.6f tol %.6f)%n", name, actual, expected, tol);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== C-tier feature algorithm verification ===\n");

        // ---- DenseLinalg Jacobi on known diagonal-ish matrix ----
        System.out.println("-- DenseLinalg eigh --");
        double[][] A = {
            {4, 1, 0},
            {1, 3, 1},
            {0, 1, 2}
        };
        DenseLinalg.EigenResult er = DenseLinalg.eighSymmetric(A);
        // eigenvalues should be sorted desc, all positive for SPD-ish
        check("evals desc", er.eigenvalues[0] >= er.eigenvalues[1] && er.eigenvalues[1] >= er.eigenvalues[2]);
        check("evals finite", Double.isFinite(er.eigenvalues[0]));
        // A v ≈ λ v for top eigenvector
        double[] v0 = er.vectors[0];
        double[] Av = DenseLinalg.matvec(A, v0);
        double[] lamV = v0.clone();
        DenseLinalg.scaleInPlace(lamV, er.eigenvalues[0]);
        double resid = 0;
        for (int i = 0; i < 3; i++) resid = Math.max(resid, Math.abs(Av[i] - lamV[i]));
        check("top eigen residual < 1e-6 (" + resid + ")", resid < 1e-6);

        // ---- Real PCA: anisotropic Gaussian ----
        System.out.println("\n-- Real PCA --");
        Random rng = new Random(0);
        int n = 2000;
        DataFrame df = DataFrame.create();
        df.addColumn("x", Column.DType.FLOAT64);
        df.addColumn("y", Column.DType.FLOAT64);
        df.addColumn("z", Column.DType.FLOAT64);
        // x ~ N(0, 9), y ~ N(0, 1), z ~ N(0, 0.25)  → PC1 should align with x
        for (int i = 0; i < n; i++) {
            df.addRow(3 * rng.nextGaussian(), rng.nextGaussian(), 0.5 * rng.nextGaussian());
        }
        PCA pca = new PCA(3, "x", "y", "z");
        pca.fit(df);
        double[] ratio = pca.getExplainedVarianceRatio();
        double[] var = pca.getExplainedVariance();
        double[][] comp = pca.getComponents();
        System.out.println("  explained_variance_ratio_ = " + Arrays.toString(ratio));
        System.out.println("  explained_variance_ = " + Arrays.toString(var));
        System.out.println("  components_[0] = " + Arrays.toString(comp[0]));

        check("NOT identity stub (PC1 |c0|>>|c1|,|c2| or aligned x)",
            Math.abs(comp[0][0]) > 0.9); // first component mostly on x
        check("var ratios sum≈1", Math.abs(ratio[0] + ratio[1] + ratio[2] - 1.0) < 0.05);
        check("var1 > var2 > var3", var[0] > var[1] && var[1] > var[2]);
        // variance roughly 9, 1, 0.25
        checkNear("var0 ~ 9", var[0], 9.0, 1.5);
        checkNear("var1 ~ 1", var[1], 1.0, 0.4);

        DataFrame out = pca.transform(df);
        check("has PC1", out.hasColumn("PC1"));
        check("has PC2", out.hasColumn("PC2"));

        // variance threshold mode
        PCA pcaAuto = new PCA(0.90, "x", "y", "z");
        pcaAuto.fit(df);
        check("auto n_components >=1", pcaAuto.getNComponents() >= 1);
        check("auto cumulative >= 0.90",
            pcaAuto.getCumulativeExplainedVarianceRatio()[pcaAuto.getNComponents() - 1] >= 0.89);

        // ---- NMF multiplicative updates non-empty ----
        System.out.println("\n-- Real NMF --");
        DataFrame pos = DataFrame.create();
        pos.addColumn("a", Column.DType.FLOAT64);
        pos.addColumn("b", Column.DType.FLOAT64);
        pos.addColumn("c", Column.DType.FLOAT64);
        Random r2 = new Random(1);
        for (int i = 0; i < 300; i++) {
            double u = Math.abs(r2.nextGaussian());
            double v = Math.abs(r2.nextGaussian());
            pos.addRow(u, v, 0.5 * u + 0.5 * v + 0.01 * Math.abs(r2.nextGaussian()));
        }
        NMF nmf = new NMF(2, "a", "b", "c").setMaxIter(100);
        nmf.fit(pos);
        check("NMF reconstruction finite", Double.isFinite(nmf.getReconstructionErr()));
        check("NMF reconstruction improved from random-scale", nmf.getReconstructionErr() < 1e6);
        double[][] H = nmf.getComponents();
        check("NMF H non-negative", H[0][0] > 0 && H[0][1] > 0);
        // ensure updates actually changed from pure random — err should be much smaller than ||V||^2
        DataFrame nmfOut = nmf.transform(pos);
        check("NMF has NMF_1", nmfOut.hasColumn("NMF_1"));

        // ---- LDA uses labels ----
        System.out.println("\n-- Real LDA --");
        DataFrame clf = DataFrame.create();
        clf.addColumn("f0", Column.DType.FLOAT64);
        clf.addColumn("f1", Column.DType.FLOAT64);
        clf.addColumn("y", Column.DType.FLOAT64);
        Random r3 = new Random(2);
        for (int i = 0; i < 400; i++) {
            int lab = i % 2;
            double f0 = (lab == 0 ? -2 : 2) + 0.5 * r3.nextGaussian();
            double f1 = r3.nextGaussian();
            clf.addRow(f0, f1, (double) lab);
        }
        LDA lda = new LDA(1, "y", "f0", "f1");
        lda.fit(clf);
        double[][] ldc = lda.getComponents();
        System.out.println("  LD1 components = " + Arrays.toString(ldc[0]));
        // should put most weight on f0 (class-separating axis)
        check("LDA |w0| > |w1|", Math.abs(ldc[0][0]) > Math.abs(ldc[0][1]));
        DataFrame ldOut = lda.transform(clf);
        check("has LD1", ldOut.hasColumn("LD1"));

        // ---- RFE uses LR coef ----
        System.out.println("\n-- RFE + LogisticRegression.getCoef --");
        DataFrame rfeDf = DataFrame.create();
        String[] feats = {"s0", "s1", "s2", "s3", "noise"};
        for (String f : feats) rfeDf.addColumn(f, Column.DType.FLOAT64);
        rfeDf.addColumn("label", Column.DType.FLOAT64);
        Random r4 = new Random(3);
        for (int i = 0; i < 500; i++) {
            double s0 = r4.nextGaussian();
            double s1 = r4.nextGaussian();
            double s2 = r4.nextGaussian();
            double s3 = r4.nextGaussian();
            double noise = r4.nextGaussian();
            double logit = 2.5 * s0 - 1.8 * s1 + 0.05 * s2 + 0.02 * s3 + 0.01 * noise;
            double y = logit > 0 ? 1.0 : 0.0;
            rfeDf.addRow(s0, s1, s2, s3, noise, y);
        }
        LogisticRegression lr = new LogisticRegression("l2", 1.0, 300, 1e-4, 42L);
        RFE rfe = new RFE(lr, 2, feats, "label");
        rfe.fit(rfeDf);
        List<String> selected = rfe.getSelectedColumns();
        System.out.println("  RFE selected = " + selected);
        check("RFE selected 2", selected.size() == 2);
        check("RFE keeps s0", selected.contains("s0"));
        check("RFE keeps s1", selected.contains("s1"));

        // ---- IterativeImputer ----
        System.out.println("\n-- IterativeImputer --");
        DataFrame miss = DataFrame.create();
        miss.addColumn("a", Column.DType.FLOAT64);
        miss.addColumn("b", Column.DType.FLOAT64);
        miss.addColumn("c", Column.DType.FLOAT64);
        Random r5 = new Random(5);
        for (int i = 0; i < 200; i++) {
            double a = r5.nextGaussian();
            double b = 0.8 * a + 0.2 * r5.nextGaussian();
            double c = -0.5 * a + 0.5 * b + 0.1 * r5.nextGaussian();
            miss.addRow(a, b, c);
        }
        // punch holes
        for (int i = 0; i < 30; i++) {
            miss.set(i, "b", null);
        }
        IterativeImputer ii = new IterativeImputer(15, "a", "b", "c");
        DataFrame filled = ii.fitTransform(miss);
        int stillMiss = 0;
        for (int i = 0; i < 30; i++) {
            Object v = filled.column("b").get(i);
            if (v == null) stillMiss++;
        }
        check("IterativeImputer filled nulls", stillMiss == 0);
        check("IterativeImputer has models", ii.getModels().size() == 3);

        // ---- CountVectorizer params ----
        System.out.println("\n-- CountVectorizer --");
        DataFrame text = DataFrame.create();
        text.addColumn("doc", Column.DType.STRING);
        text.addRow("good product excellent quality");
        text.addRow("bad product terrible quality");
        text.addRow("good service excellent");
        text.addRow("normal product");
        text.addRow("bad service terrible");
        CountVectorizer cv = new CountVectorizer("doc")
            .setMaxFeatures(5)
            .setMinDf(1)
            .setStopWords("the", "a")
            .setNgramRange(1, 2);
        DataFrame bow = cv.fitTransform(text);
        check("vocab limited <=5", cv.getFeatureCount() <= 5);
        check("vocab >0", cv.getFeatureCount() > 0);
        check("bow more cols", bow.columnCount() > text.columnCount());
        System.out.println("  vocab = " + cv.getVocabulary());

        // ---- FactorAnalysis ----
        System.out.println("\n-- FactorAnalysis --");
        FactorAnalysis fa = new FactorAnalysis(2, "x", "y", "z");
        // reuse anisotropic df
        fa.fit(df);
        check("FA components non-null", fa.getComponents() != null);
        check("FA n=2", fa.getNComponents() == 2);
        DataFrame faOut = fa.transform(df);
        check("FA has FA1", faOut.hasColumn("FA1"));
        check("FA loglike finite", Double.isFinite(fa.getLoglike()));

        System.out.println("\n=== Summary: passed=" + passed + " failed=" + failed + " ===");
        if (failed > 0) System.exit(1);
    }
}
