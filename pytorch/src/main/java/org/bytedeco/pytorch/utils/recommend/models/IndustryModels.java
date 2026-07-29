/*
 * IndustryModels — catalog of industry-specialized recommend / ranking models.
 *
 * This class is documentation + discovery only (no runtime logic). It lists
 * models implemented under org.bytedeco.pytorch.utils.recommend.models.* for
 * e-commerce, news, short-video, live, fintech, pharma and bioinformatics,
 * with paper / production references so callers do not "guess" origins.
 *
 * Shared industrial layers live in:
 *   org.bytedeco.pytorch.utils.recommend.basic.layers.industry
 *
 * Generic matching / ranking / multi_task / generative models remain under
 * their existing packages (DIN, ESMM, MIND, SASRec, ...).
 */
package org.bytedeco.pytorch.utils.recommend.models;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class IndustryModels {

    private IndustryModels() {}

    public enum Industry {
        ECOMMERCE,
        NEWS,
        SHORT_VIDEO,
        LIVE,
        FINTECH,
        PHARMA,
        BIO
    }

    /** Immutable catalog entry. */
    public static final class Entry {
        public final String className;
        public final Industry industry;
        public final String paperOrSystem;
        public final String oneLine;

        public Entry(String className, Industry industry, String paperOrSystem, String oneLine) {
            this.className = className;
            this.industry = industry;
            this.paperOrSystem = paperOrSystem;
            this.oneLine = oneLine;
        }

        @Override
        public String toString() {
            return className + " [" + industry + "] — " + oneLine + " (" + paperOrSystem + ")";
        }
    }

    private static final List<Entry> CATALOG = Collections.unmodifiableList(Arrays.asList(
            // ---- News (Microsoft MIND family) ----
            new Entry("news.NRMS", Industry.NEWS,
                    "Wu et al., EMNLP 2019 (Microsoft MIND)",
                    "Multi-head self-attention news + user encoders, dot-product ranking"),
            new Entry("news.NAML", Industry.NEWS,
                    "Wu et al., IJCAI 2019",
                    "Attentive multi-view learning (title/abstract/category)"),
            new Entry("news.LSTUR", Industry.NEWS,
                    "An et al., ACL 2019",
                    "Long-term user-id preference + short-term GRU over clicks"),
            new Entry("news.NPA", Industry.NEWS,
                    "Wu et al., WWW 2019",
                    "Personalized attention queries conditioned on user id"),
            new Entry("news.DKN", Industry.NEWS,
                    "Wang et al., WWW 2018",
                    "Knowledge-aware CNN with entity embeddings + candidate attention"),

            // ---- Short video ----
            new Entry("shortvideo.WLR", Industry.SHORT_VIDEO,
                    "Covington et al., RecSys 2016 (YouTube); industrial watch-time",
                    "Watch-time weighted logistic ranking tower (+ optional D2Q)"),
            new Entry("shortvideo.D2Q", Industry.SHORT_VIDEO,
                    "Zhan et al., KDD 2022 (Kuaishou)",
                    "Duration-bucket deconfounded watch-time + interest heads"),
            new Entry("shortvideo.PEPNet", Industry.SHORT_VIDEO,
                    "Chang et al., KDD/CIKM industrial (Kuaishou PEPNet)",
                    "EPNet+PPNet gates for multi-scenario multi-task ranking"),

            // ---- Live ----
            new Entry("live.LiveMultiTask", Industry.LIVE,
                    "ESMM (SIGIR'18) + industrial live multi-task practice",
                    "CTR / stay / gift CVR(CTCVR) / follow entire-space multi-task"),

            // ---- E-commerce ----
            new Entry("ecommerce.ESCM2", Industry.ECOMMERCE,
                    "ESCM2 SIGIR 2022 (Alibaba); ESMM SIGIR 2018; Chapelle KDD 2014",
                    "Entire-space CVR with counterfactual head + optional delayed feedback / domain"),
            new Entry("ecommerce.MultiDomainCTR", Industry.ECOMMERCE,
                    "Sheng et al., CIKM 2021 STAR (Alibaba)",
                    "Star-topology domain-adaptive CTR (one model, many domains)"),
            new Entry("ecommerce.DBMTL", Industry.ECOMMERCE,
                    "Uncertainty-weighted MTL (Kendall CVPR'18) + ESMM industrial ranking",
                    "Shared-bottom multi-task CTR/CVR/CTCVR/aux with learnable loss weights"),
            new Entry("ecommerce.SearchConversion", Industry.ECOMMERCE,
                    "Industrial search CTR/CVR (Amazon/Taobao search + ESMM)",
                    "Query-token attentive pool + item EmbeddingLayer dual-task CTCVR"),

            // ---- Fintech ----
            new Entry("fintech.TabTransformer", Industry.FINTECH,
                    "Huang et al., arXiv 2020 (Amazon AWS AI)",
                    "Contextual embeddings over categorical columns + continuous LN"),
            new Entry("fintech.FTTransformer", Industry.FINTECH,
                    "Gorishniy et al., NeurIPS 2021",
                    "Feature Tokenizer + Transformer with [CLS] for tabular risk/fraud"),
            new Entry("fintech.SequenceRiskModel", Industry.FINTECH,
                    "Industrial sequential fraud / risk (Alipay-style event sequences)",
                    "Event-sequence Transformer + amount/time side features → risk score"),
            // existing: ranking.FraudGNN

            // ---- Pharma ----
            new Entry("pharma.DeepDTA", Industry.PHARMA,
                    "Öztürk et al., Bioinformatics 2018",
                    "CNN drug SMILES + CNN protein AA → binding affinity regression"),
            new Entry("pharma.MolTrans", Industry.PHARMA,
                    "Huang et al., Bioinformatics 2021",
                    "Transformer encoders + interaction-map CNN for DTI"),
            new Entry("pharma.GraphDrugEncoder", Industry.PHARMA,
                    "GraphDTA (Nguyen et al., Bioinformatics 2021) GCN drug encoder",
                    "Atom-feature GCN + mean pool for molecular graphs (DrugBAN/DeepDTA tower)"),
            new Entry("pharma.DrugBAN", Industry.PHARMA,
                    "DrugBAN bilinear attention DTI (Bai et al. line of work)",
                    "Bilinear attention between drug fragments and protein residues"),

            // ---- Bio ----
            new Entry("bio.ProteinSeqEncoder", Industry.BIO,
                    "Interface aligned with ESM (Rives et al., PNAS 2021) / ProtTrans",
                    "Lightweight residue Transformer encoder (not full ESM weights)"),
            new Entry("bio.TwinTowerPPI", Industry.BIO,
                    "Siamese PPI (e.g. Chen et al., Bioinformatics 2019 style)",
                    "Shared protein encoder twin-tower interaction classifier"),
            new Entry("bio.GeneExpressionMLP", Industry.BIO,
                    "TCGA / GDSC-style expression MLP (Way & Greene PSB 2018 context)",
                    "LayerNorm + MLP multi-task heads on gene-expression vectors"),
            new Entry("bio.DnaSeqCnn", Industry.BIO,
                    "DeepBind (Alipanahi et al., Nat Biotech 2015) / DeepSEA-style CNN",
                    "Multi-kernel Conv1d over nucleotide sequences for regulatory genomics")
    ));

    public static List<Entry> catalog() {
        return CATALOG;
    }

    public static List<Entry> byIndustry(Industry industry) {
        List<Entry> out = new java.util.ArrayList<>();
        for (Entry e : CATALOG) {
            if (e.industry == industry) out.add(e);
        }
        return Collections.unmodifiableList(out);
    }

    public static Map<Industry, Integer> counts() {
        Map<Industry, Integer> m = new LinkedHashMap<>();
        for (Industry ind : Industry.values()) m.put(ind, 0);
        for (Entry e : CATALOG) m.put(e.industry, m.get(e.industry) + 1);
        return Collections.unmodifiableMap(m);
    }

    /** Human-readable summary for logs / docs. */
    public static String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Industry recommend models (").append(CATALOG.size()).append("):\n");
        for (Industry ind : Industry.values()) {
            List<Entry> list = byIndustry(ind);
            if (list.isEmpty()) continue;
            sb.append("  ").append(ind).append(" (").append(list.size()).append(")\n");
            for (Entry e : list) {
                sb.append("    - ").append(e.className).append(": ").append(e.oneLine).append('\n');
            }
        }
        sb.append("Shared layers: basic.layers.industry.{AdditiveAttention, MultiHeadSelfAttention,")
                .append(" GateFusion, DurationDeconfoundHead, DelayedFeedbackHead, DomainAdapter}\n");
        sb.append("Also reuse: matching/*, ranking/* (DIN/DIEN/SIM/FraudGNN), multi_task/* (ESMM/MMoE/PLE),")
                .append(" generative/* (HSTU/TIGER)\n");
        return sb.toString();
    }
}
