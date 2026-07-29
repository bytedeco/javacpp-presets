/*
 * Pharma / bio feature catalog (DTI, molecule, protein, assay).
 * Shapes from drug-target interaction / binding affinity pipelines
 * (Davis/KIBA-style entities; TwinTowerPPI, DnaSeqCnn, ProteinSeqEncoder).
 */
package org.bytedeco.pytorch.utils.feature.industry;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureTable;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.Project;
import org.bytedeco.pytorch.utils.feature.core.ValueType;
import org.bytedeco.pytorch.utils.feature.multimodal.MultimodalFeatureView;
import org.bytedeco.pytorch.utils.feature.registry.FeatureRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Pharma / bio DTI feature warehouse template. */
public final class PharmaBioCatalog implements IndustryFeatureCatalog {

    public static final String PROJECT = "pharma";
    public static final String SERVICE = "dti_features";

    private final List<FeatureView> views = new ArrayList<>();

    @Override public IndustryDomain domain() { return IndustryDomain.PHARMA_BIO; }
    @Override public String project() { return PROJECT; }
    @Override public String primaryService() { return SERVICE; }
    @Override public List<FeatureView> featureViews() { return List.copyOf(views); }

    @Override
    public List<String> registerAll(FeatureRegistry registry) {
        views.clear();
        registry.registerProject(Project.builder(PROJECT)
                .description("Pharma / bio DTI and assay features")
                .owner("bio-ml")
                .tag("domain", "pharma")
                .build());

        Entity compound = Entity.builder("compound_id").project(PROJECT).valueType(ValueType.INT64)
                .description("small molecule / drug compound").build();
        Entity target = Entity.builder("target_id").project(PROJECT).valueType(ValueType.INT64)
                .joinKey("target_id").description("protein / gene target").build();
        registry.registerEntity(compound);
        registry.registerEntity(target);

        FeatureView compoundTab = FeatureView.builder("compound_props")
                .project(PROJECT).entities(compound).ttlDays(30).online(true)
                .description("Physicochemical properties (MW, LogP, HBD/HBA — tabular)")
                .schema(
                        Field.of("mol_weight", ValueType.FLOAT64),
                        Field.of("logp", ValueType.FLOAT64),
                        Field.of("hbd", ValueType.INT64),
                        Field.of("hba", ValueType.INT64),
                        Field.of("tpsa", ValueType.FLOAT64),
                        Field.of("rotatable_bonds", ValueType.INT64))
                .source(FeatureTable.memory("compound_props"))
                .build();

        FeatureView compoundMm = MultimodalFeatureView.builder("compound_multimodal")
                .project(PROJECT).entities(compound)
                .text("smiles", 128)
                .embedding("mol_graph_emb", 64)
                .ttlDays(30).online(true)
                .description("SMILES string + molecular graph embedding")
                .tag("domain", "pharma")
                .build().toFeatureView();

        FeatureView targetTab = FeatureView.builder("target_props")
                .project(PROJECT).entities(target).ttlDays(30).online(true)
                .description("Target family, organism, sequence length")
                .schema(
                        Field.of("family_id", ValueType.INT64),
                        Field.of("organism_id", ValueType.INT64),
                        Field.of("seq_len", ValueType.INT64),
                        Field.of("is_enzyme", ValueType.INT64))
                .source(FeatureTable.memory("target_props"))
                .build();

        FeatureView targetMm = MultimodalFeatureView.builder("target_multimodal")
                .project(PROJECT).entities(target)
                .text("fasta", 256)
                .embedding("protein_seq_emb", 128)
                .ttlDays(30).online(true)
                .description("Protein sequence + encoder embedding")
                .build().toFeatureView();

        FeatureView assay = FeatureView.builder("assay_stats")
                .project(PROJECT).entities(compound, target).ttlDays(14).online(true)
                .description("Historical assay aggregates for compound-target pair")
                .schema(
                        Field.of("assay_count", ValueType.INT64),
                        Field.of("mean_pKi", ValueType.FLOAT64),
                        Field.of("max_pKi", ValueType.FLOAT64),
                        Field.of("std_pKi", ValueType.FLOAT64))
                .source(FeatureTable.memory("assay_stats"))
                .build();

        for (FeatureView v : List.of(compoundTab, compoundMm, targetTab, targetMm, assay)) {
            registry.registerFeatureView(v);
            views.add(v);
        }

        registry.registerFeatureService(FeatureService.builder(SERVICE)
                .project(PROJECT)
                .views("compound_props", "compound_multimodal", "target_props", "target_multimodal", "assay_stats")
                .description("Drug-target interaction feature service")
                .tag("model", "TwinTowerPPI/DnaSeqCnn/ProteinSeqEncoder")
                .build());
        return List.of(SERVICE);
    }

    @Override
    public Map<String, List<Map<String, Object>>> sampleOfflineData(long nowMs, int nUsers, int nItems) {
        // nUsers ~ compounds, nItems ~ targets
        int nComp = Math.max(1, nUsers);
        int nTgt = Math.max(1, nItems);
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();

        List<Map<String, Object>> cp = new ArrayList<>();
        List<Map<String, Object>> cm = new ArrayList<>();
        for (int c = 1; c <= nComp; c++) {
            long ts = nowMs - c * 100_000L;
            Map<String, Object> p = new LinkedHashMap<>();
            p.put("compound_id", (long) c);
            p.put("event_timestamp", ts);
            p.put("mol_weight", 150.0 + c * 3);
            p.put("logp", 1.0 + (c % 50) / 10.0);
            p.put("hbd", (long) (c % 5));
            p.put("hba", (long) (c % 8));
            p.put("tpsa", 20.0 + c);
            p.put("rotatable_bonds", (long) (c % 12));
            cp.add(p);

            Map<String, Object> m = new LinkedHashMap<>();
            m.put("compound_id", (long) c);
            m.put("event_timestamp", ts);
            m.put("smiles", "CCO" + c);
            float[] ge = new float[64];
            float[] se = new float[128];
            for (int d = 0; d < 64; d++) ge[d] = (float) Math.sin(c * 0.07 + d);
            for (int d = 0; d < 128; d++) se[d] = (float) Math.cos(c * 0.03 + d * 0.01);
            m.put("mol_graph_emb", ge);
            m.put("smiles_emb", se);
            cm.add(m);
        }

        List<Map<String, Object>> tp = new ArrayList<>();
        List<Map<String, Object>> tm = new ArrayList<>();
        for (int t = 1; t <= nTgt; t++) {
            long ts = nowMs - t * 110_000L;
            Map<String, Object> p = new LinkedHashMap<>();
            p.put("target_id", (long) t);
            p.put("event_timestamp", ts);
            p.put("family_id", (long) (t % 20));
            p.put("organism_id", (long) (t % 5));
            p.put("seq_len", 100L + t * 10L);
            p.put("is_enzyme", (long) (t % 2));
            tp.add(p);

            Map<String, Object> m = new LinkedHashMap<>();
            m.put("target_id", (long) t);
            m.put("event_timestamp", ts);
            m.put("fasta", "MKTAYIAK" + t);
            float[] pe = new float[128];
            float[] fe = new float[256];
            for (int d = 0; d < 128; d++) pe[d] = (float) Math.sin(t * 0.05 + d);
            for (int d = 0; d < 256; d++) fe[d] = (float) Math.cos(t * 0.02 + d * 0.005);
            m.put("protein_seq_emb", pe);
            m.put("fasta_emb", fe);
            tm.add(m);
        }

        List<Map<String, Object>> assay = new ArrayList<>();
        int pairs = Math.min(nComp * nTgt, Math.max(nComp, nTgt) * 3);
        int added = 0;
        for (int c = 1; c <= nComp && added < pairs; c++) {
            for (int t = 1; t <= nTgt && added < pairs; t++) {
                if ((c + t) % 3 != 0) continue;
                Map<String, Object> a = new LinkedHashMap<>();
                a.put("compound_id", (long) c);
                a.put("target_id", (long) t);
                a.put("event_timestamp", nowMs - (c + t) * 50_000L);
                a.put("assay_count", 1L + (c + t) % 5);
                double pki = 5.0 + ((c * 7 + t * 3) % 40) / 10.0;
                a.put("mean_pKi", pki);
                a.put("max_pKi", pki + 0.5);
                a.put("std_pKi", 0.2 + (c % 5) / 20.0);
                assay.add(a);
                added++;
            }
        }

        out.put("compound_props", cp);
        out.put("compound_multimodal", cm);
        out.put("target_props", tp);
        out.put("target_multimodal", tm);
        out.put("assay_stats", assay);
        return out;
    }
}
