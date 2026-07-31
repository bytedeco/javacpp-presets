/*
 * News recommendation feature catalog (MIND / NAML / NRMS / LSTUR / DKN shapes).
 * user interest + news content embeddings + topic ids.
 */
package org.bytedeco.pytorch.feature.industry;

import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureTable;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.Project;
import org.bytedeco.pytorch.feature.core.ValueType;
import org.bytedeco.pytorch.feature.multimodal.MultimodalFeatureView;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** News ranking feature warehouse template. */
public final class NewsCatalog implements IndustryFeatureCatalog {

    public static final String PROJECT = "news";
    public static final String SERVICE = "news_rank";

    private final List<FeatureView> views = new ArrayList<>();

    @Override public IndustryDomain domain() { return IndustryDomain.NEWS; }
    @Override public String project() { return PROJECT; }
    @Override public String primaryService() { return SERVICE; }
    @Override public List<FeatureView> featureViews() { return List.copyOf(views); }

    @Override
    public List<String> registerAll(FeatureRegistry registry) {
        views.clear();
        registry.registerProject(Project.builder(PROJECT)
                .description("News recommendation features (MIND-style)")
                .owner("news-rank")
                .build());

        Entity user = Entity.builder("user_id").project(PROJECT).valueType(ValueType.INT64).build();
        Entity news = Entity.builder("news_id").project(PROJECT).valueType(ValueType.INT64)
                .joinKey("news_id").build();
        registry.registerEntity(user);
        registry.registerEntity(news);

        FeatureView userInterest = FeatureView.builder("user_interest")
                .project(PROJECT).entities(user).ttlDays(7).online(true)
                .description("User topic interests + click history sequence (LSTUR/NPA)")
                .schema(
                        Field.of("topic_pref", ValueType.INT64_LIST),
                        Field.of("click_hist", ValueType.INT64_LIST),
                        Field.of("click_cnt_7d", ValueType.INT64),
                        Field.of("category_pref_top", ValueType.INT64))
                .source(FeatureTable.memory("user_interest"))
                .build();

        FeatureView newsMeta = FeatureView.builder("news_meta")
                .project(PROJECT).entities(news).ttlDays(14).online(true)
                .description("News category, subcategory, freshness hours")
                .schema(
                        Field.of("category_id", ValueType.INT64),
                        Field.of("subcategory_id", ValueType.INT64),
                        Field.of("freshness_hours", ValueType.FLOAT64),
                        Field.of("popularity_1d", ValueType.FLOAT64))
                .source(FeatureTable.memory("news_meta"))
                .build();

        FeatureView newsContent = MultimodalFeatureView.builder("news_content")
                .project(PROJECT).entities(news)
                .text("title", 64)
                .text("body", 128)
                .embedding("news_tower_emb", 64)
                .ttlDays(14).online(true)
                .description("Title/body text embeddings + news tower (NAML/NRMS/DKN)")
                .tag("domain", "news")
                .build().toFeatureView();

        for (FeatureView v : List.of(userInterest, newsMeta, newsContent)) {
            registry.registerFeatureView(v);
            views.add(v);
        }

        registry.registerFeatureService(FeatureService.builder(SERVICE)
                .project(PROJECT)
                .views("user_interest", "news_meta", "news_content")
                .description("News ranking feature service")
                .tag("model", "NAML/NRMS/LSTUR/DKN/NPA")
                .build());
        return List.of(SERVICE);
    }

    @Override
    public Map<String, List<Map<String, Object>>> sampleOfflineData(long nowMs, int nUsers, int nItems) {
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        List<Map<String, Object>> ui = new ArrayList<>();
        for (int u = 1; u <= nUsers; u++) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("user_id", (long) u);
            r.put("event_timestamp", nowMs - u * 45_000L);
            long[] topics = new long[]{u % 10L, (u + 1) % 10L, (u + 2) % 10L};
            long[] hist = new long[10];
            for (int i = 0; i < hist.length; i++) hist[i] = (u * 3L + i) % Math.max(1, nItems) + 1;
            r.put("topic_pref", topics);
            r.put("click_hist", hist);
            r.put("click_cnt_7d", 5L + u % 40);
            r.put("category_pref_top", (long) (u % 15));
            ui.add(r);
        }
        List<Map<String, Object>> nm = new ArrayList<>();
        List<Map<String, Object>> nc = new ArrayList<>();
        for (int i = 1; i <= nItems; i++) {
            long ts = nowMs - i * 90_000L;
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("news_id", (long) i);
            m.put("event_timestamp", ts);
            m.put("category_id", (long) (i % 15));
            m.put("subcategory_id", (long) (i % 40));
            m.put("freshness_hours", (double) (i % 72));
            m.put("popularity_1d", Math.log1p(100 + i));
            nm.add(m);

            Map<String, Object> c = new LinkedHashMap<>();
            c.put("news_id", (long) i);
            c.put("event_timestamp", ts);
            c.put("title", "news_title_" + i);
            c.put("body", "news_body_" + i);
            float[] te = new float[64];
            float[] be = new float[128];
            float[] tower = new float[64];
            for (int d = 0; d < 64; d++) {
                te[d] = (float) Math.sin(i * 0.1 + d);
                tower[d] = (float) Math.cos(i * 0.1 + d);
            }
            for (int d = 0; d < 128; d++) be[d] = (float) Math.sin(i * 0.05 + d * 0.01);
            c.put("title_emb", te);
            c.put("body_emb", be);
            c.put("news_tower_emb", tower);
            nc.add(c);
        }
        out.put("user_interest", ui);
        out.put("news_meta", nm);
        out.put("news_content", nc);
        return out;
    }
}
