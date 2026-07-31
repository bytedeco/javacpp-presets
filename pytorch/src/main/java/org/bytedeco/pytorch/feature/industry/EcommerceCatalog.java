/*
 * E-commerce feature catalog (CTR/CVR, SKU, cart, price, search).
 * Shapes from Alibaba/Taobao, Amazon, JD, Shopee ranking stacks.
 */
package org.bytedeco.pytorch.feature.industry;

import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureTable;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.feature.core.Project;
import org.bytedeco.pytorch.feature.core.ValueType;
import org.bytedeco.pytorch.feature.multimodal.MultimodalFeatureView;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;
import org.bytedeco.pytorch.feature.transform.OnDemandCompute;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** E-commerce CTR/CVR feature warehouse template. */
public final class EcommerceCatalog implements IndustryFeatureCatalog {

    public static final String PROJECT = "ecommerce";
    public static final String SERVICE = "ctr_cvr_rank";

    private final List<FeatureView> views = new ArrayList<>();

    @Override public IndustryDomain domain() { return IndustryDomain.ECOMMERCE; }
    @Override public String project() { return PROJECT; }
    @Override public String primaryService() { return SERVICE; }
    @Override public List<FeatureView> featureViews() { return List.copyOf(views); }

    @Override
    public List<String> registerAll(FeatureRegistry registry) {
        views.clear();
        registry.registerProject(Project.builder(PROJECT)
                .description("E-commerce CTR/CVR feature project").owner("ecom-rank").build());

        Entity user = Entity.builder("user_id").project(PROJECT).valueType(ValueType.INT64).build();
        Entity sku = Entity.builder("sku_id").project(PROJECT).valueType(ValueType.INT64).joinKey("sku_id").build();
        registry.registerEntity(user);
        registry.registerEntity(sku);

        FeatureView userBeh = FeatureView.builder("user_behavior")
                .project(PROJECT).entities(user).ttlDays(7).online(true)
                .description("User click/cart/order window aggs (Alibaba DIN/DIEN style)")
                .schema(
                        Field.of("click_7d", ValueType.INT64),
                        Field.of("cart_7d", ValueType.INT64),
                        Field.of("order_30d", ValueType.INT64),
                        Field.of("gmv_30d", ValueType.FLOAT64),
                        Field.of("clk_seq", ValueType.INT64_LIST))
                .source(FeatureTable.memory("user_behavior"))
                .build();

        FeatureView skuStats = FeatureView.builder("sku_stats")
                .project(PROJECT).entities(sku).ttlDays(3).online(true)
                .description("SKU price, sales, CTR proxy, brand/cate")
                .schema(
                        Field.of("price", ValueType.FLOAT64),
                        Field.of("avg_price_7d", ValueType.FLOAT64),
                        Field.of("sales_7d", ValueType.INT64),
                        Field.of("ctr_7d", ValueType.FLOAT64),
                        Field.of("brand_id", ValueType.INT64),
                        Field.of("cate_id", ValueType.INT64),
                        Field.of("shop_id", ValueType.INT64))
                .source(FeatureTable.memory("sku_stats"))
                .build();

        FeatureView searchCtx = FeatureView.builder("search_context")
                .project(PROJECT).entities(user).ttl(Duration.ofHours(1)).online(true)
                .description("Search query features when in search funnel")
                .schema(
                        Field.of("query_hash", ValueType.INT64),
                        Field.of("query_len", ValueType.INT64),
                        Field.of("query_cate_id", ValueType.INT64))
                .source(FeatureTable.memory("search_context"))
                .build();

        FeatureView skuMm = MultimodalFeatureView.builder("sku_multimodal")
                .project(PROJECT).entities(sku)
                .image("product", 64)
                .text("title", 32)
                .ttlDays(14).online(true)
                .description("Product image + title embeddings")
                .build().toFeatureView();

        for (FeatureView v : List.of(userBeh, skuStats, searchCtx, skuMm)) {
            registry.registerFeatureView(v);
            views.add(v);
        }

        OnDemandFeatureView priceOd = OnDemandFeatureView.builder("od_price_diff")
                .project(PROJECT)
                .sources("sku_stats")
                .schema(
                        Field.of("price_diff", ValueType.FLOAT64),
                        Field.of("price_ratio", ValueType.FLOAT64))
                .requestSchema(Field.of("price", ValueType.FLOAT64))
                .compute(OnDemandCompute.priceDiff("price", "sku_stats", "avg_price_7d", "price_diff"))
                .description("Request price vs SKU avg price")
                .build();
        registry.registerOnDemandFeatureView(priceOd);

        registry.registerFeatureService(FeatureService.builder(SERVICE)
                .project(PROJECT)
                .views("user_behavior", "sku_stats", "search_context", "sku_multimodal")
                .onDemandView("od_price_diff")
                .description("CTR/CVR ranking feature service (ESCM2/DBMTL/MultiDomainCTR)")
                .tag("model", "DeepFM/DIN/ESCM2")
                .build());
        return List.of(SERVICE);
    }

    @Override
    public Map<String, List<Map<String, Object>>> sampleOfflineData(long nowMs, int nUsers, int nItems) {
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        List<Map<String, Object>> ub = new ArrayList<>();
        List<Map<String, Object>> sc = new ArrayList<>();
        for (int u = 1; u <= nUsers; u++) {
            long ts = nowMs - u * 60_000L;
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("user_id", (long) u);
            r.put("event_timestamp", ts);
            r.put("click_7d", 10L + u);
            r.put("cart_7d", 1L + u % 5);
            r.put("order_30d", (long) (u % 8));
            r.put("gmv_30d", 100.0 * (u % 20));
            long[] seq = new long[8];
            for (int i = 0; i < seq.length; i++) seq[i] = (u + i) % Math.max(1, nItems) + 1;
            r.put("clk_seq", seq);
            ub.add(r);
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("user_id", (long) u);
            s.put("event_timestamp", ts);
            s.put("query_hash", (long) (u * 31 % 10000));
            s.put("query_len", 2L + u % 6);
            s.put("query_cate_id", (long) (u % 50));
            sc.add(s);
        }
        List<Map<String, Object>> sku = new ArrayList<>();
        List<Map<String, Object>> mm = new ArrayList<>();
        for (int i = 1; i <= nItems; i++) {
            long ts = nowMs - i * 120_000L;
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("sku_id", (long) i);
            r.put("event_timestamp", ts);
            double price = 9.9 + i;
            r.put("price", price);
            r.put("avg_price_7d", price * 0.95);
            r.put("sales_7d", 20L + i);
            r.put("ctr_7d", 0.02 + (i % 30) / 1000.0);
            r.put("brand_id", (long) (i % 100));
            r.put("cate_id", (long) (i % 50));
            r.put("shop_id", (long) (i % 200));
            sku.add(r);
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("sku_id", (long) i);
            m.put("event_timestamp", ts);
            m.put("product_uri", "s3://sku/" + i + ".jpg");
            m.put("title", "sku_title_" + i);
            float[] pe = new float[64];
            float[] te = new float[32];
            for (int d = 0; d < 64; d++) pe[d] = (float) Math.sin(i + d);
            for (int d = 0; d < 32; d++) te[d] = (float) Math.cos(i + d);
            m.put("product_emb", pe);
            m.put("title_emb", te);
            mm.add(m);
        }
        out.put("user_behavior", ub);
        out.put("sku_stats", sku);
        out.put("search_context", sc);
        out.put("sku_multimodal", mm);
        return out;
    }
}
