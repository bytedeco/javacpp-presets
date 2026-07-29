/*
 * Recsys / short-video feature catalog.
 *
 * Production shapes (Douyin/TikTok, YouTube, Kuaishou, Meta Reels):
 *   user_id, item_id entities
 *   user_profile, user_seq_play, item_stats, context_realtime views
 *   FeatureService shortvideo_rank for ranking models (DIN/ ent/PEPNet/WLR)
 */
package org.bytedeco.pytorch.utils.feature.industry;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureTable;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.utils.feature.core.ValueType;
import org.bytedeco.pytorch.utils.feature.multimodal.MultimodalFeatureView;
import org.bytedeco.pytorch.utils.feature.registry.FeatureRegistry;
import org.bytedeco.pytorch.utils.feature.transform.OnDemandCompute;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Short-video / general recommendation feature warehouse template. */
public final class RecsysCatalog implements IndustryFeatureCatalog {

    public static final String PROJECT = "recsys";
    public static final String SERVICE = "shortvideo_rank";

    private final List<FeatureView> views = new ArrayList<>();

    @Override
    public IndustryDomain domain() {
        return IndustryDomain.SHORT_VIDEO;
    }

    @Override
    public String project() {
        return PROJECT;
    }

    @Override
    public String primaryService() {
        return SERVICE;
    }

    @Override
    public List<FeatureView> featureViews() {
        return List.copyOf(views);
    }

    @Override
    public List<String> registerAll(FeatureRegistry registry) {
        views.clear();
        registry.registerProject(org.bytedeco.pytorch.utils.feature.core.Project.builder(PROJECT)
                .description("Recsys / short-video feature project")
                .owner("recsys-platform")
                .build());

        Entity user = Entity.builder("user_id").project(PROJECT).valueType(ValueType.INT64)
                .description("end user").build();
        Entity item = Entity.builder("item_id").project(PROJECT).valueType(ValueType.INT64)
                .description("video / item").joinKey("item_id").build();
        registry.registerEntity(user);
        registry.registerEntity(item);

        FeatureView userProfile = FeatureView.builder("user_profile")
                .project(PROJECT)
                .entities(user)
                .ttlDays(7)
                .online(true)
                .description("Static + slow-changing user profile (age bucket, gender, city tier, device)")
                .schema(
                        Field.of("age_bucket", ValueType.INT64),
                        Field.of("gender", ValueType.INT64),
                        Field.of("city_tier", ValueType.INT64),
                        Field.of("device_type", ValueType.INT64),
                        Field.of("register_days", ValueType.FLOAT64))
                .source(FeatureTable.memory("user_profile"))
                .tag("domain", "short_video")
                .build();

        FeatureView userSeq = FeatureView.builder("user_seq_play")
                .project(PROJECT)
                .entities(user)
                .ttl(Duration.ofHours(6))
                .online(true)
                .description("Recent play sequence for DIN/SIM long-seq (ByteDance/Alibaba pattern)")
                .schema(
                        Field.of("play_seq", ValueType.INT64_LIST),
                        Field.of("play_len", ValueType.INT64),
                        Field.of("finish_rate_1d", ValueType.FLOAT64),
                        Field.of("watch_time_1d", ValueType.FLOAT64))
                .source(FeatureTable.memory("user_seq_play"))
                .tag("domain", "short_video")
                .build();

        FeatureView itemStats = FeatureView.builder("item_stats")
                .project(PROJECT)
                .entities(item)
                .ttlDays(3)
                .online(true)
                .description("Item counters: vv, like, share, author, duration, category")
                .schema(
                        Field.of("vv_1d", ValueType.INT64),
                        Field.of("like_1d", ValueType.INT64),
                        Field.of("share_1d", ValueType.INT64),
                        Field.of("duration_sec", ValueType.FLOAT64),
                        Field.of("author_id", ValueType.INT64),
                        Field.of("category_id", ValueType.INT64),
                        Field.of("avg_finish_rate", ValueType.FLOAT64))
                .source(FeatureTable.memory("item_stats"))
                .tag("domain", "short_video")
                .build();

        // Multimodal: cover embedding + ASR text emb (common short-video content features)
        FeatureView itemMm = MultimodalFeatureView.builder("item_multimodal")
                .project(PROJECT)
                .entities(item)
                .image("cover", 64)
                .text("asr_title", 32)
                .embedding("item_tower_emb", 64)
                .ttlDays(7)
                .online(true)
                .description("Cover image emb + ASR/title text emb + item tower emb")
                .tag("domain", "short_video")
                .build()
                .toFeatureView();

        FeatureView contextRt = FeatureView.builder("context_realtime")
                .project(PROJECT)
                .entities(user)
                .ttl(Duration.ofMinutes(30))
                .online(true)
                .description("Session / network context (often joined with on-demand hour)")
                .schema(
                        Field.of("network_type", ValueType.INT64),
                        Field.of("app_version_bucket", ValueType.INT64),
                        Field.of("session_depth", ValueType.INT64))
                .source(FeatureTable.memory("context_realtime"))
                .build();

        for (FeatureView v : List.of(userProfile, userSeq, itemStats, itemMm, contextRt)) {
            registry.registerFeatureView(v);
            views.add(v);
        }

        OnDemandFeatureView ctxOd = OnDemandFeatureView.builder("od_time_context")
                .project(PROJECT)
                .schema(
                        Field.of("hour_of_day", ValueType.INT64),
                        Field.of("day_of_week", ValueType.INT64),
                        Field.of("is_weekend", ValueType.INT64))
                .requestSchema(Field.of("request_ts", ValueType.INT64))
                .compute(OnDemandCompute.timeContext("request_ts"))
                .description("Request-time clock features")
                .build();
        registry.registerOnDemandFeatureView(ctxOd);

        FeatureService svc = FeatureService.builder(SERVICE)
                .project(PROJECT)
                .views("user_profile", "user_seq_play", "item_stats", "item_multimodal", "context_realtime")
                .onDemandView("od_time_context")
                .description("Short-video ranking feature service")
                .owner("recsys-rank")
                .tag("model", "DIN/PEPNet/WLR")
                .build();
        registry.registerFeatureService(svc);
        return List.of(SERVICE);
    }

    @Override
    public Map<String, List<Map<String, Object>>> sampleOfflineData(long nowMs, int nUsers, int nItems) {
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        List<Map<String, Object>> userProfile = new ArrayList<>();
        List<Map<String, Object>> userSeq = new ArrayList<>();
        List<Map<String, Object>> context = new ArrayList<>();
        for (int u = 1; u <= nUsers; u++) {
            long ts = nowMs - (u % 5) * 3_600_000L;
            Map<String, Object> up = new LinkedHashMap<>();
            up.put("user_id", (long) u);
            up.put("event_timestamp", ts);
            up.put("age_bucket", (long) (u % 6));
            up.put("gender", (long) (u % 2));
            up.put("city_tier", (long) (u % 4 + 1));
            up.put("device_type", (long) (u % 3));
            up.put("register_days", (double) (u * 3 % 365));
            userProfile.add(up);

            Map<String, Object> seq = new LinkedHashMap<>();
            seq.put("user_id", (long) u);
            seq.put("event_timestamp", ts);
            long[] play = new long[Math.min(20, 5 + u % 10)];
            for (int i = 0; i < play.length; i++) play[i] = (u * 17L + i) % Math.max(1, nItems) + 1;
            seq.put("play_seq", play);
            seq.put("play_len", (long) play.length);
            seq.put("finish_rate_1d", 0.3 + (u % 50) / 100.0);
            seq.put("watch_time_1d", 60.0 + u);
            userSeq.add(seq);

            Map<String, Object> ctx = new LinkedHashMap<>();
            ctx.put("user_id", (long) u);
            ctx.put("event_timestamp", ts);
            ctx.put("network_type", (long) (u % 4));
            ctx.put("app_version_bucket", (long) (u % 10));
            ctx.put("session_depth", (long) (u % 30));
            context.add(ctx);
        }
        List<Map<String, Object>> itemStats = new ArrayList<>();
        List<Map<String, Object>> itemMm = new ArrayList<>();
        for (int i = 1; i <= nItems; i++) {
            long ts = nowMs - (i % 7) * 86_400_000L;
            Map<String, Object> is = new LinkedHashMap<>();
            is.put("item_id", (long) i);
            is.put("event_timestamp", ts);
            is.put("vv_1d", 1000L + i * 10L);
            is.put("like_1d", 50L + i);
            is.put("share_1d", 5L + i % 20);
            is.put("duration_sec", 15.0 + (i % 60));
            is.put("author_id", (long) (i % Math.max(1, nUsers) + 1));
            is.put("category_id", (long) (i % 20));
            is.put("avg_finish_rate", 0.4 + (i % 40) / 100.0);
            itemStats.add(is);

            Map<String, Object> mm = new LinkedHashMap<>();
            mm.put("item_id", (long) i);
            mm.put("event_timestamp", ts);
            mm.put("cover_uri", "s3://covers/" + i + ".jpg");
            float[] cover = new float[64];
            float[] asr = new float[32];
            float[] tower = new float[64];
            for (int d = 0; d < 64; d++) {
                cover[d] = (float) Math.sin(i * 0.01 + d);
                tower[d] = (float) Math.cos(i * 0.02 + d);
            }
            for (int d = 0; d < 32; d++) asr[d] = (float) Math.sin(i * 0.03 + d);
            mm.put("cover_emb", cover);
            mm.put("asr_title", "title_" + i);
            mm.put("asr_title_emb", asr);
            mm.put("item_tower_emb", tower);
            itemMm.add(mm);
        }
        out.put("user_profile", userProfile);
        out.put("user_seq_play", userSeq);
        out.put("item_stats", itemStats);
        out.put("item_multimodal", itemMm);
        out.put("context_realtime", context);
        return out;
    }
}
