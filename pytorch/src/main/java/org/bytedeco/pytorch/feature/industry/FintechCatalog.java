/*
 * Fintech / anti-fraud feature catalog.
 * Shapes from banking risk, payment fraud (device graph, txn windows, account).
 * Models: SequenceRiskModel, FTTransformer, TabTransformer, FraudGNN.
 */
package org.bytedeco.pytorch.feature.industry;

import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureTable;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.Project;
import org.bytedeco.pytorch.feature.core.StreamFeatureView;
import org.bytedeco.pytorch.feature.core.ValueType;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Fintech risk / fraud feature warehouse template. */
public final class FintechCatalog implements IndustryFeatureCatalog {

    public static final String PROJECT = "fintech";
    public static final String SERVICE = "fraud_score_fs";

    private final List<FeatureView> views = new ArrayList<>();

    @Override public IndustryDomain domain() { return IndustryDomain.FINTECH; }
    @Override public String project() { return PROJECT; }
    @Override public String primaryService() { return SERVICE; }
    @Override public List<FeatureView> featureViews() { return List.copyOf(views); }

    @Override
    public List<String> registerAll(FeatureRegistry registry) {
        views.clear();
        registry.registerProject(Project.builder(PROJECT)
                .description("Fintech fraud / credit risk features")
                .owner("risk-platform")
                .tag("compliance", "pii-aware")
                .build());

        Entity account = Entity.builder("account_id").project(PROJECT).valueType(ValueType.INT64)
                .description("payment / bank account").build();
        Entity device = Entity.builder("device_id").project(PROJECT).valueType(ValueType.STRING)
                .joinKey("device_id").description("device fingerprint hash").build();
        registry.registerEntity(account);
        registry.registerEntity(device);

        FeatureView acctProfile = FeatureView.builder("account_profile")
                .project(PROJECT).entities(account).ttlDays(30).online(true)
                .description("KYC / account static risk attributes")
                .schema(
                        Field.of("account_age_days", ValueType.FLOAT64),
                        Field.of("kyc_level", ValueType.INT64),
                        Field.of("country_risk", ValueType.FLOAT64),
                        Field.of("has_linked_card", ValueType.INT64))
                .source(FeatureTable.memory("account_profile"))
                .tag("domain", "fintech")
                .build();

        FeatureView txnWindow = FeatureView.builder("txn_window_aggs")
                .project(PROJECT).entities(account).ttl(Duration.ofHours(6)).online(true)
                .description("Transaction window aggregates (1h/24h amount, count, unique merchants)")
                .schema(
                        Field.of("txn_cnt_1h", ValueType.INT64),
                        Field.of("txn_amt_1h", ValueType.FLOAT64),
                        Field.of("txn_cnt_24h", ValueType.INT64),
                        Field.of("txn_amt_24h", ValueType.FLOAT64),
                        Field.of("uniq_merchant_24h", ValueType.INT64),
                        Field.of("max_txn_24h", ValueType.FLOAT64),
                        Field.of("night_txn_ratio_7d", ValueType.FLOAT64))
                .source(FeatureTable.memory("txn_window_aggs"))
                .build();

        FeatureView deviceRisk = FeatureView.builder("device_risk")
                .project(PROJECT).entities(device).ttlDays(7).online(true)
                .description("Device graph degree, emulator flags, IP risk")
                .schema(
                        Field.of("device_acct_degree", ValueType.INT64),
                        Field.of("is_emulator", ValueType.INT64),
                        Field.of("ip_risk_score", ValueType.FLOAT64),
                        Field.of("device_age_days", ValueType.FLOAT64))
                .source(FeatureTable.memory("device_risk"))
                .build();

        // Stream view for real-time txn events (descriptor; materialize via batch sim)
        StreamFeatureView txnStream = StreamFeatureView.builder("txn_stream_aggs")
                .project(PROJECT)
                .entities(account)
                .ttl(Duration.ofHours(1))
                .online(true)
                .source(FeatureTable.kafka("txn_events", "fintech.txn.events"))
                .schema(
                        Field.of("rt_txn_cnt_5m", ValueType.INT64),
                        Field.of("rt_txn_amt_5m", ValueType.FLOAT64))
                .aggregation("COUNT(*) OVER TUMBLE 5m GROUP BY account_id")
                .aggregation("SUM(amount) OVER TUMBLE 5m GROUP BY account_id")
                .description("Near-real-time tumbling window txn features")
                .build();
        registry.registerStreamFeatureView(txnStream);

        for (FeatureView v : List.of(acctProfile, txnWindow, deviceRisk)) {
            registry.registerFeatureView(v);
            views.add(v);
        }
        views.add(txnStream.asBatchView());

        registry.registerFeatureService(FeatureService.builder(SERVICE)
                .project(PROJECT)
                .views("account_profile", "txn_window_aggs", "device_risk", "txn_stream_aggs")
                .description("Fraud scoring feature service")
                .tag("model", "SequenceRisk/FTTransformer/FraudGNN")
                .build());
        return List.of(SERVICE);
    }

    @Override
    public Map<String, List<Map<String, Object>>> sampleOfflineData(long nowMs, int nUsers, int nItems) {
        // nUsers ~ accounts, nItems unused (use as device count proxy)
        int nAcct = Math.max(1, nUsers);
        int nDev = Math.max(1, nItems);
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        List<Map<String, Object>> ap = new ArrayList<>();
        List<Map<String, Object>> tw = new ArrayList<>();
        List<Map<String, Object>> ts = new ArrayList<>();
        for (int a = 1; a <= nAcct; a++) {
            long tsMs = nowMs - a * 30_000L;
            Map<String, Object> p = new LinkedHashMap<>();
            p.put("account_id", (long) a);
            p.put("event_timestamp", tsMs);
            p.put("account_age_days", 30.0 + a);
            p.put("kyc_level", (long) (a % 3 + 1));
            p.put("country_risk", 0.1 + (a % 10) / 50.0);
            p.put("has_linked_card", (long) (a % 2));
            ap.add(p);

            Map<String, Object> w = new LinkedHashMap<>();
            w.put("account_id", (long) a);
            w.put("event_timestamp", tsMs);
            w.put("txn_cnt_1h", (long) (a % 5));
            w.put("txn_amt_1h", 50.0 * (a % 7));
            w.put("txn_cnt_24h", 5L + a % 20);
            w.put("txn_amt_24h", 200.0 * (a % 15));
            w.put("uniq_merchant_24h", 1L + a % 8);
            w.put("max_txn_24h", 100.0 + a * 3);
            w.put("night_txn_ratio_7d", (a % 10) / 20.0);
            tw.add(w);

            Map<String, Object> s = new LinkedHashMap<>();
            s.put("account_id", (long) a);
            s.put("event_timestamp", tsMs);
            s.put("rt_txn_cnt_5m", (long) (a % 3));
            s.put("rt_txn_amt_5m", 20.0 * (a % 4));
            ts.add(s);
        }
        List<Map<String, Object>> dr = new ArrayList<>();
        for (int d = 1; d <= nDev; d++) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("device_id", "dev_" + d);
            r.put("event_timestamp", nowMs - d * 60_000L);
            r.put("device_acct_degree", (long) (1 + d % 5));
            r.put("is_emulator", (long) (d % 17 == 0 ? 1 : 0));
            r.put("ip_risk_score", (d % 100) / 100.0);
            r.put("device_age_days", 10.0 + d);
            dr.add(r);
        }
        out.put("account_profile", ap);
        out.put("txn_window_aggs", tw);
        out.put("device_risk", dr);
        out.put("txn_stream_aggs", ts);
        return out;
    }
}
