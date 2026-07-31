/**
 * Enterprise multi-modal Feature Management Platform (Feature Store).
 *
 * <p>Industrial feature warehouse + feature store aligned with:
 * <ul>
 *   <li><b>Feast</b> — Entity, FeatureView, OnDemand/Stream views, FeatureService,
 *       online/offline stores, materialize, point-in-time join</li>
 *   <li><b>Featureform</b> — resource registration, provider abstraction, serving API</li>
 *   <li><b>Alibaba Feathub</b> — stream/batch unified descriptors, window aggregations</li>
 *   <li><b>Databricks Feature Store</b> — Feature Registry + Feature Provider</li>
 *   <li>Production practice at Meta / Google / Uber / ByteDance / Alibaba / Tencent / Didi
 *       (training-serving consistency, freshness SLO, multimodal embeddings)</li>
 * </ul>
 *
 * <p>Package map:
 * <ul>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.core} — domain model</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.registry} — metadata SoT + lifecycle</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.offline} — historical store + PIT join</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.online} — low-latency KV store</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.materialize} — offline→online sync</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.serving} — Feature Provider APIs</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.transform} — aggregations / on-demand compute</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.multimodal} — text/image/audio/embedding specs</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.lifecycle} — validation, drift, freshness</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.industry} — recsys/ecom/fintech/news/pharma catalogs</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.bridge} — recommend + DataFrame adapters</li>
 *   <li>{@link org.bytedeco.pytorch.utils.feature.benchmarks} — multi-dimension bench suite</li>
 * </ul>
 *
 * <pre>{@code
 * try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
 *     // register → put offline → materialize → getOnlineFeatures
 * }
 * }</pre>
 *
 * <p>Note: {@code FeatureDef} here is registry metadata — distinct from
 * {@link org.bytedeco.pytorch.recommend.basic.features.Feature} (model embedding input).
 * Use {@link org.bytedeco.pytorch.feature.bridge.RecommendFeatureBridge} to map.
 *
 * @see FeaturePlatform
 * @see <a href="https://docs.feast.dev/">Feast docs</a>
 * @see <a href="https://github.com/alibaba/feathub">Alibaba Feathub</a>
 */
package org.bytedeco.pytorch.feature;
