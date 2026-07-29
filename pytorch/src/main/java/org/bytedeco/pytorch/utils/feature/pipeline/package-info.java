/**
 * DataFrame ↔ Feature Store pipeline glue.
 *
 * <p>Closes the loop between table-level feature engineering
 * ({@link org.bytedeco.pytorch.dataframe.feature.FeatureEngineering} /
 * {@link org.bytedeco.pytorch.dataframe.feature.pipeline.Pipeline}) and the
 * enterprise feature platform (registry / offline / online / serving):
 *
 * <ol>
 *   <li>{@link FeatureIngest} — engineered {@code DataFrame} → FeatureView offline rows
 *       (auto-register schema, validate, put/replace)</li>
 *   <li>{@link FeatureMaterializeJob} — offline → online materialize (+ freshness)</li>
 *   <li>{@link FeatureTrainingExport} — entity DataFrame + FeatureService → PIT join
 *       → {@code TrainingDataset} / DataFrame / recommend {@code Batch} list</li>
 *   <li>{@link LifecyclePipeline} — raw → FE → ingest → materialize → online smoke
 *       → train export → optional DeepFM steps → quality report</li>
 * </ol>
 *
 * <pre>{@code
 * // ingest after FE
 * DataFrame eng = raw.feature().impute("mean", "age").standardScale("age").build();
 * FeatureIngest.Result ing = FeatureIngest.into(fp)
 *     .project("demo").view("user_feats").entities("user_id")
 *     .from(eng).run();
 * FeatureMaterializeJob.on(fp).fromIngest(ing).run();
 *
 * // full lifecycle
 * LifecyclePipeline.Result r = LifecyclePipeline.on(fp)
 *     .raw(raw).entities("user_id")
 *     .featureEngineering(fe -> fe.impute("mean", "age").standardScale("age", "score").build())
 *     .trainDeepFM(true).trainSteps(20)
 *     .run();
 * }</pre>
 *
 * @see org.bytedeco.pytorch.utils.feature.FeaturePlatform
 * @see org.bytedeco.pytorch.utils.feature.bridge.DataFrameBridge
 */
package org.bytedeco.pytorch.utils.feature.pipeline;
