/**
 * Zero-SDK vector database adapters for DataFrame / ANN pipelines.
 *
 * <p>Design goals:
 * <ul>
 *   <li><b>No hard vendor dependencies</b> — Redis / Qdrant / Milvus / OpenSearch /
 *       MongoDB Atlas talk over HTTP or a tiny hand-rolled RESP client; pgvector
 *       uses only {@code java.sql} (driver on the <em>application</em> classpath).</li>
 *   <li><b>Uniform SPI</b> — {@link org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore}
 *       for upsert / delete / knn search / drop; factory via
 *       {@link org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores}.</li>
 *   <li><b>Full clients (Redis-style)</b> — also see
 *       {@link org.bytedeco.pytorch.data.dataframe.milvus.Milvus},
 *       {@link org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch},
 *       {@link org.bytedeco.pytorch.data.dataframe.mongo.Mongo},
 *       {@link org.bytedeco.pytorch.data.dataframe.pgvector.PgVector}
 *       for official-package-shaped APIs + DataFrame to/read.</li>
 *   <li><b>Optional plugins</b> — implement
 *       {@link org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreProvider}
 *       (and/or client-level {@code *Backend} SPIs) and register via
 *       {@code META-INF/services}. A provider whose name matches a built-in scheme
 *       <b>overrides</b> the pure-protocol adapter — seamless official-SDK switch
 *       without changing call sites.</li>
 *   <li><b>In-process default</b> —
 *       {@link org.bytedeco.pytorch.data.dataframe.vectorstore.memory.InMemoryVectorStore}
 *       wraps the pure-Java {@code HnswIndex}.</li>
 * </ul>
 *
 * <pre>{@code
 * try (VectorStore vs = VectorStores.qdrant("http://localhost:6333", "clips", 768, VectorMetric.COSINE)) {
 *     vs.ensureCollection();
 *     vs.upsert(records);
 *     VectorSearchResult top = vs.search(VectorQuery.of(query, 10));
 * }
 * }</pre>
 */
package org.bytedeco.pytorch.data.dataframe.vectorstore;
