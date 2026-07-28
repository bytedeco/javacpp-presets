package org.bytedeco.pytorch.dataframe.vectorstore;

import java.util.Map;

/**
 * Optional SPI for third-party / SDK-backed stores.
 *
 * <p>Register via {@code META-INF/services/org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreProvider}:
 * <pre>
 * com.example.OfficialMilvusVectorStoreProvider
 * </pre>
 *
 * <p>A provider whose {@link #name()} matches a built-in scheme
 * ({@code milvus}, {@code opensearch}, {@code mongo}, {@code pgvector}, {@code redis}, …)
 * <b>overrides</b> the pure-protocol adapter when {@link VectorStores#open(String, Map)}
 * is called — enabling seamless switch to an official SDK without changing call sites.
 *
 * <p>You may also register under a distinct name (e.g. {@code "milvus-sdk"}) and open
 * that scheme explicitly. Built-in pure-protocol adapters do <em>not</em> need a provider —
 * use {@link VectorStores#qdrant}, {@link VectorStores#redis}, …
 *
 * <p>Prefer also implementing the client-level SPI
 * ({@code MilvusBackend}, {@code OpenSearchBackend}, {@code MongoBackend},
 * {@code PgVectorBackend}) so {@code Milvus.connect} / {@code OpenSearch.connect} /
 * … resolve to the same official wrapper.
 */
public interface VectorStoreProvider {

    /** Unique scheme / name used in {@link VectorStores#open(String, Map)}. */
    String name();

    /**
     * Open a store from a free-form config map.
     * Common keys: {@code url}, {@code host}, {@code port}, {@code collection},
     * {@code dim}, {@code metric}, {@code apiKey}, {@code username}, {@code password}.
     */
    VectorStore open(Map<String, Object> config);
}
