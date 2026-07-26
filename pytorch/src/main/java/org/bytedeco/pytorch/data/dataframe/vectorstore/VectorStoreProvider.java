package org.bytedeco.pytorch.data.dataframe.vectorstore;

import java.util.Map;

/**
 * Optional SPI for third-party / SDK-backed stores.
 *
 * <p>Register via {@code META-INF/services/org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStoreProvider}:
 * <pre>
 * com.example.JedisVectorStoreProvider
 * </pre>
 *
 * <p>Then open with {@code VectorStores.open("redis-jedis", config)}.
 * Built-in pure-protocol adapters do <em>not</em> need a provider —
 * use {@link VectorStores#qdrant}, {@link VectorStores#redis}, …
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
