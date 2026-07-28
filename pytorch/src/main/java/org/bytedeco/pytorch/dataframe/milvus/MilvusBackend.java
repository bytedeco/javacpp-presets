package org.bytedeco.pytorch.dataframe.milvus;

import java.util.Map;

/**
 * Optional SPI for third-party / official-SDK-backed Milvus clients.
 *
 * <p>Register via {@code META-INF/services/org.bytedeco.pytorch.dataframe.milvus.MilvusBackend}
 * or {@link Milvus#registerBackend(MilvusBackend)}. A provider whose
 * {@link #name()} is {@code "milvus"} (or any alias returned by {@link #aliases()})
 * <b>overrides</b> the built-in REST client when {@link Milvus#connect} /
 * {@link Milvus#open} is called — enabling seamless switch to
 * {@code milvus-sdk-java} without changing call sites.
 *
 * <pre>{@code
 * // in a third-party jar that depends on milvus-sdk-java:
 * public final class OfficialMilvusBackend implements MilvusBackend {
 *     public String name() { return "milvus"; }
 *     public Milvus open(Map&lt;String, Object&gt; config) {
 *         // wrap official client behind the same Milvus surface
 *         return OfficialMilvus.wrap(...);
 *     }
 * }
 * }</pre>
 */
public interface MilvusBackend {

    /** Primary scheme name (e.g. {@code "milvus"}). */
    String name();

    /** Extra scheme aliases that should resolve to this backend. */
    default String[] aliases() {
        return new String[0];
    }

    /**
     * Open a client from free-form config.
     * Common keys: {@code url}, {@code token}, {@code apiKey}, {@code dbName},
     * {@code timeout}, {@code collection}.
     */
    Milvus open(Map<String, Object> config);
}
