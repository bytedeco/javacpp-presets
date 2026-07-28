package org.bytedeco.pytorch.dataframe.mongo;

import java.util.Map;

/**
 * Optional SPI for third-party / official-SDK-backed Mongo clients.
 *
 * <p>Built-in {@link Mongo} speaks the <b>Atlas Data API</b> (HTTPS).
 * To use self-hosted MongoDB or {@code mongodb-driver-sync}, implement this SPI
 * and register it so {@link Mongo#connect} / {@link Mongo#open} resolve to your
 * wrapper under scheme {@code "mongo"} / {@code "mongodb"} / {@code "atlas"}.
 *
 * <pre>{@code
 * // META-INF/services/...MongoBackend
 * public final class SyncDriverMongoBackend implements MongoBackend {
 *     public String name() { return "mongo"; }
 *     public String[] aliases() { return new String[]{"mongodb", "atlas"}; }
 *     public Mongo open(Map&lt;String, Object&gt; config) {
 *         return SyncDriverMongo.wrap(MongoClients.create(...), config);
 *     }
 * }
 * }</pre>
 */
public interface MongoBackend {

    String name();

    default String[] aliases() {
        return new String[0];
    }

    /**
     * Open a client from free-form config.
     * Common keys: {@code url}, {@code apiKey}, {@code dataSource},
     * {@code database}/{@code db}, {@code collection}, {@code timeout}.
     */
    Mongo open(Map<String, Object> config);
}
