package org.bytedeco.pytorch.data.dataframe.pgvector;

import java.util.Map;

/**
 * Optional SPI for third-party / official-SDK-backed pgvector clients.
 *
 * <p>Built-in {@link PgVector} uses plain JDBC ({@code java.sql}) and expects
 * {@code org.postgresql:postgresql} on the <em>application</em> classpath.
 * Register a backend to wrap a pooled DataSource, jOOQ, or a custom type mapper
 * while keeping the same {@link PgVector} surface.
 */
public interface PgVectorBackend {

    String name();

    default String[] aliases() {
        return new String[0];
    }

    /**
     * Open a client from free-form config.
     * Common keys: {@code url}/{@code jdbcUrl}, {@code user}/{@code username},
     * {@code password}, {@code table}, {@code dim}, {@code metric}.
     */
    PgVector open(Map<String, Object> config);
}
