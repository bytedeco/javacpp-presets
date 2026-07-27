package org.bytedeco.pytorch.data.dataframe.opensearch;

import java.util.Map;

/**
 * Optional SPI for third-party / official-SDK-backed OpenSearch clients.
 *
 * <p>Register via {@code META-INF/services/org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchBackend}
 * or {@link OpenSearch#registerBackend(OpenSearchBackend)}. A provider whose
 * {@link #name()} is {@code "opensearch"} (or an alias) <b>overrides</b> the
 * built-in REST client — e.g. wrap {@code org.opensearch.client.opensearch.OpenSearchClient}.
 */
public interface OpenSearchBackend {

    String name();

    default String[] aliases() {
        return new String[0];
    }

    /**
     * Open a client from free-form config.
     * Common keys: {@code url}, {@code username}/{@code user}, {@code password},
     * {@code apiKey}, {@code timeout}, {@code index}.
     */
    OpenSearch open(Map<String, Object> config);
}
