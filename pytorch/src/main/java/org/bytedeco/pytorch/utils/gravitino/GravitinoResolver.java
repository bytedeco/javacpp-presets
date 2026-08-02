/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.gravitino;

import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeOptions;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Resolve Gravitino table metadata ({@code provider} / {@code format} / {@code location})
 * into concrete {@link LakeOptions} for Doris / Iceberg / Paimon / Hudi / Daft / Parquet.
 *
 * <p>Value: feature / recsys only need the federated name
 * {@code metalake.catalog.schema.table}; backend is swappable.</p>
 */
public final class GravitinoResolver {

    /**
     * Resolved backend handle.
     *
     * @param format   concrete lake format
     * @param options  options for {@link org.bytedeco.pytorch.utils.lake.LakeFactory}
     * @param provider raw provider string from Gravitino
     * @param location table location / warehouse / JDBC URI
     */
    public record Resolved(LakeFormat format, LakeOptions options, String provider, String location) {}

    private GravitinoResolver() {}

    /**
     * Map provider + properties to a backend format and {@link LakeOptions}.
     *
     * @param fullName metalake.catalog.schema.table (informational)
     * @param provider e.g. {@code lakehouse-iceberg}, {@code jdbc-doris}, {@code hive}, {@code fileset}
     * @param location table location or JDBC URI
     * @param properties table / catalog properties from Gravitino
     * @param base       Gravitino client options (batch, columns, filters carry through)
     */
    public static Resolved resolve(String fullName, String provider, String location,
                                   Map<String, String> properties, GravitinoOptions base) {
        Objects.requireNonNull(base, "base");
        Map<String, String> props = properties == null ? Map.of() : properties;
        String p = provider == null ? props.getOrDefault("provider", "") : provider;
        String loc = location != null ? location
                : firstNonBlank(props.get("location"), props.get("warehouse"),
                props.get("uri"), props.get("jdbc-url"), props.get("jdbcUrl"));
        String formatHint = firstNonBlank(props.get("format"), props.get("table-format"),
                props.get("lake.format"), props.get("provider"));

        LakeFormat fmt = detectFormat(p, formatHint, loc, props);
        LakeOptions.Builder b = LakeOptions.builder(fmt)
                .batchRows(base.batchRows())
                .parallelism(base.parallelism())
                .partitionFilter(base.partitionFilter())
                .replicaPolicy(base.replicaPolicy())
                .columns(base.columns())
                .idleStop(base.idleStop())
                .connectTimeoutMs(base.connectTimeoutMs())
                .socketTimeoutMs(base.socketTimeoutMs())
                .properties(props);

        if (base.username() != null) b.username(base.username());
        if (base.password() != null) b.password(base.password());

        // Parse namespace/table from fullName when backend needs them
        String[] parts = fullName == null ? new String[0] : fullName.split("\\.");
        String schema = base.schemaName();
        String table = base.table();
        if (parts.length >= 4) {
            schema = parts[2];
            table = parts[3];
        } else if (parts.length == 3 && table == null) {
            table = parts[2];
        }
        if (schema != null) b.namespaceName(schema);
        if (table != null) b.table(table);

        switch (fmt) {
            case DORIS -> {
                String jdbc = firstNonBlank(loc, props.get("jdbc-url"), props.get("jdbcUrl"),
                        props.get("uri"));
                if (jdbc != null && !jdbc.startsWith("jdbc:")) {
                    // host:port/db → mysql jdbc
                    jdbc = "jdbc:mysql://" + jdbc;
                }
                b.uri(jdbc);
            }
            case ICEBERG, PAIMON, HUDI, PARQUET, DAFT -> {
                String wh = firstNonBlank(loc, props.get("warehouse"), props.get("location"));
                b.uri(wh).warehouse(wh);
            }
            case GRAVITINO -> b.uri(base.uri());
        }

        // Carry auth token if present
        if (base.authToken() != null) b.property("auth_token", base.authToken());
        b.property("gravitino.full_name", fullName == null ? "" : fullName);
        b.property("gravitino.provider", p == null ? "" : p);

        return new Resolved(fmt, b.build(), p, loc);
    }

    static LakeFormat detectFormat(String provider, String formatHint, String location,
                                   Map<String, String> props) {
        String blob = ((provider == null ? "" : provider) + " "
                + (formatHint == null ? "" : formatHint) + " "
                + props.getOrDefault("catalog-backend", "") + " "
                + props.getOrDefault("type", "")).toLowerCase(Locale.ROOT);

        if (blob.contains("doris")) return LakeFormat.DORIS;
        if (blob.contains("iceberg") || blob.contains("lakehouse-iceberg")) return LakeFormat.ICEBERG;
        if (blob.contains("paimon")) return LakeFormat.PAIMON;
        if (blob.contains("hudi")) return LakeFormat.HUDI;
        if (blob.contains("daft")) return LakeFormat.DAFT;
        if (blob.contains("jdbc") && (blob.contains("mysql") || location != null
                && location.toLowerCase(Locale.ROOT).contains("mysql"))) {
            // generic mysql often fronts Doris FE
            if (blob.contains("doris") || props.containsKey("doris.fe.http")) return LakeFormat.DORIS;
        }
        if (location != null) {
            String loc = location.toLowerCase(Locale.ROOT);
            if (loc.startsWith("jdbc:mysql") || loc.contains(":9030")) return LakeFormat.DORIS;
            if (loc.contains("iceberg") || loc.contains("/warehouse")) {
                // default open-table warehouse → Iceberg unless props say otherwise
                if (blob.contains("paimon")) return LakeFormat.PAIMON;
                if (blob.contains("hudi")) return LakeFormat.HUDI;
                return LakeFormat.ICEBERG;
            }
            if (loc.endsWith(".parquet") || loc.contains("/parquet")) return LakeFormat.PARQUET;
        }
        // fileset / hive / unknown → parquet path if location set, else iceberg warehouse
        if (location != null && !location.isBlank()) return LakeFormat.PARQUET;
        return LakeFormat.ICEBERG;
    }

    private static String firstNonBlank(String... vals) {
        if (vals == null) return null;
        for (String v : vals) {
            if (v != null && !v.isBlank()) return v;
        }
        return null;
    }

    /**
     * Build a minimal in-memory / file mock table entry for offline tests.
     */
    public static Map<String, Object> mockTableEntry(String fullName, String provider,
                                                     String location, LakeFormat format) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", fullName);
        m.put("provider", provider);
        m.put("location", location);
        m.put("format", format == null ? null : format.name());
        Map<String, String> props = new LinkedHashMap<>();
        props.put("provider", provider == null ? "" : provider);
        props.put("location", location == null ? "" : location);
        if (format != null) props.put("format", format.name().toLowerCase(Locale.ROOT));
        m.put("properties", props);
        return m;
    }
}
