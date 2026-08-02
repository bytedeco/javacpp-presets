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
package org.bytedeco.pytorch.utils.lake;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Partition selection for scans and writes.
 *
 * <p>Equality and IN predicates on partition columns; engines translate to
 * path prune (Iceberg/Hudi/Paimon/Parquet) or SQL predicates (Doris).</p>
 *
 * <pre>{@code
 * PartitionFilter f = PartitionFilter.of()
 *     .eq("dt", "2026-08-01")
 *     .in("region", "cn", "us")
 *     .build();
 * }</pre>
 */
public final class PartitionFilter {

    public enum Op { EQ, IN, GT, GTE, LT, LTE }

    public record Predicate(String column, Op op, List<String> values) {
        public Predicate {
            Objects.requireNonNull(column, "column");
            Objects.requireNonNull(op, "op");
            values = values == null ? List.of() : List.copyOf(values);
        }
    }

    private final List<Predicate> predicates;

    private PartitionFilter(List<Predicate> predicates) {
        this.predicates = List.copyOf(predicates);
    }

    public static Builder of() {
        return new Builder();
    }

    /** Single equality convenience. */
    public static PartitionFilter eq(String column, String value) {
        return of().eq(column, value).build();
    }

    public List<Predicate> predicates() {
        return predicates;
    }

    public boolean isEmpty() {
        return predicates.isEmpty();
    }

    /** Render as SQL AND-chain for engines that accept SQL (Doris / JDBC). */
    public String toSql() {
        if (predicates.isEmpty()) return "1=1";
        List<String> parts = new ArrayList<>(predicates.size());
        for (Predicate p : predicates) {
            String col = quoteIdent(p.column());
            switch (p.op()) {
                case EQ -> parts.add(col + " = " + quoteLit(p.values().get(0)));
                case IN -> {
                    StringBuilder sb = new StringBuilder(col).append(" IN (");
                    for (int i = 0; i < p.values().size(); i++) {
                        if (i > 0) sb.append(',');
                        sb.append(quoteLit(p.values().get(i)));
                    }
                    sb.append(')');
                    parts.add(sb.toString());
                }
                case GT -> parts.add(col + " > " + quoteLit(p.values().get(0)));
                case GTE -> parts.add(col + " >= " + quoteLit(p.values().get(0)));
                case LT -> parts.add(col + " < " + quoteLit(p.values().get(0)));
                case LTE -> parts.add(col + " <= " + quoteLit(p.values().get(0)));
            }
        }
        return String.join(" AND ", parts);
    }

    /** Flat map of EQ predicates only (path-style partition dirs). */
    public Map<String, String> equalityMap() {
        Map<String, String> m = new LinkedHashMap<>();
        for (Predicate p : predicates) {
            if (p.op() == Op.EQ && !p.values().isEmpty()) {
                m.put(p.column(), p.values().get(0));
            }
        }
        return Collections.unmodifiableMap(m);
    }

    private static String quoteIdent(String name) {
        return "`" + name.replace("`", "``") + "`";
    }

    private static String quoteLit(String v) {
        if (v == null) return "NULL";
        return "'" + v.replace("'", "''") + "'";
    }

    public static final class Builder {
        private final List<Predicate> preds = new ArrayList<>();

        public Builder eq(String column, String value) {
            preds.add(new Predicate(column, Op.EQ, List.of(value)));
            return this;
        }

        public Builder in(String column, String... values) {
            preds.add(new Predicate(column, Op.IN, List.of(values)));
            return this;
        }

        public Builder gt(String column, String value) {
            preds.add(new Predicate(column, Op.GT, List.of(value)));
            return this;
        }

        public Builder gte(String column, String value) {
            preds.add(new Predicate(column, Op.GTE, List.of(value)));
            return this;
        }

        public Builder lt(String column, String value) {
            preds.add(new Predicate(column, Op.LT, List.of(value)));
            return this;
        }

        public Builder lte(String column, String value) {
            preds.add(new Predicate(column, Op.LTE, List.of(value)));
            return this;
        }

        public Builder predicate(Predicate p) {
            preds.add(Objects.requireNonNull(p));
            return this;
        }

        public PartitionFilter build() {
            return new PartitionFilter(preds);
        }
    }

    @Override
    public String toString() {
        return "PartitionFilter" + predicates;
    }
}
