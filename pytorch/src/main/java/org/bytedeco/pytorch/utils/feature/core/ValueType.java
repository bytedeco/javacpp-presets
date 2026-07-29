/*
 * Feature value types aligned with Feast ValueType / Featureform dtypes /
 * Databricks Feature Store column types, plus embedding for multimodal stores.
 *
 * Distinct from recommend.basic.features.* which describe model embedding inputs.
 */
package org.bytedeco.pytorch.utils.feature.core;

import java.util.Locale;

/**
 * Primitive and composite value types for feature columns.
 */
public enum ValueType {
    UNKNOWN,
    BOOL,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STRING,
    BYTES,
    BOOL_LIST,
    INT32_LIST,
    INT64_LIST,
    FLOAT32_LIST,
    FLOAT64_LIST,
    STRING_LIST,
    /** Fixed-dim float vector (image/text/audio embeddings, item towers). */
    EMBEDDING,
    UNIX_TIMESTAMP;

    public boolean isList() {
        switch (this) {
            case BOOL_LIST:
            case INT32_LIST:
            case INT64_LIST:
            case FLOAT32_LIST:
            case FLOAT64_LIST:
            case STRING_LIST:
                return true;
            default:
                return false;
        }
    }

    public boolean isNumeric() {
        switch (this) {
            case INT32:
            case INT64:
            case FLOAT32:
            case FLOAT64:
            case UNIX_TIMESTAMP:
                return true;
            default:
                return false;
        }
    }

    public boolean isIntegral() {
        return this == INT32 || this == INT64 || this == UNIX_TIMESTAMP;
    }

    public boolean isFloating() {
        return this == FLOAT32 || this == FLOAT64 || this == EMBEDDING;
    }

    /** Element type for list/embedding; self for scalars. */
    public ValueType elementType() {
        switch (this) {
            case BOOL_LIST:
                return BOOL;
            case INT32_LIST:
                return INT32;
            case INT64_LIST:
                return INT64;
            case FLOAT32_LIST:
            case EMBEDDING:
                return FLOAT32;
            case FLOAT64_LIST:
                return FLOAT64;
            case STRING_LIST:
                return STRING;
            default:
                return this;
        }
    }

    public static ValueType parse(String raw) {
        if (raw == null || raw.isEmpty()) return UNKNOWN;
        String s = raw.trim().toUpperCase(Locale.ROOT).replace('-', '_');
        switch (s) {
            case "BOOL":
            case "BOOLEAN":
                return BOOL;
            case "INT":
            case "INTEGER":
            case "INT32":
                return INT32;
            case "LONG":
            case "INT64":
                return INT64;
            case "FLOAT":
            case "FLOAT32":
                return FLOAT32;
            case "DOUBLE":
            case "FLOAT64":
                return FLOAT64;
            case "STR":
            case "STRING":
            case "UTF8":
                return STRING;
            case "BYTES":
            case "BINARY":
                return BYTES;
            case "EMBEDDING":
            case "VECTOR":
            case "FLOAT_VECTOR":
                return EMBEDDING;
            case "TIMESTAMP":
            case "UNIX_TIMESTAMP":
            case "EVENT_TS":
                return UNIX_TIMESTAMP;
            default:
                try {
                    return ValueType.valueOf(s);
                } catch (IllegalArgumentException e) {
                    return UNKNOWN;
                }
        }
    }
}
