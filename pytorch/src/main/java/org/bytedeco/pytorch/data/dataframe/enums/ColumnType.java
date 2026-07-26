package org.bytedeco.pytorch.data.dataframe.enums;

import org.bytedeco.pytorch.data.dataframe.Column;

/**
 * Multimodal / schema column type tags used by dtype wrappers.
 * Bridge to {@link Column.DType} via {@link #toDType()}.
 */
public enum ColumnType {
    INT32, INT64, FLOAT32, FLOAT64, STRING, BOOLEAN, VECTOR,
    IMAGE, AUDIO, VIDEO, TEXT_TOKEN, TENSOR, BINARY,
    STRUCT, JSON, LIST_VIEW, DATE, MAP_VIEW, TIME, TIMESTAMP, LOG_RECORD, EMBEDDING, GRAPH_VIEW,
    POINT_CLOUD;

    public boolean isNumericType() {
        return this == INT32 || this == INT64 || this == FLOAT32 || this == FLOAT64;
    }

    /** Map to DataFrame {@link Column.DType} (best-effort). */
    public Column.DType toDType() {
        return switch (this) {
            case INT32 -> Column.DType.INT32;
            case INT64 -> Column.DType.INT64;
            case FLOAT32 -> Column.DType.FLOAT32;
            case FLOAT64 -> Column.DType.FLOAT64;
            case STRING, TEXT_TOKEN -> Column.DType.STRING;
            case BOOLEAN -> Column.DType.BOOLEAN;
            case VECTOR -> Column.DType.VECTOR;
            case TENSOR -> Column.DType.TENSOR;
            case DATE -> Column.DType.DATE;
            case TIME -> Column.DType.TIME;
            case TIMESTAMP -> Column.DType.DATETIME;
            case IMAGE -> Column.DType.IMAGE;
            case AUDIO -> Column.DType.AUDIO;
            case VIDEO -> Column.DType.VIDEO;
            case EMBEDDING -> Column.DType.EMBEDDING;
            case BINARY -> Column.DType.BINARY;
            case JSON -> Column.DType.JSON;
            case LIST_VIEW -> Column.DType.LIST;
            case MAP_VIEW -> Column.DType.MAP;
            case STRUCT, LOG_RECORD -> Column.DType.STRUCT;
            case GRAPH_VIEW -> Column.DType.GRAPH;
            case POINT_CLOUD -> Column.DType.POINT_CLOUD;
        };
    }

    public static ColumnType fromDType(Column.DType d) {
        if (d == null) return STRING;
        return switch (d) {
            case INT32 -> INT32;
            case INT64 -> INT64;
            case FLOAT32 -> FLOAT32;
            case FLOAT64 -> FLOAT64;
            case BOOLEAN -> BOOLEAN;
            case STRING -> STRING;
            case TENSOR -> TENSOR;
            case DATE -> DATE;
            case DATETIME -> TIMESTAMP;
            case TIME -> TIME;
            case VECTOR -> VECTOR;
            case IMAGE -> IMAGE;
            case AUDIO -> AUDIO;
            case VIDEO -> VIDEO;
            case EMBEDDING -> EMBEDDING;
            case BINARY -> BINARY;
            case JSON -> JSON;
            case LIST -> LIST_VIEW;
            case MAP -> MAP_VIEW;
            case STRUCT -> STRUCT;
            case GRAPH -> GRAPH_VIEW;
            case POINT_CLOUD -> POINT_CLOUD;
            default -> STRING;
        };
    }
}
