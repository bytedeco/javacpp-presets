package org.bytedeco.pytorch.data.arrow;

import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.bytedeco.pytorch.data.dataframe.Column;

/**
 * Maps between DataFrame {@link Column.DType} and Apache Arrow field types.
 */
public final class ArrowSchemaMapper {
    private ArrowSchemaMapper() {}

    public static Field toField(String name, Column.DType dtype) {
        ArrowType type = switch (dtype) {
            case INT32 -> new ArrowType.Int(32, true);
            case INT64, DURATION -> new ArrowType.Int(64, true);
            case FLOAT32 -> new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
            case FLOAT64 -> new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
            case BOOLEAN -> new ArrowType.Bool();
            case STRING, TENSOR, VECTOR,
                 IMAGE, AUDIO, VIDEO, EMBEDDING, BINARY, JSON,
                 LIST, MAP, STRUCT, GRAPH, POINT_CLOUD -> new ArrowType.Utf8();
            case DATE -> new ArrowType.Date(DateUnit.DAY);
            case DATETIME -> new ArrowType.Timestamp(TimeUnit.MILLISECOND, null);
            case TIME -> new ArrowType.Time(TimeUnit.MILLISECOND, 32);
        };
        return new Field(name, FieldType.nullable(type), null);
    }

    public static Column.DType fromField(Field field) {
        ArrowType type = field.getType();
        if (type instanceof ArrowType.Int intType) {
            return intType.getBitWidth() <= 32 ? Column.DType.INT32 : Column.DType.INT64;
        }
        if (type instanceof ArrowType.FloatingPoint fp) {
            return fp.getPrecision() == FloatingPointPrecision.SINGLE
                ? Column.DType.FLOAT32 : Column.DType.FLOAT64;
        }
        if (type instanceof ArrowType.Bool) return Column.DType.BOOLEAN;
        if (type instanceof ArrowType.Utf8 || type instanceof ArrowType.LargeUtf8) {
            return Column.DType.STRING;
        }
        if (type instanceof ArrowType.Date) return Column.DType.DATE;
        if (type instanceof ArrowType.Timestamp) return Column.DType.DATETIME;
        if (type instanceof ArrowType.Time) return Column.DType.TIME;
        if (type instanceof ArrowType.Duration) return Column.DType.DURATION;
        if (type instanceof ArrowType.Decimal) return Column.DType.FLOAT64;
        return Column.DType.STRING;
    }
}
