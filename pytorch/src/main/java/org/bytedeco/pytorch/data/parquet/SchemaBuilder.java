package org.bytedeco.pytorch.data.parquet;
import java.util.ArrayList;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;
import org.apache.parquet.schema.Type.Repetition;
import org.apache.parquet.schema.Types;

/**
 * Utility for building {@link MessageType} schemas programmatically.
 *
 * <p>Example — flat schema:
 * <pre>
 *   MessageType schema = SchemaBuilder.builder("root")
 *       .requiredInt64("id")
 *       .optionalString("name")
 *       .requiredDouble("score")
 *       .build();
 * </pre>
 *
 * <p>Example — nested schema:
 * <pre>
 *   MessageType schema = SchemaBuilder.builder("root")
 *       .requiredInt64("id")
 *       .startGroup("stats", false)
 *           .requiredInt32("count")
 *           .optionalDouble("score")
 *       .endGroup()
 *       .build();
 * </pre>
 */
public final class SchemaBuilder {
    private final ArrayList<Type> fields = new ArrayList<>();
    private final String rootName;

    private SchemaBuilder(String rootName) { this.rootName = rootName; }

    public static SchemaBuilder builder(String rootName) { return new SchemaBuilder(rootName); }
    public static SchemaBuilder builder() { return builder("root"); }

    // ---- factory helpers ----

    private SchemaBuilder add(Type field) {
        fields.add(field);
        return this;
    }

    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName INT32 =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT32;
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName INT64 =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT64;
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName FLOAT =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FLOAT;
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName DOUBLE =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.DOUBLE;
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName BOOLEAN =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BOOLEAN;
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName BINARY =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY;
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName FIXED =
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FIXED_LEN_BYTE_ARRAY;

    // ---- scalar primitives ----

    public SchemaBuilder requiredInt32(String name) {
        return add(Types.required(INT32).named(name));
    }

    public SchemaBuilder optionalInt32(String name) {
        return add(Types.optional(INT32).named(name));
    }

    public SchemaBuilder requiredInt64(String name) {
        return add(Types.required(INT64).named(name));
    }

    public SchemaBuilder optionalInt64(String name) {
        return add(Types.optional(INT64).named(name));
    }

    public SchemaBuilder requiredFloat(String name) {
        return add(Types.required(FLOAT).named(name));
    }

    public SchemaBuilder optionalFloat(String name) {
        return add(Types.optional(FLOAT).named(name));
    }

    public SchemaBuilder requiredDouble(String name) {
        return add(Types.required(DOUBLE).named(name));
    }

    public SchemaBuilder optionalDouble(String name) {
        return add(Types.optional(DOUBLE).named(name));
    }

    public SchemaBuilder requiredBoolean(String name) {
        return add(Types.required(BOOLEAN).named(name));
    }

    public SchemaBuilder optionalBoolean(String name) {
        return add(Types.optional(BOOLEAN).named(name));
    }

    public SchemaBuilder requiredBinary(String name) {
        return add(Types.required(BINARY).named(name));
    }

    public SchemaBuilder optionalBinary(String name) {
        return add(Types.optional(BINARY).named(name));
    }

    public SchemaBuilder requiredFixed(int length, String name) {
        return add(Types.required(FIXED).length(length).named(name));
    }

    public SchemaBuilder optionalFixed(int length, String name) {
        return add(Types.optional(FIXED).length(length).named(name));
    }

    // ---- logical-type wrappers ----

    public SchemaBuilder requiredString(String name) {
        return add(Types.required(BINARY).as(LogicalTypeAnnotation.stringType()).named(name));
    }

    public SchemaBuilder optionalString(String name) {
        return add(Types.optional(BINARY).as(LogicalTypeAnnotation.stringType()).named(name));
    }

    // ---- group nesting ----

    /**
     * Start a group field. Always call {@link GroupBuilder#endGroup()} to close it.
     * @param name  field name
     * @param required true for required, false for optional
     */
    public GroupBuilder startGroup(String name, boolean required) {
        return new GroupBuilder(this, name, required);
    }

    /** Build the final MessageType. */
    public MessageType build() {
        return new MessageType(rootName, fields.toArray(new Type[0]));
    }

    // ---- nested group builder ----

    public class GroupBuilder {
        private final ArrayList<Type> groupFields = new ArrayList<>();
        private final String name;
        private final boolean required;

        GroupBuilder(SchemaBuilder outer, String name, boolean required) {
            this.name = name;
            this.required = required;
        }

        public GroupBuilder requiredInt32(String n) {
            groupFields.add(Types.required(INT32).named(n)); return this;
        }

        public GroupBuilder optionalInt32(String n) {
            groupFields.add(Types.optional(INT32).named(n)); return this;
        }

        public GroupBuilder requiredInt64(String n) {
            groupFields.add(Types.required(INT64).named(n)); return this;
        }

        public GroupBuilder optionalInt64(String n) {
            groupFields.add(Types.optional(INT64).named(n)); return this;
        }

        public GroupBuilder requiredFloat(String n) {
            groupFields.add(Types.required(FLOAT).named(n)); return this;
        }

        public GroupBuilder optionalFloat(String n) {
            groupFields.add(Types.optional(FLOAT).named(n)); return this;
        }

        public GroupBuilder requiredDouble(String n) {
            groupFields.add(Types.required(DOUBLE).named(n)); return this;
        }

        public GroupBuilder optionalDouble(String n) {
            groupFields.add(Types.optional(DOUBLE).named(n)); return this;
        }

        public GroupBuilder requiredString(String n) {
            groupFields.add(Types.required(BINARY).as(LogicalTypeAnnotation.stringType()).named(n));
            return this;
        }

        public GroupBuilder optionalString(String n) {
            groupFields.add(Types.optional(BINARY).as(LogicalTypeAnnotation.stringType()).named(n));
            return this;
        }

        public GroupBuilder requiredBinary(String n) {
            groupFields.add(Types.required(BINARY).named(n)); return this;
        }

        public GroupBuilder optionalBinary(String n) {
            groupFields.add(Types.optional(BINARY).named(n)); return this;
        }

        /** Close this group and return the parent builder. */
        public SchemaBuilder endGroup() {
            Repetition rep = required ? Repetition.REQUIRED : Repetition.OPTIONAL;
            GroupType gt = new GroupType(rep, name, groupFields);
            SchemaBuilder.this.add(gt);
            return SchemaBuilder.this;
        }
    }
}
