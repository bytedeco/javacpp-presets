package org.bytedeco.pytorch.plot.tensorboard;
import org.bytedeco.pytorch.jit.*;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;

/**
 * Minimal protobuf3 wire encoder (no generated stubs).
 * Only the field types needed by TensorBoard event / plugin protos.
 */
public final class ProtoWire {
    public static final int WIRE_VARINT = 0;
    public static final int WIRE_64 = 1;
    public static final int WIRE_LEN = 2;
    public static final int WIRE_32 = 5;

    // tensorflow.DataType
    public static final int DT_FLOAT = 1;
    public static final int DT_DOUBLE = 2;
    public static final int DT_INT32 = 3;
    public static final int DT_UINT8 = 4;
    public static final int DT_INT16 = 5;
    public static final int DT_INT8 = 6;
    public static final int DT_STRING = 7;
    public static final int DT_INT64 = 9;
    public static final int DT_BOOL = 10;

    private ProtoWire() {}

    public static ByteArrayOutputStream buf() {
        return new ByteArrayOutputStream(64);
    }

    public static void tag(ByteArrayOutputStream out, int field, int wire) {
        varint(out, ((long) field << 3) | wire);
    }

    public static void varint(ByteArrayOutputStream out, long v) {
        // treat as unsigned
        while ((v & ~0x7fL) != 0) {
            out.write((int) ((v & 0x7f) | 0x80));
            v >>>= 7;
        }
        out.write((int) v);
    }

    public static void svarint(ByteArrayOutputStream out, long v) {
        // zigzag not needed for our positive enums / sizes
        varint(out, v);
    }

    public static void bytes(ByteArrayOutputStream out, int field, byte[] raw) {
        if (raw == null) raw = new byte[0];
        tag(out, field, WIRE_LEN);
        varint(out, raw.length);
        out.write(raw, 0, raw.length);
    }

    public static void string(ByteArrayOutputStream out, int field, String s) {
        if (s == null) s = "";
        bytes(out, field, s.getBytes(StandardCharsets.UTF_8));
    }

    public static void message(ByteArrayOutputStream out, int field, byte[] msg) {
        bytes(out, field, msg);
    }

    public static void float32(ByteArrayOutputStream out, int field, float v) {
        tag(out, field, WIRE_32);
        int bits = Float.floatToIntBits(v);
        out.write(bits & 0xff);
        out.write((bits >>> 8) & 0xff);
        out.write((bits >>> 16) & 0xff);
        out.write((bits >>> 24) & 0xff);
    }

    public static void double64(ByteArrayOutputStream out, int field, double v) {
        tag(out, field, WIRE_64);
        long bits = Double.doubleToLongBits(v);
        for (int i = 0; i < 8; i++) {
            out.write((int) (bits & 0xff));
            bits >>>= 8;
        }
    }

    public static void int32(ByteArrayOutputStream out, int field, int v) {
        tag(out, field, WIRE_VARINT);
        varint(out, v & 0xffffffffL);
    }

    public static void int64(ByteArrayOutputStream out, int field, long v) {
        tag(out, field, WIRE_VARINT);
        varint(out, v);
    }

    public static void bool(ByteArrayOutputStream out, int field, boolean v) {
        tag(out, field, WIRE_VARINT);
        out.write(v ? 1 : 0);
    }

    public static void enum_(ByteArrayOutputStream out, int field, int number) {
        int64(out, field, number);
    }

    /** Packed repeated float (length-delimited block of little-endian f32). */
    public static void packedFloats(ByteArrayOutputStream out, int field, float[] vals) {
        if (vals == null || vals.length == 0) return;
        byte[] raw = new byte[vals.length * 4];
        for (int i = 0; i < vals.length; i++) {
            int bits = Float.floatToIntBits(vals[i]);
            int o = i * 4;
            raw[o] = (byte) (bits);
            raw[o + 1] = (byte) (bits >>> 8);
            raw[o + 2] = (byte) (bits >>> 16);
            raw[o + 3] = (byte) (bits >>> 24);
        }
        bytes(out, field, raw);
    }

    /** Repeated non-packed float (one field per value) — TensorProto.float_val uses this. */
    public static void repeatedFloat(ByteArrayOutputStream out, int field, float[] vals) {
        if (vals == null) return;
        for (float v : vals) float32(out, field, v);
    }

    public static void repeatedFloat(ByteArrayOutputStream out, int field, Collection<? extends Number> vals) {
        if (vals == null) return;
        for (Number n : vals) float32(out, field, n.floatValue());
    }

    public static void repeatedDouble(ByteArrayOutputStream out, int field, double[] vals) {
        if (vals == null) return;
        for (double v : vals) double64(out, field, v);
    }

    public static void repeatedInt64(ByteArrayOutputStream out, int field, long[] vals) {
        if (vals == null) return;
        for (long v : vals) int64(out, field, v);
    }

    public static void repeatedInt32(ByteArrayOutputStream out, int field, int[] vals) {
        if (vals == null) return;
        for (int v : vals) int32(out, field, v);
    }

    public static void repeatedString(ByteArrayOutputStream out, int field, String[] vals) {
        if (vals == null) return;
        for (String s : vals) string(out, field, s);
    }

    public static void repeatedBytes(ByteArrayOutputStream out, int field, byte[][] vals) {
        if (vals == null) return;
        for (byte[] b : vals) bytes(out, field, b);
    }

    public static void repeatedMessage(ByteArrayOutputStream out, int field, Collection<byte[]> msgs) {
        if (msgs == null) return;
        for (byte[] m : msgs) message(out, field, m);
    }

    // ---- TensorBoard / TF common messages ---------------------------------

    /** TensorShapeProto with given dim sizes. */
    public static byte[] tensorShape(long... dims) {
        ByteArrayOutputStream shape = buf();
        if (dims != null) {
            for (long d : dims) {
                ByteArrayOutputStream dim = buf();
                int64(dim, 1, d); // Dim.size
                message(shape, 2, dim.toByteArray()); // TensorShapeProto.dim
            }
        }
        return shape.toByteArray();
    }

    /**
     * TensorProto field numbers (tensorflow.TensorProto):
     * <pre>
     *   1 dtype, 2 tensor_shape, 3 version_number, 4 tensor_content,
     *   5 float_val (packed), 6 double_val (packed), 7 int_val (packed),
     *   8 string_val, 10 int64_val (packed), 11 bool_val (packed)
     * </pre>
     */
    public static byte[] tensorProtoFloat(long[] shape, float[] floatVal) {
        ByteArrayOutputStream t = buf();
        int64(t, 1, DT_FLOAT);
        message(t, 2, tensorShape(shape));
        // float_val is packed repeated float → length-delimited block
        if (floatVal != null && floatVal.length > 0) {
            packedFloats(t, 5, floatVal);
        }
        return t.toByteArray();
    }

    public static byte[] tensorProtoDouble(long[] shape, double[] doubleVal) {
        ByteArrayOutputStream t = buf();
        int64(t, 1, DT_DOUBLE);
        message(t, 2, tensorShape(shape));
        if (doubleVal != null && doubleVal.length > 0) {
            // packed repeated double
            byte[] raw = new byte[doubleVal.length * 8];
            for (int i = 0; i < doubleVal.length; i++) {
                long bits = Double.doubleToLongBits(doubleVal[i]);
                int o = i * 8;
                for (int b = 0; b < 8; b++) {
                    raw[o + b] = (byte) (bits & 0xff);
                    bits >>>= 8;
                }
            }
            bytes(t, 6, raw);
        }
        return t.toByteArray();
    }

    public static byte[] tensorProtoString(long[] shape, String... stringVal) {
        ByteArrayOutputStream t = buf();
        int64(t, 1, DT_STRING);
        message(t, 2, tensorShape(shape));
        if (stringVal != null) {
            for (String s : stringVal) {
                // string_val = field 8
                bytes(t, 8, s == null ? new byte[0] : s.getBytes(StandardCharsets.UTF_8));
            }
        }
        return t.toByteArray();
    }

    public static byte[] tensorProtoStringBytes(long[] shape, byte[]... stringVal) {
        ByteArrayOutputStream t = buf();
        int64(t, 1, DT_STRING);
        message(t, 2, tensorShape(shape));
        if (stringVal != null) {
            for (byte[] s : stringVal) bytes(t, 8, s == null ? new byte[0] : s);
        }
        return t.toByteArray();
    }

    public static byte[] tensorProtoInt32(long[] shape, int[] intVal) {
        ByteArrayOutputStream t = buf();
        int64(t, 1, DT_INT32);
        message(t, 2, tensorShape(shape));
        // int_val is packed repeated int32 (as varints)
        if (intVal != null && intVal.length > 0) {
            ByteArrayOutputStream packed = buf();
            for (int v : intVal) varint(packed, v & 0xffffffffL);
            bytes(t, 7, packed.toByteArray());
        }
        return t.toByteArray();
    }

    public static byte[] tensorProtoContent(int dtype, long[] shape, byte[] content) {
        ByteArrayOutputStream t = buf();
        int64(t, 1, dtype);
        message(t, 2, tensorShape(shape));
        bytes(t, 4, content);
        return t.toByteArray();
    }

    /**
     * SummaryMetadata { plugin_data { plugin_name, content } [, data_class] }
     * PluginData: 1 plugin_name, 2 content
     * SummaryMetadata: 1 plugin_data, 2 display_name, 3 summary_description, 4 data_class
     */
    public static byte[] pluginMetadata(String pluginName) {
        return pluginMetadata(pluginName, null, 0);
    }

    public static byte[] pluginMetadata(String pluginName, byte[] content) {
        return pluginMetadata(pluginName, content, 0);
    }

    public static byte[] pluginMetadata(String pluginName, byte[] content, int dataClass) {
        ByteArrayOutputStream plugin = buf();
        string(plugin, 1, pluginName);
        if (content != null && content.length > 0) bytes(plugin, 2, content);
        ByteArrayOutputStream meta = buf();
        message(meta, 1, plugin.toByteArray());
        if (dataClass != 0) enum_(meta, 4, dataClass);
        return meta.toByteArray();
    }

    /** google.protobuf.Value with number_value (field 2 double). */
    public static byte[] structNumber(double v) {
        ByteArrayOutputStream b = buf();
        double64(b, 2, v);
        return b.toByteArray();
    }

    /** google.protobuf.Value with string_value (field 3). */
    public static byte[] structString(String v) {
        ByteArrayOutputStream b = buf();
        string(b, 3, v);
        return b.toByteArray();
    }

    /** google.protobuf.Value with bool_value (field 4). */
    public static byte[] structBool(boolean v) {
        ByteArrayOutputStream b = buf();
        bool(b, 4, v);
        return b.toByteArray();
    }

    /** google.protobuf.ListValue { repeated Value values = 1 }. */
    public static byte[] structList(byte[]... values) {
        ByteArrayOutputStream b = buf();
        for (byte[] v : values) message(b, 1, v);
        return b.toByteArray();
    }
}
