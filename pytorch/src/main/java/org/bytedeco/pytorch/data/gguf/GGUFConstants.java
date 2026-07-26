package org.bytedeco.pytorch.data.gguf;

/**
 * GGUF (GPT-Generated Unified Format) constants and metadata keys.
 *
 * <p>File layout (v3+):
 * <pre>
 *   magic u32 ("GGUF") | version u32 | n_tensors u64 | n_kv u64
 *   | metadata KVs | tensor infos | pad-to-alignment | tensor payloads
 * </pre>
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF spec</a>
 */
public final class GGUFConstants {
    private GGUFConstants() {}

    /** Little-endian "GGUF". */
    public static final int GGUF_MAGIC = 0x46554747;

    public static final int GGUF_VERSION_2 = 2;
    public static final int GGUF_VERSION_3 = 3;
    public static final int GGUF_VERSION_4 = 4;
    public static final int GGUF_VERSION_5 = 5;
    public static final int GGUF_VERSION_LATEST = GGUF_VERSION_3;

    /** Default tensor-data alignment in bytes. */
    public static final int ALIGNMENT = 32;

    // ---- GGML type ids (subset used for float/int tensors) -------------------
    public static final int GGML_TYPE_F32  = 0;
    public static final int GGML_TYPE_F16  = 1;
    public static final int GGML_TYPE_Q4_0 = 2;
    public static final int GGML_TYPE_Q4_1 = 3;
    public static final int GGML_TYPE_Q5_0 = 6;
    public static final int GGML_TYPE_Q5_1 = 7;
    public static final int GGML_TYPE_Q8_0 = 8;
    public static final int GGML_TYPE_Q8_1 = 9;
    public static final int GGML_TYPE_I8   = 24;
    public static final int GGML_TYPE_I16  = 25;
    public static final int GGML_TYPE_I32  = 26;
    public static final int GGML_TYPE_I64  = 27;
    public static final int GGML_TYPE_F64  = 28;
    public static final int GGML_TYPE_BF16 = 30;

    // ---- GGUF value types ----------------------------------------------------
    public static final int VALUE_UINT8   = 0;
    public static final int VALUE_INT8    = 1;
    public static final int VALUE_UINT16  = 2;
    public static final int VALUE_INT16   = 3;
    public static final int VALUE_UINT32  = 4;
    public static final int VALUE_INT32   = 5;
    public static final int VALUE_FLOAT32 = 6;
    public static final int VALUE_BOOL    = 7;
    public static final int VALUE_STRING  = 8;
    public static final int VALUE_ARRAY   = 9;
    public static final int VALUE_UINT64  = 10;
    public static final int VALUE_INT64   = 11;
    public static final int VALUE_FLOAT64 = 12;

    public static boolean isSupportedVersion(int version) {
        return version >= GGUF_VERSION_2 && version <= GGUF_VERSION_5;
    }

    public static int[] getSupportedVersions() {
        return new int[]{GGUF_VERSION_2, GGUF_VERSION_3, GGUF_VERSION_4, GGUF_VERSION_5};
    }

    /** Bytes-per-element for non-quantized GGML types; -1 if quantized/unknown. */
    public static int bytesPerElement(int ggmlType) {
        switch (ggmlType) {
            case GGML_TYPE_F32:
            case GGML_TYPE_I32: return 4;
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_I16: return 2;
            case GGML_TYPE_F64:
            case GGML_TYPE_I64: return 8;
            case GGML_TYPE_I8:  return 1;
            default: return -1;
        }
    }

    /**
     * Byte size of a tensor payload for the given GGML type and element count.
     * Quantized block sizes follow the ggml reference layout.
     */
    public static long nbytes(int ggmlType, long nElements) {
        int bpe = bytesPerElement(ggmlType);
        if (bpe > 0) return nElements * bpe;
        // Quantized block layouts (elements per block / bytes per block)
        switch (ggmlType) {
            case GGML_TYPE_Q4_0: return (nElements / 32L) * 18L;
            case GGML_TYPE_Q4_1: return (nElements / 32L) * 20L;
            case GGML_TYPE_Q5_0: return (nElements / 32L) * 22L;
            case GGML_TYPE_Q5_1: return (nElements / 32L) * 24L;
            case GGML_TYPE_Q8_0: return (nElements / 32L) * 34L;
            case GGML_TYPE_Q8_1: return (nElements / 32L) * 36L;
            default:
                // best-effort: treat as raw bytes of unknown layout
                return Math.max(0L, nElements);
        }
    }

    /** Common GGUF metadata keys. */
    public static final class MetadataKeys {
        public static final String GENERAL_ALIGNMENT = "general.alignment";
        public static final String MODEL_NAME = "general.name";
        public static final String MODEL_ARCHITECTURE = "general.architecture";
        public static final String MODEL_FILE_TYPE = "general.file_type";
        public static final String CONTEXT_LENGTH = "llama.context_length";
        public static final String EMBEDDING_LENGTH = "llama.embedding_length";
        public static final String BLOCK_COUNT = "llama.block_count";
        public static final String FEED_FORWARD_LENGTH = "llama.feed_forward_length";
        public static final String ROPE_DIMENSION_COUNT = "llama.rope.dimension_count";
        public static final String ROPE_FREQ_BASE = "llama.rope.freq_base";
        public static final String ROPE_FREQ_SCALE = "llama.rope.freq_scale";
        public static final String ATTENTION_HEAD_COUNT = "llama.attention.head_count";
        public static final String ATTENTION_HEAD_COUNT_KV = "llama.attention.head_count_kv";
        private MetadataKeys() {}
    }
}
