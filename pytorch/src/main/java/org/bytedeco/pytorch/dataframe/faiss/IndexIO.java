package org.bytedeco.pytorch.dataframe.faiss;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Legacy JDF1 Java-serialization persistence for FAISS-like indexes.
 *
 * <p><b>Not</b> wire-compatible with C++/Python {@code faiss.write_index}.
 * Prefer {@link NativeFaissIO} / {@link Faiss#write_index} for Python interop.
 * This format remains useful for types or graphs that need lossless Java
 * {@code Serializable} round-trips.
 *
 * <p>Format:
 * <pre>
 *   magic "JDF1" (4 bytes, big-endian)
 *   version int32 = 1
 *   type UTF
 *   payload: Java serialized Index
 * </pre>
 */
public final class IndexIO {
    public static final int MAGIC = 0x4A444631; // 'JDF1'
    public static final int VERSION = 1;

    private IndexIO() {}

    public static void write(Index index, String path) throws IOException {
        write(index, Path.of(path));
    }

    public static void write(Index index, Path path) throws IOException {
        // GPU indexes must be brought back to CPU first (FAISS parity)
        if (index.is_gpu()) {
            index.to_cpu_storage();
        }
        try (OutputStream raw = Files.newOutputStream(path);
             DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(raw))) {
            dos.writeInt(MAGIC);
            dos.writeInt(VERSION);
            dos.writeUTF(index.indexType());
            ObjectOutputStream oos = new ObjectOutputStream(dos);
            oos.writeObject(index);
            oos.flush();
        }
    }

    public static Index read(String path) throws IOException, ClassNotFoundException {
        return read(Path.of(path));
    }

    public static Index read(Path path) throws IOException, ClassNotFoundException {
        try (InputStream raw = Files.newInputStream(path);
             DataInputStream dis = new DataInputStream(new BufferedInputStream(raw))) {
            int magic = dis.readInt();
            if (magic != MAGIC) {
                throw new IOException("Not a javacpp-dataframe FAISS index (bad magic). "
                    + "This format is NOT compatible with C++ faiss.read_index.");
            }
            int ver = dis.readInt();
            if (ver > VERSION) {
                throw new IOException("Unsupported index version " + ver);
            }
            dis.readUTF(); // type tag, informational
            ObjectInputStream ois = new ObjectInputStream(dis);
            Object obj = ois.readObject();
            if (!(obj instanceof Index)) {
                throw new IOException("Payload is not an Index: " + obj);
            }
            return (Index) obj;
        }
    }
}
