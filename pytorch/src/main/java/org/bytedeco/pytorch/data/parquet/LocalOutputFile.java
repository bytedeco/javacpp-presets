package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.parquet.io.OutputFile;
import org.apache.parquet.io.PositionOutputStream;

/**
 * Pure-Java {@link OutputFile} backed by a local {@link Path}.
 * No Hadoop Path / FileSystem / UGI.
 */
public final class LocalOutputFile implements OutputFile {

    private final Path path;

    public LocalOutputFile(String path) { this.path = Paths.get(path); }

    @Override
    public PositionOutputStream create(long blockSizeHint) throws IOException {
        return new LocalPositionOutputStream(Files.newOutputStream(path), path.toString());
    }

    @Override
    public PositionOutputStream createOrOverwrite(long blockSizeHint) throws IOException {
        return new LocalPositionOutputStream(
            Files.newOutputStream(Files.exists(path) ? path : path),
            path.toString()
        );
    }

    @Override
    public boolean supportsBlockSize() { return false; }

    @Override
    public long defaultBlockSize() { return 128 * 1024 * 1024L; }

    private static final class LocalPositionOutputStream extends PositionOutputStream {
        private final OutputStream out;
        private final String name;
        private long position;

        LocalPositionOutputStream(OutputStream out, String name) {
            this.out = out;
            this.name = name;
        }

        @Override
        public long getPos() { return position; }

        @Override
        public void write(int b) throws IOException {
            out.write(b);
            position++;
        }

        @Override
        public void write(byte[] buf, int off, int len) throws IOException {
            out.write(buf, off, len);
            position += len;
        }

        @Override
        public void flush() throws IOException { out.flush(); }

        @Override
        public void close() throws IOException { out.close(); }
    }
}
