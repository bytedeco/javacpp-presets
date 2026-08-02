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
package org.bytedeco.pytorch.utils.iceberg;

import org.apache.iceberg.exceptions.NotFoundException;
import org.apache.iceberg.exceptions.RuntimeIOException;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.io.InputFile;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.io.PositionOutputStream;
import org.apache.iceberg.io.SeekableInputStream;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * Pure-Java {@link FileIO} for local filesystem paths — no Hadoop dependency.
 *
 * <p>Accepts {@code file://} URIs and plain absolute/relative paths.
 * Used with Iceberg {@code TableMetadataParser} / {@code BaseTable} for warehouse access.</p>
 */
public final class LocalFsFileIO implements FileIO {

    @Override
    public InputFile newInputFile(String path) {
        return new LocalInputFile(toPath(path));
    }

    @Override
    public OutputFile newOutputFile(String path) {
        return new LocalOutputFile(toPath(path));
    }

    @Override
    public void deleteFile(String path) {
        try {
            Files.deleteIfExists(toPath(path));
        } catch (IOException e) {
            throw new RuntimeIOException(e, "Failed to delete file: %s", path);
        }
    }

    @Override
    public void initialize(Map<String, String> properties) {
        // no-op
    }

    @Override
    public void close() {
        // no-op
    }

    static Path toPath(String location) {
        if (location == null || location.isBlank()) {
            throw new IllegalArgumentException("path required");
        }
        String s = location.trim();
        if (s.startsWith("file://")) s = s.substring("file://".length());
        else if (s.startsWith("file:")) s = s.substring("file:".length());
        return Path.of(s);
    }

    static String toLocation(Path path) {
        return path.toAbsolutePath().normalize().toString();
    }

    static final class LocalInputFile implements InputFile {
        private final Path path;

        LocalInputFile(Path path) {
            this.path = path;
        }

        @Override
        public long getLength() {
            try {
                return Files.size(path);
            } catch (IOException e) {
                throw new RuntimeIOException(e, "Failed to get length: %s", path);
            }
        }

        @Override
        public SeekableInputStream newStream() {
            try {
                if (!Files.exists(path)) {
                    throw new NotFoundException("File does not exist: %s", path);
                }
                RandomAccessFile raf = new RandomAccessFile(path.toFile(), "r");
                return new SeekableInputStream() {
                    @Override
                    public void seek(long newPos) throws IOException {
                        raf.seek(newPos);
                    }

                    @Override
                    public long getPos() throws IOException {
                        return raf.getFilePointer();
                    }

                    @Override
                    public int read() throws IOException {
                        return raf.read();
                    }

                    @Override
                    public int read(byte[] b, int off, int len) throws IOException {
                        return raf.read(b, off, len);
                    }

                    @Override
                    public void close() throws IOException {
                        raf.close();
                    }
                };
            } catch (IOException e) {
                if (e instanceof FileNotFoundException) {
                    throw new NotFoundException(e, "File does not exist: %s", path);
                }
                throw new RuntimeIOException(e, "Failed to open: %s", path);
            }
        }

        @Override
        public String location() {
            return toLocation(path);
        }

        @Override
        public boolean exists() {
            return Files.exists(path);
        }
    }

    static final class LocalOutputFile implements OutputFile {
        private final Path path;

        LocalOutputFile(Path path) {
            this.path = path;
        }

        @Override
        public PositionOutputStream create() {
            return open(false);
        }

        @Override
        public PositionOutputStream createOrOverwrite() {
            return open(true);
        }

        private PositionOutputStream open(boolean overwrite) {
            try {
                Path parent = path.getParent();
                if (parent != null) Files.createDirectories(parent);
                if (!overwrite && Files.exists(path)) {
                    throw new RuntimeIOException("File already exists: %s", path);
                }
                // write to temp then move for createOrOverwrite atomicity best-effort
                Path tmp = path.resolveSibling(path.getFileName() + ".tmp." + Thread.currentThread().getId());
                var out = Files.newOutputStream(tmp,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE);
                return new PositionOutputStream() {
                    private long pos = 0;
                    private boolean closed = false;

                    @Override
                    public long getPos() {
                        return pos;
                    }

                    @Override
                    public void write(int b) throws IOException {
                        out.write(b);
                        pos++;
                    }

                    @Override
                    public void write(byte[] b, int off, int len) throws IOException {
                        out.write(b, off, len);
                        pos += len;
                    }

                    @Override
                    public void close() throws IOException {
                        if (closed) return;
                        closed = true;
                        out.close();
                        try {
                            if (overwrite) {
                                Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
                            } else {
                                Files.move(tmp, path, StandardCopyOption.ATOMIC_MOVE);
                            }
                        } catch (IOException atomicFail) {
                            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
                        }
                    }
                };
            } catch (IOException e) {
                throw new RuntimeIOException(e, "Failed to create output: %s", path);
            }
        }

        @Override
        public String location() {
            return toLocation(path);
        }

        @Override
        public InputFile toInputFile() {
            return new LocalInputFile(path);
        }
    }
}
