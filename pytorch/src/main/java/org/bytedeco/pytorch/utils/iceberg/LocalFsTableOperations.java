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

import org.apache.iceberg.LocationProviders;
import org.apache.iceberg.TableMetadata;
import org.apache.iceberg.TableMetadataParser;
import org.apache.iceberg.TableOperations;
import org.apache.iceberg.exceptions.CommitFailedException;
import org.apache.iceberg.exceptions.RuntimeIOException;
import org.apache.iceberg.exceptions.ValidationException;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.io.LocationProvider;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hadoop-free {@link TableOperations} for local filesystem Iceberg warehouses.
 *
 * <p>Mirrors the public {@code HadoopTableOperations} protocol:
 * {@code metadata/version-hint.text} + atomic rename of {@code v{N}.metadata.json}.</p>
 *
 * <p>No Hadoop {@code Configuration} / {@code FileSystem} — uses {@link LocalFsFileIO}
 * and {@link java.nio.file.Files} only.</p>
 */
public final class LocalFsTableOperations implements TableOperations {

    private static final Pattern VERSION_PATTERN = Pattern.compile("v([^.]*)\\..*");
    private static final String VERSION_HINT = "version-hint.text";
    private static final TableMetadataParser.Codec[] CODECS = TableMetadataParser.Codec.values();

    private final Path tableLocation;
    private final FileIO fileIO;

    private volatile TableMetadata currentMetadata;
    private volatile Integer version;
    private volatile boolean shouldRefresh = true;

    public LocalFsTableOperations(Path tableLocation, FileIO fileIO) {
        this.tableLocation = Objects.requireNonNull(tableLocation, "tableLocation")
                .toAbsolutePath().normalize();
        this.fileIO = fileIO == null ? new LocalFsFileIO() : fileIO;
    }

    public Path tableLocation() {
        return tableLocation;
    }

    @Override
    public TableMetadata current() {
        if (shouldRefresh) {
            return refresh();
        }
        return currentMetadata;
    }

    @Override
    public synchronized TableMetadata refresh() {
        int ver = version != null ? version : findVersion();
        try {
            Path metadataFile = getMetadataFile(ver);
            if (version == null && metadataFile == null && ver == 0) {
                // table does not exist yet
                this.currentMetadata = null;
                this.version = null;
                this.shouldRefresh = false;
                return null;
            }
            if (metadataFile == null) {
                throw new ValidationException(
                        "Metadata file for version %d is missing under %s", ver, metadataRoot());
            }
            Path next = getMetadataFile(ver + 1);
            while (next != null) {
                ver += 1;
                metadataFile = next;
                next = getMetadataFile(ver + 1);
            }
            updateVersionAndMetadata(ver, LocalFsFileIO.toLocation(metadataFile));
            this.shouldRefresh = false;
            return currentMetadata;
        } catch (IOException e) {
            throw new RuntimeIOException(e, "Failed to refresh the table at %s", tableLocation);
        }
    }

    @Override
    public synchronized void commit(TableMetadata base, TableMetadata metadata) {
        if (base != currentMetadata) {
            throw new CommitFailedException("Cannot commit changes based on stale table metadata");
        }
        if (base == metadata) {
            return;
        }
        if (base != null && !Objects.equals(base.location(), metadata.location())) {
            throw new IllegalArgumentException("Local path-based tables cannot be relocated");
        }

        TableMetadataParser.Codec codec = TableMetadataParser.Codec.NONE;
        String fileExtension = TableMetadataParser.getFileExtension(codec);
        Path tempMetadataFile = metadataRoot().resolve(UUID.randomUUID() + fileExtension);
        try {
            Files.createDirectories(metadataRoot());
        } catch (IOException e) {
            throw new RuntimeIOException(e, "Failed to create metadata dir %s", metadataRoot());
        }
        TableMetadataParser.write(metadata, fileIO.newOutputFile(LocalFsFileIO.toLocation(tempMetadataFile)));

        int nextVersion = (version != null ? version : 0) + 1;
        Path finalMetadataFile = metadataFilePath(nextVersion, codec);

        try {
            // atomic commit: rename temp → final (fail if dest exists)
            if (Files.exists(finalMetadataFile)) {
                Files.deleteIfExists(tempMetadataFile);
                throw new CommitFailedException("Failed to commit changes using rename: %s already exists",
                        finalMetadataFile);
            }
            try {
                Files.move(tempMetadataFile, finalMetadataFile, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicFail) {
                Files.move(tempMetadataFile, finalMetadataFile);
            }
        } catch (CommitFailedException e) {
            throw e;
        } catch (IOException e) {
            try { Files.deleteIfExists(tempMetadataFile); } catch (IOException ignored) {}
            throw new CommitFailedException(e, "Failed to commit changes using rename: %s", finalMetadataFile);
        }

        writeVersionHint(nextVersion);
        this.shouldRefresh = true;
    }

    @Override
    public FileIO io() {
        return fileIO;
    }

    @Override
    public String metadataFileLocation(String fileName) {
        return LocalFsFileIO.toLocation(metadataRoot().resolve(fileName));
    }

    @Override
    public LocationProvider locationProvider() {
        TableMetadata meta = current();
        if (meta == null) {
            return LocationProviders.locationsFor(LocalFsFileIO.toLocation(tableLocation), java.util.Map.of());
        }
        return LocationProviders.locationsFor(meta.location(), meta.properties());
    }

    @Override
    public TableOperations temp(TableMetadata uncommittedMetadata) {
        LocalFsTableOperations self = this;
        return new TableOperations() {
            @Override
            public TableMetadata current() {
                return uncommittedMetadata;
            }

            @Override
            public TableMetadata refresh() {
                throw new UnsupportedOperationException("Cannot call refresh on temporary table operations");
            }

            @Override
            public void commit(TableMetadata base, TableMetadata metadata) {
                throw new UnsupportedOperationException("Cannot call commit on temporary table operations");
            }

            @Override
            public String metadataFileLocation(String fileName) {
                return self.metadataFileLocation(fileName);
            }

            @Override
            public LocationProvider locationProvider() {
                return LocationProviders.locationsFor(
                        uncommittedMetadata.location(), uncommittedMetadata.properties());
            }

            @Override
            public FileIO io() {
                return self.io();
            }

            @Override
            public long newSnapshotId() {
                return self.newSnapshotId();
            }
        };
    }

    private synchronized void updateVersionAndMetadata(int newVersion, String metadataFile) {
        if (version == null || version != newVersion) {
            this.version = newVersion;
            TableMetadata loaded = TableMetadataParser.read(fileIO, metadataFile);
            if (currentMetadata != null && currentMetadata.uuid() != null
                    && loaded.uuid() != null
                    && !loaded.uuid().equals(currentMetadata.uuid())) {
                throw new ValidationException(
                        "Table UUID does not match: %s != %s", loaded.uuid(), currentMetadata.uuid());
            }
            this.currentMetadata = loaded;
        }
    }

    private Path getMetadataFile(int metadataVersion) throws IOException {
        for (TableMetadataParser.Codec codec : CODECS) {
            Path metadataFile = metadataFilePath(metadataVersion, codec);
            if (Files.exists(metadataFile)) {
                return metadataFile;
            }
            if (codec == TableMetadataParser.Codec.GZIP) {
                // backward-compat: vN.metadata.json.gz
                Path old = metadataRoot().resolve(
                        "v" + metadataVersion + TableMetadataParser.getOldFileExtension(codec));
                if (Files.exists(old)) return old;
            }
        }
        return null;
    }

    private Path metadataFilePath(int metadataVersion, TableMetadataParser.Codec codec) {
        return metadataRoot().resolve("v" + metadataVersion + TableMetadataParser.getFileExtension(codec));
    }

    private Path metadataRoot() {
        return tableLocation.resolve("metadata");
    }

    private int findVersion() {
        Path hint = metadataRoot().resolve(VERSION_HINT);
        try {
            if (Files.exists(hint)) {
                String line = Files.readString(hint, StandardCharsets.UTF_8).trim();
                // may contain trailing newline already stripped by trim
                return Integer.parseInt(line.replace("\n", "").trim());
            }
        } catch (Exception e) {
            // fall through to directory scan
        }
        try {
            if (!Files.isDirectory(metadataRoot())) {
                return 0;
            }
            int maxVersion = 0;
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(metadataRoot())) {
                for (Path p : stream) {
                    String name = p.getFileName().toString();
                    int v = versionOf(name);
                    if (v > maxVersion) {
                        try {
                            if (getMetadataFile(v) != null) maxVersion = v;
                        } catch (IOException ignored) {}
                    }
                }
            }
            return maxVersion;
        } catch (IOException e) {
            return 0;
        }
    }

    private static int versionOf(String fileName) {
        Matcher matcher = VERSION_PATTERN.matcher(fileName);
        if (!matcher.matches()) return -1;
        try {
            return Integer.parseInt(matcher.group(1));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private void writeVersionHint(int versionToWrite) {
        Path hint = metadataRoot().resolve(VERSION_HINT);
        Path tmp = metadataRoot().resolve(UUID.randomUUID() + "-version-hint.temp");
        try {
            Files.createDirectories(metadataRoot());
            Files.writeString(tmp, String.valueOf(versionToWrite), StandardCharsets.UTF_8);
            Files.deleteIfExists(hint);
            try {
                Files.move(tmp, hint, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                Files.move(tmp, hint, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            try { Files.deleteIfExists(tmp); } catch (IOException ignored) {}
            // best-effort — commit already succeeded via metadata rename
        }
    }

    /** Whether a table already exists at this location (has metadata). */
    public boolean tableExists() {
        TableMetadata meta = refresh();
        return meta != null;
    }
}
