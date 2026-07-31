/*
 * Ported from torch-rechub-scala: torchrec/data/DatasetDownloader.scala
 *
 * HTTP downloader with disk caching, zip/gzip extraction, and lazy CSV/TSV line reader.
 * Cache root: ~/.torchrec-datasets
 */
package org.bytedeco.pytorch.recommend.data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class DatasetDownloader {

    private static final File CACHE_DIR;

    static {
        File home = new File(System.getProperty("user.home"));
        CACHE_DIR = new File(home, ".torchrec-datasets");
        if (!CACHE_DIR.exists()) {
            //noinspection ResultOfMethodCallIgnored
            CACHE_DIR.mkdirs();
        }
    }

    private DatasetDownloader() {}

    public static String cachePath() {
        return CACHE_DIR.getAbsolutePath();
    }

    public static File cacheDir() {
        return CACHE_DIR;
    }

    /**
     * Download a file from URL, cache locally, optionally extract zip/gzip.
     *
     * @return path to downloaded/extracted file or directory
     */
    public static File download(String urlString, String name) {
        return download(urlString, name, false);
    }

    public static File download(String urlString, String name, boolean forceRedownload) {
        boolean isZip = urlString.endsWith(".zip");
        boolean isGz = urlString.endsWith(".gz") || urlString.endsWith(".gzip");

        File targetFile;
        if (isZip) {
            targetFile = new File(CACHE_DIR, name + ".zip");
        } else if (isGz) {
            targetFile = new File(CACHE_DIR, name + ".txt");
        } else {
            targetFile = new File(CACHE_DIR, name);
        }

        boolean alreadyCached = false;
        if (!forceRedownload) {
            if (targetFile.isDirectory()) {
                alreadyCached = true;
            } else {
                alreadyCached = targetFile.exists() && targetFile.length() > 1000;
            }
            // For zip: also accept extracted sibling directory named `name`
            if (!alreadyCached && isZip) {
                File extracted = new File(CACHE_DIR, name);
                if (extracted.isDirectory() || (extracted.exists() && extracted.length() > 1000)) {
                    return extracted;
                }
            }
        }

        if (alreadyCached) {
            System.out.println("  [Cache] Using cached: " + targetFile.getAbsolutePath());
            return targetFile;
        }

        System.out.println("  [Download] " + urlString);
        System.out.println("  [Save to] " + targetFile.getAbsolutePath());

        File tempFile = new File(CACHE_DIR, name + ".tmp");
        if (tempFile.exists()) {
            //noinspection ResultOfMethodCallIgnored
            tempFile.delete();
        }

        try {
            downloadWithCurlOrJava(urlString, tempFile);
        } catch (Exception e) {
            //noinspection ResultOfMethodCallIgnored
            tempFile.delete();
            throw new RuntimeException("download failed for " + urlString + ": " + e.getMessage(), e);
        }

        if (!tempFile.exists() || tempFile.length() < 1000) {
            //noinspection ResultOfMethodCallIgnored
            tempFile.delete();
            throw new RuntimeException("download produced empty/small file for " + urlString);
        }

        System.out.println("  [Done] " + formatSize(tempFile.length()) + " downloaded");

        if (targetFile.exists()) {
            //noinspection ResultOfMethodCallIgnored
            targetFile.delete();
        }
        if (!tempFile.renameTo(targetFile)) {
            try {
                Files.copy(tempFile.toPath(), targetFile.toPath());
                //noinspection ResultOfMethodCallIgnored
                tempFile.delete();
            } catch (IOException ioe) {
                throw new RuntimeException("failed to move download to cache", ioe);
            }
        }

        if (isZip) {
            System.out.println("  [Extract] " + name + ".zip");
            return extractZip(targetFile, CACHE_DIR);
        } else if (isGz) {
            System.out.println("  [Extract] gzip to .txt");
            return extractGzip(targetFile, CACHE_DIR);
        }
        return targetFile;
    }

    /** Try multiple URLs; return first successful download. */
    public static File tryMirrors(List<String> urls, String name) {
        List<String> errors = new ArrayList<>();
        for (String url : urls) {
            try {
                System.out.println("  [Try] " + url);
                File file = download(url, name, false);
                if (file != null && file.exists()
                        && (file.isDirectory() || file.length() > 1000)) {
                    return file;
                }
            } catch (Throwable t) {
                System.out.println("  [Fail] " + t.getMessage());
                errors.add(t.getMessage());
            }
        }
        throw new RuntimeException("All mirrors failed for " + name + ": " + errors);
    }

    public static File tryMirrors(String[] urls, String name) {
        List<String> list = new ArrayList<>();
        CollectionsAddAll(list, urls);
        return tryMirrors(list, name);
    }

    private static void CollectionsAddAll(List<String> list, String[] urls) {
        for (String u : urls) list.add(u);
    }

    private static void downloadWithCurlOrJava(String urlString, File dest) throws Exception {
        // Prefer curl for progress + redirects (matches Scala)
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "curl", "-L",
                    "--progress-bar",
                    "--connect-timeout", "60",
                    "--max-time", "1800",
                    "--retry", "2",
                    "-o", dest.getAbsolutePath(),
                    urlString);
            pb.redirectErrorStream(true);
            Process p = pb.start();
            // drain stdout so process does not block
            try (InputStream in = p.getInputStream()) {
                byte[] buf = new byte[4096];
                while (in.read(buf) >= 0) { /* discard / progress optional */ }
            }
            int code = p.waitFor();
            if (code == 0 && dest.exists() && dest.length() > 1000) {
                return;
            }
            System.out.println("  [curl] exit=" + code + ", falling back to Java HttpURLConnection");
        } catch (Throwable t) {
            System.out.println("  [curl] unavailable: " + t.getMessage() + ", using Java download");
        }

        // Java fallback
        URL url = URI.create(urlString).toURL();
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setInstanceFollowRedirects(true);
        conn.setConnectTimeout(60_000);
        conn.setReadTimeout(1_800_000);
        conn.setRequestProperty("User-Agent", "torchrec-dataset-downloader/1.0");
        try (InputStream in = new BufferedInputStream(conn.getInputStream());
             OutputStream out = new BufferedOutputStream(new FileOutputStream(dest))) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) >= 0) {
                out.write(buf, 0, n);
            }
            out.flush();
        } finally {
            conn.disconnect();
        }
    }

    private static File extractZip(File zipFile, File destDir) {
        File firstExtracted = null;
        try (ZipInputStream zis = new ZipInputStream(
                new BufferedInputStream(new FileInputStream(zipFile)))) {
            ZipEntry entry;
            byte[] buffer = new byte[8192];
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.isDirectory() || entry.getName().contains("__MACOSX")) {
                    zis.closeEntry();
                    continue;
                }
                File outFile = new File(destDir, entry.getName());
                File parent = outFile.getParentFile();
                if (parent != null && !parent.exists()) {
                    //noinspection ResultOfMethodCallIgnored
                    parent.mkdirs();
                }
                try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(outFile))) {
                    int read;
                    while ((read = zis.read(buffer)) != -1) {
                        bos.write(buffer, 0, read);
                    }
                }
                if (firstExtracted == null) firstExtracted = outFile;
                zis.closeEntry();
            }
        } catch (IOException e) {
            throw new RuntimeException("zip extract failed: " + e.getMessage(), e);
        }
        return firstExtracted != null ? firstExtracted : destDir;
    }

    private static File extractGzip(File gzFile, File destDir) {
        String outName = gzFile.getName()
                .replaceAll("\\.gz$", "")
                .replaceAll("\\.gzip$", "");
        File outFile = new File(destDir, outName);
        if (outFile.exists()) return outFile;
        try (GZIPInputStream gis = new GZIPInputStream(new FileInputStream(gzFile));
             BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(outFile))) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = gis.read(buffer)) != -1) {
                bos.write(buffer, 0, read);
            }
            bos.flush();
            return outFile;
        } catch (IOException e) {
            throw new RuntimeException("gzip extract failed: " + e.getMessage(), e);
        }
    }

    private static String formatSize(long bytes) {
        if (bytes < 1024) return bytes + "B";
        if (bytes < 1024 * 1024) return (bytes / 1024) + "KB";
        if (bytes < 1024L * 1024 * 1024) return (bytes / (1024 * 1024)) + "MB";
        return (bytes / (1024L * 1024 * 1024)) + "GB";
    }

    /**
     * Lazy CSV/TSV line iterator. Caller should exhaust or close via try-with-resources
     * on the returned {@link LineIterator}.
     */
    public static LineIterator readLines(File file, String delimiter, boolean skipHeader, long maxLines) {
        return new LineIterator(file, delimiter, skipHeader, maxLines);
    }

    public static LineIterator readLines(File file, String delimiter, boolean skipHeader) {
        return readLines(file, delimiter, skipHeader, Long.MAX_VALUE);
    }

    public static LineIterator readLines(File file) {
        return readLines(file, "\t", false, Long.MAX_VALUE);
    }

    public static final class LineIterator implements Iterator<String[]>, AutoCloseable {
        private final BufferedReader reader;
        private final String delimiter;
        private final long maxLines;
        private String[] nextLine;
        private long linesRead;
        private boolean closed;

        LineIterator(File file, String delimiter, boolean skipHeader, long maxLines) {
            try {
                this.reader = new BufferedReader(new FileReader(file));
                this.delimiter = delimiter != null ? delimiter : "\t";
                this.maxLines = maxLines;
                this.linesRead = 0;
                this.closed = false;
                if (skipHeader) {
                    reader.readLine();
                }
                fetchNext();
            } catch (IOException e) {
                throw new RuntimeException("cannot open " + file, e);
            }
        }

        private void fetchNext() {
            if (closed || linesRead >= maxLines) {
                nextLine = null;
                return;
            }
            try {
                String line = reader.readLine();
                if (line == null) {
                    nextLine = null;
                    close();
                    return;
                }
                linesRead++;
                // -1 limit keeps trailing empty fields
                nextLine = line.split(java.util.regex.Pattern.quote(delimiter), -1);
            } catch (IOException e) {
                nextLine = null;
                close();
            }
        }

        @Override
        public boolean hasNext() {
            return nextLine != null;
        }

        @Override
        public String[] next() {
            if (nextLine == null) throw new NoSuchElementException();
            String[] cur = nextLine;
            fetchNext();
            return cur;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                try {
                    reader.close();
                } catch (IOException ignored) {
                }
            }
        }
    }
}
