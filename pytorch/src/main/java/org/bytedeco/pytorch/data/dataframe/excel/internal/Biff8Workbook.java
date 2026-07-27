package org.bytedeco.pytorch.data.dataframe.excel.internal;

import org.bytedeco.pytorch.data.dataframe.excel.ExcelParseException;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Best-effort BIFF8 (.xls) reader for tabular sheets.
 * Supports SST, LabelSST, Number, RK, BoolErr, Blank, and formula cached results.
 * Does not write .xls and does not support encryption/VBA/charts.
 */
public final class Biff8Workbook {
    // BIFF record types
    private static final int BOF = 0x0809;
    private static final int EOF = 0x000A;
    private static final int BOUNDSHEET = 0x0085;
    private static final int SST = 0x00FC;
    private static final int CONTINUE = 0x003C;
    private static final int LABELSST = 0x00FD;
    private static final int NUMBER = 0x0203;
    private static final int RK = 0x027E;
    private static final int BOOLERR = 0x0205;
    private static final int BLANK = 0x0201;
    private static final int LABEL = 0x0204;
    private static final int FORMULA = 0x0006;
    private static final int STRING = 0x0207;
    private static final int MULRK = 0x00BD;
    private static final int XF = 0x00E0;
    private static final int FORMAT = 0x041E;
    private static final int ROW = 0x0208;
    private static final int INDEX = 0x020B;
    private static final int DIMENSION = 0x0200;

    public static final class Sheet {
        public final String name;
        public final List<Object[]> rows;
        public final int maxCol;

        public Sheet(String name, List<Object[]> rows, int maxCol) {
            this.name = name;
            this.rows = rows;
            this.maxCol = maxCol;
        }
    }

    private final List<Sheet> sheets = new ArrayList<>();

    public List<Sheet> sheets() { return sheets; }

    public static Biff8Workbook read(InputStream in) throws Exception {
        byte[] all = readAll(in);
        if (all.length < 8) throw new ExcelParseException("File too small for .xls");
        // OLE CFB magic
        if ((all[0] & 0xFF) == 0xD0 && (all[1] & 0xFF) == 0xCF
            && (all[2] & 0xFF) == 0x11 && (all[3] & 0xFF) == 0xE0) {
            byte[] workbook = OleCfb.extractStream(all, "Workbook");
            if (workbook == null) workbook = OleCfb.extractStream(all, "Book");
            if (workbook == null) {
                throw new ExcelParseException("OLE container has no Workbook stream");
            }
            return parseBiff(workbook);
        }
        // raw BIFF?
        if ((all[0] & 0xFF) == 0x09 && (all[1] & 0xFF) == 0x08) {
            return parseBiff(all);
        }
        throw new ExcelParseException("Not a BIFF/OLE .xls workbook");
    }

    private static Biff8Workbook parseBiff(byte[] data) {
        Biff8Workbook wb = new Biff8Workbook();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        List<BoundSheet> bounds = new ArrayList<>();
        List<String> sst = new ArrayList<>();
        Map<Integer, Boolean> dateXf = new HashMap<>();

        // First pass: global records until first sheet BOF after bounds
        while (buf.remaining() >= 4) {
            int pos = buf.position();
            int type = buf.getShort() & 0xFFFF;
            int len = buf.getShort() & 0xFFFF;
            if (len < 0 || len > buf.remaining()) break;
            int next = buf.position() + len;
            byte[] recBytes = new byte[len];
            buf.get(recBytes);
            ByteBuffer rec = ByteBuffer.wrap(recBytes).order(ByteOrder.LITTLE_ENDIAN);

            if (type == BOUNDSHEET && len >= 8) {
                int offset = rec.getInt();
                rec.get(); // state
                rec.get(); // type
                String name = readShortString(rec);
                bounds.add(new BoundSheet(offset, name));
            } else if (type == SST) {
                sst = readSst(data, pos, type, len);
            } else if (type == XF && len >= 4) {
                // XF: numFmtId at offset 2 for BIFF8
                int numFmtId = rec.getShort() & 0xFFFF;
                int xfIndex = dateXf.size();
                dateXf.put(xfIndex, ExcelDateUtil.isDateFormat(numFmtId, null));
            }
            buf.position(next);
            if (type == EOF && bounds.size() > 0) {
                break;
            }
        }

        // Parse each sheet at BoundSheet offset
        if (bounds.isEmpty()) {
            // single sheet starting at 0
            Sheet s = parseSheet(data, 0, "Sheet1", sst, dateXf);
            if (s != null) wb.sheets.add(s);
        } else {
            for (BoundSheet b : bounds) {
                Sheet s = parseSheet(data, b.offset, b.name, sst, dateXf);
                if (s != null) wb.sheets.add(s);
            }
        }
        return wb;
    }

    private static Sheet parseSheet(byte[] data, int offset, String name,
                                    List<String> sst, Map<Integer, Boolean> dateXf) {
        if (offset < 0 || offset >= data.length) return null;
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buf.position(offset);
        Map<Long, Object> cells = new HashMap<>();
        int maxRow = -1, maxCol = -1;
        String pendingFormulaString = null;
        int lastFormulaRow = -1, lastFormulaCol = -1;

        while (buf.remaining() >= 4) {
            int type = buf.getShort() & 0xFFFF;
            int len = buf.getShort() & 0xFFFF;
            if (len < 0 || len > buf.remaining()) break;
            int next = buf.position() + len;
            byte[] recBytes = new byte[len];
            buf.get(recBytes);
            ByteBuffer rec = ByteBuffer.wrap(recBytes).order(ByteOrder.LITTLE_ENDIAN);

            if (type == EOF) break;
            if (type == BOF) {
                buf.position(next);
                continue;
            }

            try {
                if (type == LABELSST && len >= 10) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    rec.getShort(); // xf
                    int idx = rec.getInt();
                    String v = (idx >= 0 && idx < sst.size()) ? sst.get(idx) : "";
                    put(cells, row, col, v);
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                } else if (type == NUMBER && len >= 14) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    int xf = rec.getShort() & 0xFFFF;
                    double num = Double.longBitsToDouble(rec.getLong());
                    Object v = decodeNumber(num, xf, dateXf);
                    put(cells, row, col, v);
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                } else if (type == RK && len >= 10) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    int xf = rec.getShort() & 0xFFFF;
                    int rk = rec.getInt();
                    double num = decodeRk(rk);
                    Object v = decodeNumber(num, xf, dateXf);
                    put(cells, row, col, v);
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                } else if (type == MULRK && len >= 6) {
                    int row = rec.getShort() & 0xFFFF;
                    int firstCol = rec.getShort() & 0xFFFF;
                    int remaining = len - 6;
                    int n = remaining / 6;
                    for (int i = 0; i < n; i++) {
                        int xf = rec.getShort() & 0xFFFF;
                        int rk = rec.getInt();
                        int col = firstCol + i;
                        Object v = decodeNumber(decodeRk(rk), xf, dateXf);
                        put(cells, row, col, v);
                        maxCol = Math.max(maxCol, col);
                    }
                    maxRow = Math.max(maxRow, row);
                } else if (type == BOOLERR && len >= 8) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    rec.getShort(); // xf
                    int b = rec.get() & 0xFF;
                    int err = rec.get() & 0xFF;
                    Object v = (err == 0) ? (b != 0) : null;
                    put(cells, row, col, v);
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                } else if (type == LABEL && len >= 8) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    rec.getShort(); // xf
                    String v = readShortString(rec);
                    put(cells, row, col, v);
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                } else if (type == FORMULA && len >= 14) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    int xf = rec.getShort() & 0xFFFF;
                    long bits = rec.getLong();
                    Object v;
                    if ((bits & 0xFFFFFFFFFF000000L) == 0xFFFF000000000000L) {
                        int rt = (int) (bits & 0xFF);
                        if (rt == 0) {
                            lastFormulaRow = row;
                            lastFormulaCol = col;
                            v = null;
                        } else if (rt == 1) {
                            v = ((bits >> 16) & 0xFF) != 0;
                        } else {
                            v = null;
                        }
                    } else {
                        double num = Double.longBitsToDouble(bits);
                        v = decodeNumber(num, xf, dateXf);
                    }
                    if (v != null) put(cells, row, col, v);
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                } else if (type == STRING) {
                    String s = readShortString(rec);
                    if (lastFormulaRow >= 0) {
                        put(cells, lastFormulaRow, lastFormulaCol, s);
                        lastFormulaRow = -1;
                    }
                } else if (type == BLANK && len >= 6) {
                    int row = rec.getShort() & 0xFFFF;
                    int col = rec.getShort() & 0xFFFF;
                    maxRow = Math.max(maxRow, row);
                    maxCol = Math.max(maxCol, col);
                }
            } catch (Exception ignored) {
                // best-effort: skip bad record
            }
            buf.position(next);
        }

        List<Object[]> rows = new ArrayList<>();
        if (maxRow >= 0 && maxCol >= 0) {
            for (int r = 0; r <= maxRow; r++) {
                Object[] row = new Object[maxCol + 1];
                for (int c = 0; c <= maxCol; c++) {
                    row[c] = cells.get(key(r, c));
                }
                rows.add(row);
            }
        }
        return new Sheet(name == null ? "Sheet1" : name, rows, maxCol + 1);
    }

    private static Object decodeNumber(double num, int xf, Map<Integer, Boolean> dateXf) {
        if (dateXf != null && Boolean.TRUE.equals(dateXf.get(xf))) {
            java.time.LocalDateTime ldt = ExcelDateUtil.fromSerial(num);
            if (ldt.toLocalTime().equals(java.time.LocalTime.MIDNIGHT)) return ldt.toLocalDate();
            return ldt;
        }
        if (num == Math.rint(num) && !Double.isInfinite(num)
            && num <= Long.MAX_VALUE && num >= Long.MIN_VALUE) {
            return (long) num;
        }
        return num;
    }

    private static double decodeRk(int rk) {
        boolean div100 = (rk & 0x1) != 0;
        boolean isInt = (rk & 0x2) != 0;
        double val;
        if (isInt) {
            val = rk >> 2;
        } else {
            long bits = ((long) (rk & 0xFFFFFFFC)) << 32;
            val = Double.longBitsToDouble(bits);
        }
        if (div100) val /= 100.0;
        return val;
    }

    private static void put(Map<Long, Object> cells, int row, int col, Object v) {
        cells.put(key(row, col), v);
    }

    private static long key(int row, int col) {
        return (((long) row) << 32) | (col & 0xFFFFFFFFL);
    }

    private static List<String> readSst(byte[] data, int recPos, int type, int len) {
        // recPos points at record header; data after header
        List<String> out = new ArrayList<>();
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        buf.position(recPos + 4);
        if (len < 8) return out;
        int total = buf.getInt(); // total strings
        int unique = buf.getInt();
        int endOfRec = recPos + 4 + len;
        int remainingUnique = unique;
        try {
            while (remainingUnique > 0 && buf.position() < data.length) {
                if (buf.position() >= endOfRec) {
                    // need CONTINUE
                    if (buf.remaining() < 4) break;
                    int t = buf.getShort() & 0xFFFF;
                    int l = buf.getShort() & 0xFFFF;
                    if (t != CONTINUE) break;
                    endOfRec = buf.position() + l;
                }
                if (buf.remaining() < 3) break;
                int cch = buf.getShort() & 0xFFFF;
                int flags = buf.get() & 0xFF;
                boolean compressed = (flags & 0x01) == 0;
                boolean ext = (flags & 0x04) != 0;
                boolean rich = (flags & 0x08) != 0;
                int runCount = 0;
                int extSize = 0;
                if (rich) {
                    if (buf.remaining() < 2) break;
                    runCount = buf.getShort() & 0xFFFF;
                }
                if (ext) {
                    if (buf.remaining() < 4) break;
                    extSize = buf.getInt();
                }
                StringBuilder sb = new StringBuilder(cch);
                int charsRead = 0;
                while (charsRead < cch) {
                    if (buf.position() >= endOfRec) {
                        if (buf.remaining() < 4) break;
                        int t = buf.getShort() & 0xFFFF;
                        int l = buf.getShort() & 0xFFFF;
                        if (t != CONTINUE) break;
                        endOfRec = buf.position() + l;
                        if (buf.remaining() < 1) break;
                        // CONTINUE starts with option flags for encoding
                        flags = buf.get() & 0xFF;
                        compressed = (flags & 0x01) == 0;
                    }
                    if (compressed) {
                        if (buf.remaining() < 1) break;
                        sb.append((char) (buf.get() & 0xFF));
                    } else {
                        if (buf.remaining() < 2) break;
                        sb.append((char) (buf.getShort() & 0xFFFF));
                    }
                    charsRead++;
                }
                // skip rich runs and ext
                int skip = runCount * 4 + Math.max(0, extSize);
                if (skip > 0 && buf.remaining() >= skip) buf.position(buf.position() + skip);
                out.add(sb.toString());
                remainingUnique--;
            }
        } catch (Exception ignored) {}
        return out;
    }

    private static String readShortString(ByteBuffer rec) {
        if (rec.remaining() < 3) return "";
        int cch = rec.getShort() & 0xFFFF;
        int flags = rec.get() & 0xFF;
        boolean compressed = (flags & 0x01) == 0;
        if (compressed) {
            if (rec.remaining() < cch) cch = rec.remaining();
            byte[] b = new byte[cch];
            rec.get(b);
            return new String(b, StandardCharsets.ISO_8859_1);
        } else {
            if (rec.remaining() < cch * 2) cch = rec.remaining() / 2;
            char[] ch = new char[cch];
            for (int i = 0; i < cch; i++) ch[i] = (char) (rec.getShort() & 0xFFFF);
            return new String(ch);
        }
    }

    private static byte[] readAll(InputStream in) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) >= 0) bos.write(buf, 0, n);
        return bos.toByteArray();
    }

    private static final class BoundSheet {
        final int offset;
        final String name;
        BoundSheet(int offset, String name) { this.offset = offset; this.name = name; }
    }

    /**
     * Minimal OLE Compound File Binary reader for a single named stream.
     */
    static final class OleCfb {
        static byte[] extractStream(byte[] data, String streamName) {
            if (data.length < 512) return null;
            ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            // header
            bb.position(0x1E);
            int sectorShift = bb.getShort() & 0xFFFF;
            int sectorSize = 1 << sectorShift;
            bb.position(0x2C);
            int fatSectors = bb.getInt();
            int firstDirSector = bb.getInt();
            bb.position(0x3C);
            int miniCutoff = bb.getInt();
            int firstMiniFat = bb.getInt();
            bb.position(0x44);
            int firstDifat = bb.getInt();
            // DIFAT in header at 0x4C — 109 entries
            List<Integer> fatSectorList = new ArrayList<>();
            bb.position(0x4C);
            for (int i = 0; i < 109; i++) {
                int s = bb.getInt();
                if (s >= 0) fatSectorList.add(s);
            }
            // build FAT (sequential entries across FAT sectors listed in DIFAT)
            Map<Integer, Integer> fat = new HashMap<>();
            int fatBase = 0;
            int entriesPerSector = sectorSize / 4;
            for (int secId : fatSectorList) {
                int off = (secId + 1) * sectorSize;
                if (off + sectorSize > data.length) continue;
                ByteBuffer fatBuf = ByteBuffer.wrap(data, off, sectorSize).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < entriesPerSector; i++) {
                    fat.put(fatBase + i, fatBuf.getInt());
                }
                fatBase += entriesPerSector;
            }
            // directory entries: 128 bytes each, chain from firstDirSector
            List<DirEntry> dirs = new ArrayList<>();
            int dirSec = firstDirSector;
            int guard = 0;
            while (dirSec >= 0 && dirSec < 0xFFFFFFFA && guard++ < 10000) {
                int off = (dirSec + 1) * sectorSize;
                if (off + sectorSize > data.length) break;
                for (int i = 0; i < sectorSize / 128; i++) {
                    dirs.add(readDir(data, off + i * 128));
                }
                Integer next = fat.get(dirSec);
                if (next == null) break;
                dirSec = next;
            }
            DirEntry root = dirs.isEmpty() ? null : dirs.get(0);
            DirEntry target = null;
            for (DirEntry d : dirs) {
                if (d == null || d.name == null) continue;
                if (streamName.equalsIgnoreCase(d.name) && d.type == 2) {
                    target = d;
                    break;
                }
            }
            if (target == null) return null;
            if (target.size < miniCutoff && root != null) {
                // mini stream
                byte[] mini = readChain(data, sectorSize, fat, root.startSector, (int) root.size);
                return readMini(data, sectorSize, fat, firstMiniFat, mini, target.startSector, (int) target.size, miniCutoff);
            }
            return readChain(data, sectorSize, fat, target.startSector, (int) target.size);
        }

        private static byte[] readMini(byte[] data, int sectorSize, Map<Integer, Integer> fat,
                                       int firstMiniFat, byte[] miniStream, int start, int size, int miniCutoff) {
            int miniSize = 64;
            // build mini FAT from firstMiniFat chain
            Map<Integer, Integer> mfat = new HashMap<>();
            int sec = firstMiniFat;
            int guard = 0;
            int base = 0;
            while (sec >= 0 && sec < 0xFFFFFFFA && guard++ < 10000) {
                int off = (sec + 1) * sectorSize;
                if (off + sectorSize > data.length) break;
                ByteBuffer fb = ByteBuffer.wrap(data, off, sectorSize).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < sectorSize / 4; i++) {
                    mfat.put(base + i, fb.getInt());
                }
                base += sectorSize / 4;
                Integer next = fat.get(sec);
                if (next == null) break;
                sec = next;
            }
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            int cur = start;
            int left = size;
            guard = 0;
            while (cur >= 0 && cur < 0xFFFFFFFA && left > 0 && guard++ < 100000) {
                int off = cur * miniSize;
                int n = Math.min(miniSize, left);
                if (off + n <= miniStream.length) bos.write(miniStream, off, n);
                left -= n;
                Integer next = mfat.get(cur);
                if (next == null) break;
                cur = next;
            }
            return bos.toByteArray();
        }

        private static byte[] readChain(byte[] data, int sectorSize, Map<Integer, Integer> fat,
                                        int start, int size) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            int cur = start;
            int left = size;
            int guard = 0;
            while (cur >= 0 && cur < 0xFFFFFFFA && left > 0 && guard++ < 100000) {
                int off = (cur + 1) * sectorSize;
                int n = Math.min(sectorSize, left);
                if (off + n <= data.length) bos.write(data, off, n);
                left -= n;
                Integer next = fat.get(cur);
                if (next == null) break;
                cur = next;
            }
            return bos.toByteArray();
        }

        private static DirEntry readDir(byte[] data, int off) {
            if (off + 128 > data.length) return null;
            ByteBuffer bb = ByteBuffer.wrap(data, off, 128).order(ByteOrder.LITTLE_ENDIAN);
            char[] nameChars = new char[32];
            for (int i = 0; i < 32; i++) nameChars[i] = (char) (bb.getShort() & 0xFFFF);
            int nameLen = bb.getShort() & 0xFFFF; // bytes including NUL
            int type = bb.get() & 0xFF;
            bb.get(); // color
            bb.getInt(); // left
            bb.getInt(); // right
            bb.getInt(); // child
            bb.position(bb.position() + 16); // clsid
            bb.getInt(); // state
            bb.getLong(); // created
            bb.getLong(); // modified
            int start = bb.getInt();
            long size = bb.getLong();
            int chars = Math.max(0, (nameLen / 2) - 1);
            if (chars > 31) chars = 31;
            String name = new String(nameChars, 0, chars);
            DirEntry d = new DirEntry();
            d.name = name;
            d.type = type;
            d.startSector = start;
            d.size = size;
            return d;
        }

        static final class DirEntry {
            String name;
            int type;
            int startSector;
            long size;
        }
    }
}
