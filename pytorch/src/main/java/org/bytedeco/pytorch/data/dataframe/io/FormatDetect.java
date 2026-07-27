package org.bytedeco.pytorch.data.dataframe.io;

import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Extension-based format detection for {@link DataFrame#read(String)}.
 *
 * <p>Supported extensions grow with the I/O stack:
 * {@code .csv .tsv .json .jsonl .ndjson .parquet .arrow .feather .ipc
 * .pkl .pickle .xlsx .xls .h5 .hdf5 .hdf .avro .orc .npz .npy
 * .safetensors .gguf .lance}.
 */
public final class FormatDetect {
    private FormatDetect() {}

    public enum Format {
        CSV, TSV, JSON, JSONL, PARQUET, ARROW, FEATHER, PICKLE,
        EXCEL, HDF5, AVRO, ORC, NPZ, NPY, SAFETENSORS, GGUF, LANCE, UNKNOWN
    }

    public static Format detect(String path) {
        if (path == null || path.isEmpty()) return Format.UNKNOWN;
        String name = path;
        int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        if (slash >= 0 && slash + 1 < path.length()) name = path.substring(slash + 1);
        String lower = name.toLowerCase(Locale.ROOT);

        if (lower.endsWith(".safetensors")) return Format.SAFETENSORS;
        if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) return Format.JSONL;
        if (lower.endsWith(".lance")) return Format.LANCE;

        // Directory heuristics for Lance datasets (no file extension required)
        try {
            Path p = Path.of(path);
            if (Files.isDirectory(p)) {
                if (Files.isRegularFile(p.resolve("_manifest.json"))
                    || Files.isDirectory(p.resolve("_versions"))
                    || (Files.isDirectory(p.resolve("data"))
                        && (Files.isDirectory(p.resolve("_versions"))
                            || Files.isDirectory(p.resolve("indices"))
                            || Files.isDirectory(p.resolve("vectors"))))) {
                    return Format.LANCE;
                }
            }
        } catch (Exception ignored) {
            // fall through to extension switch
        }

        int dot = lower.lastIndexOf('.');
        if (dot < 0 || dot == lower.length() - 1) return Format.UNKNOWN;
        String ext = lower.substring(dot + 1);
        switch (ext) {
            case "csv": return Format.CSV;
            case "tsv": return Format.TSV;
            case "json": return Format.JSON;
            case "parquet": case "pq": return Format.PARQUET;
            case "arrow": case "ipc": return Format.ARROW;
            case "feather": return Format.FEATHER;
            case "pkl": case "pickle": return Format.PICKLE;
            case "xlsx": case "xls": case "xlsm": return Format.EXCEL;
            case "h5": case "hdf5": case "hdf": return Format.HDF5;
            case "avro": return Format.AVRO;
            case "orc": return Format.ORC;
            case "npz": return Format.NPZ;
            case "npy": return Format.NPY;
            case "gguf": return Format.GGUF;
            case "lance": return Format.LANCE;
            default: return Format.UNKNOWN;
        }
    }

    public static Format detect(Path path) {
        return path == null ? Format.UNKNOWN : detect(path.toString());
    }

    /**
     * Detect format by extension, falling back to magic-byte sniff
     * ({@link SchemaInfer#sniff(String)}) when the extension is unknown.
     * Use this for robust multi-format loading.
     */
    public static Format detectRobust(String path) {
        Format fmt = detect(path);
        if (fmt != Format.UNKNOWN) return fmt;
        return SchemaInfer.sniff(path);
    }

    /**
     * Load a DataFrame by file extension (with magic-byte fallback).
     * HDF5 auto-read uses default key {@code /df}.
     */
    public static DataFrame read(String path) throws Exception {
        Format fmt = detectRobust(path);
        switch (fmt) {
            case CSV:
                return DataFrame.readCsv(path);
            case TSV:
                return DataFrame.readTsv(path);
            case JSON:
                return DataFrame.readJson(path);
            case JSONL:
                return DataFrame.readJsonl(path);
            case PARQUET:
                return DataFrame.readParquet(path);
            case ARROW:
            case FEATHER:
                return DataFrame.readArrow(path);
            case PICKLE:
                return DataFrame.readPickle(path);
            case NPZ:
                return DataFrame.readNpz(path);
            case NPY:
                return DataFrame.readNpy(path);
            case SAFETENSORS:
                return DataFrame.readSafetensors(path);
            case GGUF:
                return DataFrame.readGguf(path);
            case EXCEL:
                return DataFrame.readExcel(path);
            case HDF5:
                return DataFrame.readHdf(path, "/df");
            case AVRO:
                return DataFrame.readAvro(path);
            case ORC:
                return DataFrame.readOrc(path);
            case LANCE:
                return DataFrame.readLance(path);
            default:
                throw new IllegalArgumentException(
                    "Cannot auto-detect DataFrame format for path: " + path
                        + " (supported: csv,tsv,json,jsonl,parquet,arrow,feather,ipc,"
                        + "pkl,xlsx,xls,h5,hdf5,avro,orc,npz,npy,safetensors,gguf,lance;"
                        + " also magic-byte sniff for PAR1/ARROW1/NUMPY/ORC/HDF/Avro/JSON)");
        }
    }
}
