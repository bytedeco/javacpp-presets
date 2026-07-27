package org.bytedeco.pytorch.data.dataframe.excel;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.excel.internal.OoxmlZip;

import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Excel writer ({@code .xlsx}) with dtype-aware cells. Pure OOXML (Zip + XML), no Apache POI.
 *
 * <p>Does not write {@code .xls}. Large frames stream sheet XML via in-memory parts then zip.
 */
public final class ExcelWriter {
    private ExcelWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        write(df, path, ExcelOptions.defaults());
    }

    public static void write(DataFrame df, String path, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        try (OutputStream out = Files.newOutputStream(Path.of(path))) {
            write(df, out, opt);
        }
    }

    public static void write(DataFrame df, OutputStream out, ExcelOptions options) throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        LinkedHashMap<String, DataFrame> sheets = new LinkedHashMap<>();
        sheets.put(opt.writeSheetName() == null ? "Sheet1" : opt.writeSheetName(),
            df == null ? DataFrame.create() : df);
        OoxmlZip.write(out, sheets, opt.header(), opt.freezeHeader(), opt.writeNullToken());
    }

    public static void writeSheets(String path, Map<String, DataFrame> sheets, ExcelOptions options)
            throws Exception {
        ExcelOptions opt = options == null ? ExcelOptions.defaults() : options;
        LinkedHashMap<String, DataFrame> ordered = new LinkedHashMap<>();
        if (sheets == null || sheets.isEmpty()) {
            ordered.put("Sheet1", DataFrame.create());
        } else {
            for (Map.Entry<String, DataFrame> e : sheets.entrySet()) {
                ordered.put(e.getKey(), e.getValue() == null ? DataFrame.create() : e.getValue());
            }
        }
        try (OutputStream out = Files.newOutputStream(Path.of(path))) {
            OoxmlZip.write(out, ordered, opt.header(), opt.freezeHeader(), opt.writeNullToken());
        }
    }
}
