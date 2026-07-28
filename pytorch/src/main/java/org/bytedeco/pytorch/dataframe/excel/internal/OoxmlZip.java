package org.bytedeco.pytorch.dataframe.excel.internal;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.excel.ExcelParseException;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Minimal OOXML package open/save for tabular workbooks.
 */
public final class OoxmlZip {
    public static final class SheetMeta {
        public final String name;
        public final String path; // e.g. xl/worksheets/sheet1.xml

        public SheetMeta(String name, String path) {
            this.name = name;
            this.path = path;
        }
    }

    public static final class Package {
        public final SharedStringsTable sst;
        public final StylesTable styles;
        public final List<SheetMeta> sheets;
        public final Map<String, byte[]> parts; // path → bytes

        public Package(SharedStringsTable sst, StylesTable styles,
                       List<SheetMeta> sheets, Map<String, byte[]> parts) {
            this.sst = sst;
            this.styles = styles;
            this.sheets = sheets;
            this.parts = parts;
        }
    }

    private OoxmlZip() {}

    public static Package open(InputStream in) throws Exception {
        Map<String, byte[]> parts = new LinkedHashMap<>();
        try (ZipInputStream zis = new ZipInputStream(in)) {
            ZipEntry e;
            byte[] buf = new byte[8192];
            while ((e = zis.getNextEntry()) != null) {
                if (e.isDirectory()) continue;
                String name = e.getName();
                if (name.startsWith("/")) name = name.substring(1);
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                int n;
                while ((n = zis.read(buf)) >= 0) bos.write(buf, 0, n);
                parts.put(name, bos.toByteArray());
                zis.closeEntry();
            }
        }
        if (parts.isEmpty()) {
            throw new ExcelParseException("Not a valid xlsx package (empty zip)");
        }
        SharedStringsTable sst = parts.containsKey("xl/sharedStrings.xml")
            ? SharedStringsTable.read(new ByteArrayInputStream(parts.get("xl/sharedStrings.xml")))
            : new SharedStringsTable();
        StylesTable styles = parts.containsKey("xl/styles.xml")
            ? StylesTable.read(new ByteArrayInputStream(parts.get("xl/styles.xml")))
            : new StylesTable();
        List<SheetMeta> sheets = parseWorkbook(parts);
        return new Package(sst, styles, sheets, parts);
    }

    private static List<SheetMeta> parseWorkbook(Map<String, byte[]> parts) throws Exception {
        byte[] wb = parts.get("xl/workbook.xml");
        if (wb == null) throw new ExcelParseException("Missing xl/workbook.xml");
        Map<String, String> rels = parseRels(parts.get("xl/_rels/workbook.xml.rels"));

        List<SheetMeta> sheets = new ArrayList<>();
        XMLInputFactory factory = XMLInputFactory.newFactory();
        factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
        factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
        XMLStreamReader r = factory.createXMLStreamReader(new ByteArrayInputStream(wb), "UTF-8");
        try {
            while (r.hasNext()) {
                int ev = r.next();
                if (ev == XMLStreamConstants.START_ELEMENT && "sheet".equals(r.getLocalName())) {
                    String name = attr(r, "name");
                    String rid = attr(r, "id");
                    if (rid == null) {
                        // r:id
                        for (int i = 0; i < r.getAttributeCount(); i++) {
                            if ("id".equals(r.getAttributeLocalName(i))) {
                                rid = r.getAttributeValue(i);
                                break;
                            }
                        }
                    }
                    String target = rid == null ? null : rels.get(rid);
                    if (target == null) {
                        // fallback sheetN by order
                        target = "worksheets/sheet" + (sheets.size() + 1) + ".xml";
                    }
                    if (target.startsWith("/")) target = target.substring(1);
                    String path = target.startsWith("xl/") ? target : "xl/" + target;
                    // handle ../
                    path = path.replace("xl/../", "");
                    sheets.add(new SheetMeta(name == null ? "Sheet" + (sheets.size() + 1) : name, path));
                }
            }
        } finally {
            r.close();
        }
        return sheets;
    }

    private static Map<String, String> parseRels(byte[] relsBytes) throws Exception {
        Map<String, String> map = new LinkedHashMap<>();
        if (relsBytes == null) return map;
        XMLInputFactory factory = XMLInputFactory.newFactory();
        factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
        factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
        XMLStreamReader r = factory.createXMLStreamReader(new ByteArrayInputStream(relsBytes), "UTF-8");
        try {
            while (r.hasNext()) {
                int ev = r.next();
                if (ev == XMLStreamConstants.START_ELEMENT && "Relationship".equals(r.getLocalName())) {
                    String id = attr(r, "Id");
                    String target = attr(r, "Target");
                    if (id != null && target != null) map.put(id, target);
                }
            }
        } finally {
            r.close();
        }
        return map;
    }

    /**
     * Write a multi-sheet workbook. sheets: ordered name → DataFrame.
     */
    public static void write(OutputStream out, LinkedHashMap<String, DataFrame> sheets,
                             boolean header, boolean freezeHeader, String writeNullToken) throws Exception {
        if (sheets == null || sheets.isEmpty()) {
            LinkedHashMap<String, DataFrame> one = new LinkedHashMap<>();
            one.put("Sheet1", DataFrame.create());
            sheets = one;
        }
        SharedStringsTable sst = new SharedStringsTable();
        // Pre-build sheet XML in memory so shared strings are complete
        List<String> names = new ArrayList<>();
        List<byte[]> sheetParts = new ArrayList<>();
        for (Map.Entry<String, DataFrame> e : sheets.entrySet()) {
            names.add(sanitizeSheetName(e.getKey()));
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            SheetWriter.write(bos, e.getValue() == null ? DataFrame.create() : e.getValue(),
                sst, header, freezeHeader, writeNullToken);
            sheetParts.add(bos.toByteArray());
        }

        try (ZipOutputStream zos = new ZipOutputStream(out)) {
            put(zos, "[Content_Types].xml", contentTypes(names.size()));
            put(zos, "_rels/.rels", rootRels());
            put(zos, "xl/workbook.xml", workbookXml(names));
            put(zos, "xl/_rels/workbook.xml.rels", workbookRels(names.size()));
            ByteArrayOutputStream stylesOut = new ByteArrayOutputStream();
            StylesTable.forWrite().write(stylesOut);
            put(zos, "xl/styles.xml", stylesOut.toByteArray());
            ByteArrayOutputStream sstOut = new ByteArrayOutputStream();
            sst.write(sstOut);
            put(zos, "xl/sharedStrings.xml", sstOut.toByteArray());
            for (int i = 0; i < sheetParts.size(); i++) {
                put(zos, "xl/worksheets/sheet" + (i + 1) + ".xml", sheetParts.get(i));
            }
        }
    }

    private static void put(ZipOutputStream zos, String name, byte[] data) throws Exception {
        ZipEntry e = new ZipEntry(name);
        zos.putNextEntry(e);
        zos.write(data);
        zos.closeEntry();
    }

    private static void put(ZipOutputStream zos, String name, String xml) throws Exception {
        put(zos, name, xml.getBytes(StandardCharsets.UTF_8));
    }

    private static String contentTypes(int sheetCount) {
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
        sb.append("<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">");
        sb.append("<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>");
        sb.append("<Default Extension=\"xml\" ContentType=\"application/xml\"/>");
        sb.append("<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>");
        sb.append("<Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>");
        sb.append("<Override PartName=\"/xl/sharedStrings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml\"/>");
        for (int i = 1; i <= sheetCount; i++) {
            sb.append("<Override PartName=\"/xl/worksheets/sheet").append(i)
                .append(".xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>");
        }
        sb.append("</Types>");
        return sb.toString();
    }

    private static String rootRels() {
        return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            + "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
            + "</Relationships>";
    }

    private static String workbookXml(List<String> names) {
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
        sb.append("<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"");
        sb.append(" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">");
        sb.append("<sheets>");
        for (int i = 0; i < names.size(); i++) {
            sb.append("<sheet name=\"").append(SharedStringsTable.xmlEscape(names.get(i)))
                .append("\" sheetId=\"").append(i + 1)
                .append("\" r:id=\"rId").append(i + 1).append("\"/>");
        }
        sb.append("</sheets></workbook>");
        return sb.toString();
    }

    private static String workbookRels(int sheetCount) {
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
        sb.append("<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">");
        for (int i = 1; i <= sheetCount; i++) {
            sb.append("<Relationship Id=\"rId").append(i)
                .append("\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet")
                .append(i).append(".xml\"/>");
        }
        int next = sheetCount + 1;
        sb.append("<Relationship Id=\"rId").append(next)
            .append("\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" Target=\"sharedStrings.xml\"/>");
        next++;
        sb.append("<Relationship Id=\"rId").append(next)
            .append("\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>");
        sb.append("</Relationships>");
        return sb.toString();
    }

    public static String sanitizeSheetName(String name) {
        if (name == null || name.isEmpty()) return "Sheet1";
        String s = name.replaceAll("[\\\\/?*\\[\\]]", "_");
        if (s.length() > 31) s = s.substring(0, 31);
        if (s.isEmpty()) s = "Sheet1";
        return s;
    }

    private static String attr(XMLStreamReader r, String name) {
        String v = r.getAttributeValue(null, name);
        if (v != null) return v;
        for (int i = 0; i < r.getAttributeCount(); i++) {
            if (name.equals(r.getAttributeLocalName(i))) return r.getAttributeValue(i);
        }
        return null;
    }
}
