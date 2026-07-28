package org.bytedeco.pytorch.dataframe.excel.internal;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Minimal styles.xml: general (0), date (1), datetime (2).
 * Also loads numFmt mapping when reading to detect date cells.
 */
public final class StylesTable {
    public static final int STYLE_GENERAL = 0;
    public static final int STYLE_DATE = 1;
    public static final int STYLE_DATETIME = 2;

    /** cellXfs index → is date format */
    private final Set<Integer> dateStyleIndexes = new HashSet<>();
    private final Map<Integer, String> customNumFmts = new HashMap<>();

    public boolean isDateStyle(int styleIndex) {
        return dateStyleIndexes.contains(styleIndex);
    }

    public static StylesTable forWrite() {
        return new StylesTable();
    }

    public static StylesTable read(InputStream in) throws Exception {
        StylesTable st = new StylesTable();
        if (in == null) return st;
        XMLInputFactory factory = XMLInputFactory.newFactory();
        factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
        factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
        XMLStreamReader r = factory.createXMLStreamReader(in, "UTF-8");
        boolean inNumFmts = false;
        boolean inCellXfs = false;
        int xfIndex = 0;
        try {
            while (r.hasNext()) {
                int ev = r.next();
                if (ev == XMLStreamConstants.START_ELEMENT) {
                    String local = r.getLocalName();
                    if ("numFmts".equals(local)) {
                        inNumFmts = true;
                    } else if (inNumFmts && "numFmt".equals(local)) {
                        int id = intAttr(r, "numFmtId", -1);
                        String code = attr(r, "formatCode");
                        if (id >= 0 && code != null) st.customNumFmts.put(id, code);
                    } else if ("cellXfs".equals(local)) {
                        inCellXfs = true;
                        xfIndex = 0;
                    } else if (inCellXfs && "xf".equals(local)) {
                        int numFmtId = intAttr(r, "numFmtId", 0);
                        String code = st.customNumFmts.get(numFmtId);
                        if (ExcelDateUtil.isDateFormat(numFmtId, code)) {
                            st.dateStyleIndexes.add(xfIndex);
                        }
                        xfIndex++;
                    }
                } else if (ev == XMLStreamConstants.END_ELEMENT) {
                    String local = r.getLocalName();
                    if ("numFmts".equals(local)) inNumFmts = false;
                    else if ("cellXfs".equals(local)) inCellXfs = false;
                }
            }
        } finally {
            r.close();
        }
        return st;
    }

    public void write(OutputStream out) throws Exception {
        Writer w = new OutputStreamWriter(out, StandardCharsets.UTF_8);
        w.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
        w.write("<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">");
        // numFmts: 164=date, 165=datetime (custom ids >= 164)
        w.write("<numFmts count=\"2\">");
        w.write("<numFmt numFmtId=\"164\" formatCode=\"yyyy-mm-dd\"/>");
        w.write("<numFmt numFmtId=\"165\" formatCode=\"yyyy-mm-dd hh:mm:ss\"/>");
        w.write("</numFmts>");
        w.write("<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>");
        w.write("<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>");
        w.write("<borders count=\"1\"><border/></borders>");
        w.write("<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>");
        // cellXfs: 0 general, 1 date, 2 datetime
        w.write("<cellXfs count=\"3\">");
        w.write("<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>");
        w.write("<xf numFmtId=\"164\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"/>");
        w.write("<xf numFmtId=\"165\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyNumberFormat=\"1\"/>");
        w.write("</cellXfs>");
        w.write("</styleSheet>");
        w.flush();
    }

    private static String attr(XMLStreamReader r, String name) {
        String v = r.getAttributeValue(null, name);
        if (v != null) return v;
        // namespace-tolerant
        for (int i = 0; i < r.getAttributeCount(); i++) {
            if (name.equals(r.getAttributeLocalName(i))) return r.getAttributeValue(i);
        }
        return null;
    }

    private static int intAttr(XMLStreamReader r, String name, int def) {
        String v = attr(r, name);
        if (v == null || v.isEmpty()) return def;
        try { return Integer.parseInt(v); } catch (NumberFormatException e) { return def; }
    }
}
