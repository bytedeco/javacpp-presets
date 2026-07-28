package org.bytedeco.pytorch.dataframe.excel.internal;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal shared-strings table for OOXML xlsx.
 */
public final class SharedStringsTable {
    private final List<String> strings = new ArrayList<>();
    private final Map<String, Integer> index = new HashMap<>();

    public int add(String s) {
        if (s == null) s = "";
        Integer existing = index.get(s);
        if (existing != null) return existing;
        int i = strings.size();
        strings.add(s);
        index.put(s, i);
        return i;
    }

    public String get(int i) {
        if (i < 0 || i >= strings.size()) return "";
        return strings.get(i);
    }

    public int size() { return strings.size(); }

    public static SharedStringsTable read(InputStream in) throws Exception {
        SharedStringsTable sst = new SharedStringsTable();
        if (in == null) return sst;
        XMLInputFactory factory = XMLInputFactory.newFactory();
        factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
        factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
        XMLStreamReader r = factory.createXMLStreamReader(in, "UTF-8");
        StringBuilder text = new StringBuilder();
        boolean inSi = false;
        boolean inT = false;
        try {
            while (r.hasNext()) {
                int ev = r.next();
                if (ev == XMLStreamConstants.START_ELEMENT) {
                    String local = r.getLocalName();
                    if ("si".equals(local)) {
                        inSi = true;
                        text.setLength(0);
                    } else if (inSi && "t".equals(local)) {
                        inT = true;
                    }
                } else if (ev == XMLStreamConstants.CHARACTERS || ev == XMLStreamConstants.CDATA) {
                    if (inT) text.append(r.getText());
                } else if (ev == XMLStreamConstants.END_ELEMENT) {
                    String local = r.getLocalName();
                    if ("t".equals(local)) {
                        inT = false;
                    } else if ("si".equals(local)) {
                        inSi = false;
                        sst.strings.add(text.toString());
                        // read path: do not require reverse index uniqueness for lookup by id
                    }
                }
            }
        } finally {
            r.close();
        }
        // rebuild index for writers that reuse a read table
        for (int i = 0; i < sst.strings.size(); i++) {
            sst.index.putIfAbsent(sst.strings.get(i), i);
        }
        return sst;
    }

    public void write(OutputStream out) throws Exception {
        Writer w = new OutputStreamWriter(out, StandardCharsets.UTF_8);
        w.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>");
        w.write("<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"");
        w.write(Integer.toString(strings.size()));
        w.write("\" uniqueCount=\"");
        w.write(Integer.toString(strings.size()));
        w.write("\">");
        for (String s : strings) {
            w.write("<si><t");
            if (s != null && !s.isEmpty() && (Character.isWhitespace(s.charAt(0))
                    || Character.isWhitespace(s.charAt(s.length() - 1)))) {
                w.write(" xml:space=\"preserve\"");
            }
            w.write('>');
            w.write(xmlEscape(s == null ? "" : s));
            w.write("</t></si>");
        }
        w.write("</sst>");
        w.flush();
    }

    static String xmlEscape(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '&': sb.append("&amp;"); break;
                case '<': sb.append("&lt;"); break;
                case '>': sb.append("&gt;"); break;
                case '"': sb.append("&quot;"); break;
                case '\'': sb.append("&apos;"); break;
                default:
                    if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') {
                        // skip invalid XML 1.0 control chars
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.toString();
    }
}
