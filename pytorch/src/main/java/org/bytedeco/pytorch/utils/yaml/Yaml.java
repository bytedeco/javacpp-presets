/*
 * Minimal YAML 1.1 subset for Docker Compose and Kubernetes manifests.
 *
 * No SnakeYAML / Jackson — hand-rolled like utils.json.Json.
 * Supports: block maps, block lists, scalars (quoted/plain), comments, multi-doc (---).
 * Does NOT support: anchors/aliases, complex tags, flow sequences beyond simple [a, b],
 * full folded blocks. Enough for model-service deploy configs.
 */
package org.bytedeco.pytorch.utils.yaml;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Encode / decode a practical YAML subset used by Compose + K8s.
 *
 * <pre>{@code
 * Map&lt;String, Object&gt; doc = Yaml.load("""
 *   apiVersion: apps/v1
 *   kind: Deployment
 *   metadata:
 *     name: ranker
 *   spec:
 *     replicas: 2
 *     """);
 * String out = Yaml.dump(doc);
 * List&lt;Object&gt; all = Yaml.loadAll(multiDocText);
 * }</pre>
 */
public final class Yaml {

    private Yaml() {}

    // =========================================================================
    // Public load / dump
    // =========================================================================

    public static Object load(String text) throws IOException {
        List<Object> docs = loadAll(text);
        if (docs.isEmpty()) return null;
        return docs.get(0);
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> loadMap(String text) throws IOException {
        Object v = load(text);
        if (v == null) return new LinkedHashMap<>();
        if (v instanceof Map<?, ?> m) return (Map<String, Object>) m;
        throw new IOException("expected YAML mapping, got " + v.getClass().getSimpleName());
    }

    public static List<Object> loadAll(String text) throws IOException {
        if (text == null || text.isBlank()) return List.of();
        Parser p = new Parser(text);
        return p.parseDocuments();
    }

    public static Object load(Path path) throws IOException {
        return load(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static Map<String, Object> loadMap(Path path) throws IOException {
        return loadMap(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static List<Object> loadAll(Path path) throws IOException {
        return loadAll(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static String dump(Object value) {
        StringBuilder sb = new StringBuilder(256);
        Dumper d = new Dumper(sb);
        d.dumpDocument(value);
        return sb.toString();
    }

    /** Dump multiple documents separated by {@code ---}. */
    public static String dumpAll(Collection<?> documents) {
        if (documents == null || documents.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(512);
        boolean first = true;
        for (Object doc : documents) {
            if (!first) sb.append("---\n");
            first = false;
            Dumper d = new Dumper(sb);
            d.dumpDocument(doc);
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) != '\n') sb.append('\n');
        }
        return sb.toString();
    }

    public static void dump(Path path, Object value) throws IOException {
        Files.writeString(path, dump(value), StandardCharsets.UTF_8);
    }

    public static void dumpAll(Path path, Collection<?> documents) throws IOException {
        Files.writeString(path, dumpAll(documents), StandardCharsets.UTF_8);
    }

    // =========================================================================
    // Navigation helpers (same spirit as HttpJson)
    // =========================================================================

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object o) {
        if (o == null) return Map.of();
        if (o instanceof Map<?, ?> m) return (Map<String, Object>) m;
        throw new IllegalArgumentException("expected map, got " + o.getClass().getSimpleName());
    }

    @SuppressWarnings("unchecked")
    public static List<Object> asList(Object o) {
        if (o == null) return List.of();
        if (o instanceof List<?> l) return (List<Object>) l;
        throw new IllegalArgumentException("expected list, got " + o.getClass().getSimpleName());
    }

    public static Object dig(Object root, String... path) {
        Object cur = root;
        for (String p : path) {
            if (cur == null) return null;
            if (cur instanceof Map<?, ?> m) cur = m.get(p);
            else return null;
        }
        return cur;
    }

    public static String asString(Object o) {
        return o == null ? null : String.valueOf(o);
    }

    public static int asInt(Object o, int def) {
        if (o instanceof Number n) return n.intValue();
        if (o instanceof String s) {
            try { return Integer.parseInt(s.trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static long asLong(Object o, long def) {
        if (o instanceof Number n) return n.longValue();
        if (o instanceof String s) {
            try { return Long.parseLong(s.trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static boolean asBool(Object o, boolean def) {
        if (o instanceof Boolean b) return b;
        if (o instanceof String s) {
            String t = s.trim().toLowerCase(Locale.ROOT);
            if ("true".equals(t) || "yes".equals(t) || "on".equals(t)) return true;
            if ("false".equals(t) || "no".equals(t) || "off".equals(t)) return false;
        }
        return def;
    }

    public static Map<String, Object> mapOf(Object... kv) {
        if (kv == null || kv.length == 0) return new LinkedHashMap<>();
        if ((kv.length & 1) != 0) throw new IllegalArgumentException("odd kv length");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put(String.valueOf(kv[i]), kv[i + 1]);
        }
        return m;
    }

    // =========================================================================
    // Parser
    // =========================================================================

    private static final class Parser {
        final String[] rawLines;
        int line; // current line index

        Parser(String text) {
            // Normalize newlines; keep original content
            String norm = text.replace("\r\n", "\n").replace('\r', '\n');
            this.rawLines = norm.split("\n", -1);
            this.line = 0;
        }

        List<Object> parseDocuments() throws IOException {
            List<Object> docs = new ArrayList<>();
            skipEmptyAndCommentsAndDocMarkers();
            while (line < rawLines.length) {
                if (isDocEnd(currentRaw())) {
                    line++;
                    skipEmptyAndCommentsAndDocMarkers();
                    continue;
                }
                if (isDocStart(currentRaw())) {
                    line++;
                    skipEmptyAndComments();
                }
                if (line >= rawLines.length) break;
                Object doc = parseBlock(0);
                if (doc != null || !docs.isEmpty() || line < rawLines.length) {
                    // allow null document only if explicit
                    docs.add(doc == null ? new LinkedHashMap<>() : doc);
                }
                skipEmptyAndComments();
                if (line < rawLines.length && (isDocStart(currentRaw()) || isDocEnd(currentRaw()))) {
                    line++;
                    skipEmptyAndCommentsAndDocMarkers();
                } else if (line < rawLines.length) {
                    // next doc without marker — still accept as new if indent 0 key
                    skipEmptyAndCommentsAndDocMarkers();
                }
            }
            return docs;
        }

        /**
         * Parse a block node whose content starts at indent {@code >= minIndent}.
         */
        Object parseBlock(int minIndent) throws IOException {
            skipEmptyAndComments();
            if (line >= rawLines.length) return null;
            Line L = peekLine();
            if (L == null) return null;
            if (L.indent < minIndent) return null;

            if (L.content.startsWith("- ") || L.content.equals("-")) {
                return parseList(L.indent);
            }
            // flow
            if (L.content.startsWith("{") || L.content.startsWith("[")) {
                Object v = parseFlow(L.content);
                line++;
                return v;
            }
            // key: value or key:
            if (looksLikeMapEntry(L.content)) {
                return parseMap(L.indent);
            }
            // bare scalar
            Object v = parseScalar(L.content);
            line++;
            return v;
        }

        Map<String, Object> parseMap(int indent) throws IOException {
            Map<String, Object> map = new LinkedHashMap<>();
            while (line < rawLines.length) {
                skipEmptyAndComments();
                if (line >= rawLines.length) break;
                if (isDocStart(currentRaw()) || isDocEnd(currentRaw())) break;
                Line L = peekLine();
                if (L == null) break;
                if (L.indent < indent) break;
                if (L.indent > indent) {
                    throw new IOException("bad indent at line " + (line + 1) + ": " + L.content);
                }
                if (L.content.startsWith("- ") || L.content.equals("-")) break;
                if (!looksLikeMapEntry(L.content)) break;

                KeyVal kv = splitKeyVal(L.content);
                line++;
                Object value;
                if (kv.valuePart != null) {
                    String vp = kv.valuePart.trim();
                    if (vp.isEmpty()) {
                        value = parseNestedAfterKey(indent);
                    } else if (vp.startsWith("|") || vp.startsWith(">")) {
                        value = parseLiteralBlock(indent, vp.startsWith("|"));
                    } else if (vp.startsWith("{") || vp.startsWith("[")) {
                        value = parseFlow(vp);
                    } else {
                        value = parseScalar(vp);
                    }
                } else {
                    value = parseNestedAfterKey(indent);
                }
                map.put(kv.key, value);
            }
            return map;
        }

        /**
         * After {@code key:} with empty inline value, parse nested map/list.
         * Accepts both indented children and compact sequences at the same indent:
         * <pre>
         * key:
         *   nested: 1
         * key:
         * - item   # same indent as key — common in kubeconfig
         * </pre>
         */
        Object parseNestedAfterKey(int keyIndent) throws IOException {
            skipEmptyAndComments();
            Line next = peekLine();
            if (next == null) return null;
            if (next.indent > keyIndent) {
                return parseBlock(keyIndent + 1);
            }
            // compact sequence: dash at same indent as the parent key
            if (next.indent == keyIndent
                    && (next.content.startsWith("- ") || next.content.equals("-"))) {
                return parseList(keyIndent);
            }
            return null;
        }

        List<Object> parseList(int indent) throws IOException {
            List<Object> list = new ArrayList<>();
            while (line < rawLines.length) {
                skipEmptyAndComments();
                if (line >= rawLines.length) break;
                if (isDocStart(currentRaw()) || isDocEnd(currentRaw())) break;
                Line L = peekLine();
                if (L == null) break;
                if (L.indent < indent) break;
                if (L.indent > indent) {
                    throw new IOException("list item indent mismatch at line " + (line + 1));
                }
                if (!(L.content.startsWith("- ") || L.content.equals("-"))) break;

                String rest = L.content.equals("-") ? "" : L.content.substring(2);
                line++;
                Object item;
                if (rest.isBlank()) {
                    skipEmptyAndComments();
                    Line next = peekLine();
                    if (next != null && next.indent > indent) {
                        item = parseBlock(indent + 1);
                    } else {
                        item = null;
                    }
                } else if (looksLikeMapEntry(rest)) {
                    // inline map start on same line as "- key: val"
                    // Put back a synthetic map parse at indent+2 conceptually:
                    // We parse the rest as first map entry, then continue map at greater indent.
                    KeyVal kv = splitKeyVal(rest);
                    Map<String, Object> m = new LinkedHashMap<>();
                    if (kv.valuePart != null && !kv.valuePart.trim().isEmpty()) {
                        String vp = kv.valuePart.trim();
                        if (vp.startsWith("|") || vp.startsWith(">")) {
                            m.put(kv.key, parseLiteralBlock(indent, vp.startsWith("|")));
                        } else if (vp.startsWith("{") || vp.startsWith("[")) {
                            m.put(kv.key, parseFlow(vp));
                        } else {
                            m.put(kv.key, parseScalar(vp));
                        }
                    } else {
                        skipEmptyAndComments();
                        Line next = peekLine();
                        if (next != null && next.indent > indent) {
                            m.put(kv.key, parseBlock(indent + 1));
                        } else {
                            m.put(kv.key, null);
                        }
                    }
                    // more keys of the same list-item map
                    while (true) {
                        skipEmptyAndComments();
                        Line n = peekLine();
                        if (n == null) break;
                        if (n.indent <= indent) break;
                        if (n.content.startsWith("- ") || n.content.equals("-")) break;
                        if (!looksLikeMapEntry(n.content)) break;
                        // nested map entries at n.indent
                        Map<String, Object> more = parseMap(n.indent);
                        m.putAll(more);
                        break;
                    }
                    item = m;
                } else if (rest.startsWith("{") || rest.startsWith("[")) {
                    item = parseFlow(rest);
                } else if (rest.startsWith("|") || rest.startsWith(">")) {
                    item = parseLiteralBlock(indent, rest.startsWith("|"));
                } else {
                    item = parseScalar(rest);
                }
                list.add(item);
            }
            return list;
        }

        String parseLiteralBlock(int parentIndent, boolean literal) {
            // Consume following lines with indent > parentIndent
            StringBuilder sb = new StringBuilder();
            int blockIndent = -1;
            while (line < rawLines.length) {
                String raw = rawLines[line];
                if (isDocStart(raw) || isDocEnd(raw)) break;
                if (raw.isEmpty()) {
                    sb.append('\n');
                    line++;
                    continue;
                }
                int ind = countIndent(raw);
                String content = raw.substring(ind);
                // comment-only lines inside block still count if indented
                if (ind <= parentIndent && !raw.isBlank()) break;
                if (blockIndent < 0 && !raw.isBlank()) blockIndent = ind;
                if (blockIndent >= 0 && ind >= blockIndent) {
                    String piece = raw.substring(blockIndent);
                    sb.append(piece).append('\n');
                    line++;
                } else if (raw.isBlank()) {
                    sb.append('\n');
                    line++;
                } else {
                    break;
                }
            }
            String s = sb.toString();
            if (!literal) {
                // folded: collapse single newlines to space (simplified)
                s = s.replaceAll("(?<!\n)\n(?!\n)", " ").replaceAll("\n+", "\n").trim() + "\n";
            }
            // chomp strip trailing newlines to single or none — keep one trailing stripped
            while (s.endsWith("\n\n")) s = s.substring(0, s.length() - 1);
            if (s.endsWith("\n")) s = s.substring(0, s.length() - 1);
            return s;
        }

        Object parseFlow(String text) throws IOException {
            FlowParser fp = new FlowParser(text.trim());
            Object v = fp.parseValue();
            fp.skipWs();
            return v;
        }

        static Object parseScalar(String raw) {
            if (raw == null) return null;
            String s = raw.trim();
            if (s.isEmpty()) return "";
            // strip inline comment if unquoted
            if (s.charAt(0) != '"' && s.charAt(0) != '\'') {
                int hash = indexOfUnquotedComment(s);
                if (hash >= 0) s = s.substring(0, hash).trim();
            }
            if (s.isEmpty()) return "";
            char c0 = s.charAt(0);
            if (c0 == '"' && s.endsWith("\"") && s.length() >= 2) {
                return unescapeDouble(s.substring(1, s.length() - 1));
            }
            if (c0 == '\'' && s.endsWith("'") && s.length() >= 2) {
                return s.substring(1, s.length() - 1).replace("''", "'");
            }
            String lower = s.toLowerCase(Locale.ROOT);
            if ("null".equals(lower) || "~".equals(s) || "null".equals(s)) return null;
            if ("true".equals(lower) || "yes".equals(lower) || "on".equals(lower)) return Boolean.TRUE;
            if ("false".equals(lower) || "no".equals(lower) || "off".equals(lower)) return Boolean.FALSE;
            // number
            if (isNumber(s)) {
                try {
                    if (s.contains(".") || s.contains("e") || s.contains("E")) {
                        return Double.parseDouble(s);
                    }
                    long lv = Long.parseLong(s);
                    if (lv >= Integer.MIN_VALUE && lv <= Integer.MAX_VALUE) return (int) lv;
                    return lv;
                } catch (NumberFormatException ignored) {
                }
            }
            return s;
        }

        static boolean isNumber(String s) {
            if (s == null || s.isEmpty()) return false;
            int i = 0;
            if (s.charAt(0) == '-' || s.charAt(0) == '+') i++;
            if (i >= s.length()) return false;
            boolean digit = false, dot = false, exp = false;
            for (; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c >= '0' && c <= '9') digit = true;
                else if (c == '.' && !dot && !exp) dot = true;
                else if ((c == 'e' || c == 'E') && digit && !exp) {
                    exp = true;
                    digit = false;
                    if (i + 1 < s.length() && (s.charAt(i + 1) == '+' || s.charAt(i + 1) == '-')) i++;
                } else return false;
            }
            return digit;
        }

        static int indexOfUnquotedComment(String s) {
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '#' && (i == 0 || s.charAt(i - 1) == ' ' || s.charAt(i - 1) == '\t')) return i;
            }
            return -1;
        }

        static String unescapeDouble(String s) {
            StringBuilder sb = new StringBuilder(s.length());
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '\\' && i + 1 < s.length()) {
                    char n = s.charAt(++i);
                    switch (n) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        case '\\' -> sb.append('\\');
                        case '"' -> sb.append('"');
                        case '0' -> sb.append('\0');
                        default -> sb.append(n);
                    }
                } else sb.append(c);
            }
            return sb.toString();
        }

        static boolean looksLikeMapEntry(String content) {
            if (content == null || content.isEmpty()) return false;
            if (content.startsWith("- ") || content.equals("-")) return false;
            // find first unquoted ':'
            boolean inSingle = false, inDouble = false;
            for (int i = 0; i < content.length(); i++) {
                char c = content.charAt(i);
                if (c == '\'' && !inDouble) inSingle = !inSingle;
                else if (c == '"' && !inSingle) inDouble = !inDouble;
                else if (c == ':' && !inSingle && !inDouble) {
                    // must not be part of URL-only midword without space — allow "key:" or "key: value"
                    if (i == 0) return false;
                    return true;
                }
            }
            return false;
        }

        static final class KeyVal {
            final String key;
            final String valuePart; // null means key only with bare ':'
            KeyVal(String key, String valuePart) {
                this.key = key;
                this.valuePart = valuePart;
            }
        }

        static KeyVal splitKeyVal(String content) throws IOException {
            boolean inSingle = false, inDouble = false;
            int colon = -1;
            for (int i = 0; i < content.length(); i++) {
                char c = content.charAt(i);
                if (c == '\'' && !inDouble) inSingle = !inSingle;
                else if (c == '"' && !inSingle) inDouble = !inDouble;
                else if (c == ':' && !inSingle && !inDouble) {
                    colon = i;
                    break;
                }
            }
            if (colon < 0) throw new IOException("expected key: value in: " + content);
            String keyRaw = content.substring(0, colon).trim();
            String valRaw = content.substring(colon + 1);
            // if valRaw is only spaces → empty value part meaning nested
            String key;
            if ((keyRaw.startsWith("\"") && keyRaw.endsWith("\""))
                    || (keyRaw.startsWith("'") && keyRaw.endsWith("'"))) {
                Object k = parseScalar(keyRaw);
                key = k == null ? "null" : String.valueOf(k);
            } else {
                key = keyRaw;
            }
            if (valRaw.isEmpty()) return new KeyVal(key, null);
            // keep valuePart even if blank-ish for nested detection
            return new KeyVal(key, valRaw);
        }

        void skipEmptyAndComments() {
            while (line < rawLines.length) {
                String raw = rawLines[line];
                if (raw.isBlank()) { line++; continue; }
                int ind = countIndent(raw);
                String c = raw.substring(ind);
                if (c.startsWith("#")) { line++; continue; }
                break;
            }
        }

        void skipEmptyAndCommentsAndDocMarkers() {
            while (line < rawLines.length) {
                String raw = rawLines[line];
                if (raw.isBlank()) { line++; continue; }
                int ind = countIndent(raw);
                String c = raw.substring(ind);
                if (c.startsWith("#")) { line++; continue; }
                if (isDocStart(raw) || isDocEnd(raw)) { line++; continue; }
                break;
            }
        }

        String currentRaw() {
            return line < rawLines.length ? rawLines[line] : "";
        }

        Line peekLine() {
            if (line >= rawLines.length) return null;
            String raw = rawLines[line];
            if (raw.isBlank()) return null;
            int ind = countIndent(raw);
            String content = raw.substring(ind);
            if (content.startsWith("#")) return null;
            // strip trailing comment for structure detection only when not quoted start
            return new Line(ind, content, raw);
        }

        static int countIndent(String raw) {
            int i = 0;
            while (i < raw.length()) {
                char c = raw.charAt(i);
                if (c == ' ') i++;
                else if (c == '\t') i += 2; // treat tab as 2
                else break;
            }
            return i;
        }

        static boolean isDocStart(String raw) {
            String t = raw.trim();
            return t.equals("---") || t.startsWith("--- ") || t.startsWith("---\t");
        }

        static boolean isDocEnd(String raw) {
            String t = raw.trim();
            return t.equals("...") || t.startsWith("... ") || t.startsWith("...\t");
        }

        static final class Line {
            final int indent;
            final String content;
            final String raw;
            Line(int indent, String content, String raw) {
                this.indent = indent;
                this.content = content;
                this.raw = raw;
            }
        }
    }

    // ---- flow style [a, b] {k: v} ----
    private static final class FlowParser {
        final String s;
        int i;

        FlowParser(String s) { this.s = s; }

        void skipWs() {
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++;
                else break;
            }
        }

        char peek() throws IOException {
            skipWs();
            if (i >= s.length()) throw new IOException("unexpected end of flow YAML");
            return s.charAt(i);
        }

        char next() throws IOException {
            skipWs();
            if (i >= s.length()) throw new IOException("unexpected end of flow YAML");
            return s.charAt(i++);
        }

        Object parseValue() throws IOException {
            char c = peek();
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"' || c == '\'') return parseQuoted();
            return parsePlain();
        }

        Map<String, Object> parseObject() throws IOException {
            next(); // {
            Map<String, Object> map = new LinkedHashMap<>();
            skipWs();
            if (peek() == '}') { next(); return map; }
            while (true) {
                Object keyObj = parseValue();
                String key = keyObj == null ? "null" : String.valueOf(keyObj);
                if (next() != ':') throw new IOException("expected ':' in flow map");
                Object val = parseValue();
                map.put(key, val);
                skipWs();
                char c = peek();
                if (c == ',') { next(); continue; }
                if (c == '}') { next(); break; }
                throw new IOException("expected ',' or '}' in flow map");
            }
            return map;
        }

        List<Object> parseArray() throws IOException {
            next(); // [
            List<Object> list = new ArrayList<>();
            skipWs();
            if (peek() == ']') { next(); return list; }
            while (true) {
                list.add(parseValue());
                skipWs();
                char c = peek();
                if (c == ',') { next(); continue; }
                if (c == ']') { next(); break; }
                throw new IOException("expected ',' or ']' in flow list");
            }
            return list;
        }

        Object parseQuoted() throws IOException {
            char q = next();
            StringBuilder sb = new StringBuilder();
            while (i < s.length()) {
                char c = s.charAt(i++);
                if (q == '"' && c == '\\' && i < s.length()) {
                    char n = s.charAt(i++);
                    switch (n) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        default -> sb.append(n);
                    }
                } else if (q == '\'' && c == '\'' && i < s.length() && s.charAt(i) == '\'') {
                    sb.append('\'');
                    i++;
                } else if (c == q) {
                    break;
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        Object parsePlain() {
            skipWs();
            int start = i;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ',' || c == ']' || c == '}' || c == ':' || c == '#' ) break;
                if (c == ' ' || c == '\t') {
                    // look ahead — stop plain at trailing spaces before delimiter
                    int j = i;
                    while (j < s.length() && (s.charAt(j) == ' ' || s.charAt(j) == '\t')) j++;
                    if (j >= s.length()) break;
                    char n = s.charAt(j);
                    if (n == ',' || n == ']' || n == '}' || n == ':' || n == '#') break;
                }
                i++;
            }
            String raw = s.substring(start, i).trim();
            return Parser.parseScalar(raw);
        }
    }

    // =========================================================================
    // Dumper
    // =========================================================================

    private static final class Dumper {
        final StringBuilder sb;

        Dumper(StringBuilder sb) { this.sb = sb; }

        void dumpDocument(Object value) {
            if (value == null) {
                sb.append("null\n");
                return;
            }
            dumpNode(value, 0, true);
            if (sb.length() == 0 || sb.charAt(sb.length() - 1) != '\n') sb.append('\n');
        }

        void dumpNode(Object value, int indent, boolean atLineStart) {
            if (value == null) {
                if (atLineStart) indentWrite(indent);
                sb.append("null");
                return;
            }
            if (value instanceof Map<?, ?> map) {
                if (map.isEmpty()) {
                    if (atLineStart) indentWrite(indent);
                    sb.append("{}");
                    return;
                }
                boolean first = true;
                for (Map.Entry<?, ?> e : map.entrySet()) {
                    if (!first) {
                        sb.append('\n');
                        indentWrite(indent);
                    } else if (atLineStart) {
                        // nested block after "key:\n" — must emit indent even for first entry
                        indentWrite(indent);
                    } else {
                        // shouldn't normally happen for block maps
                        sb.append('\n');
                        indentWrite(indent);
                    }
                    first = false;
                    String key = String.valueOf(e.getKey());
                    sb.append(formatKey(key)).append(':');
                    writeMapValue(e.getValue(), indent);
                }
                return;
            }
            if (value instanceof Collection<?> col) {
                if (col.isEmpty()) {
                    if (atLineStart) indentWrite(indent);
                    sb.append("[]");
                    return;
                }
                boolean first = true;
                for (Object item : col) {
                    if (!first) {
                        sb.append('\n');
                        indentWrite(indent);
                    } else if (atLineStart) {
                        indentWrite(indent);
                    } else {
                        sb.append('\n');
                        indentWrite(indent);
                    }
                    first = false;
                    sb.append('-');
                    if (item == null) {
                        sb.append(' ').append("null");
                    } else if (isScalar(item)) {
                        sb.append(' ').append(formatScalar(item));
                    } else if (item instanceof Map<?, ?> m) {
                        if (m.isEmpty()) {
                            sb.append(' ').append("{}");
                        } else {
                            boolean fk = true;
                            for (Map.Entry<?, ?> e : m.entrySet()) {
                                if (fk) {
                                    sb.append(' ').append(formatKey(String.valueOf(e.getKey()))).append(':');
                                    writeMapValue(e.getValue(), indent + 2);
                                    fk = false;
                                } else {
                                    sb.append('\n');
                                    indentWrite(indent + 2);
                                    sb.append(formatKey(String.valueOf(e.getKey()))).append(':');
                                    writeMapValue(e.getValue(), indent + 2);
                                }
                            }
                        }
                    } else if (item instanceof Collection<?>) {
                        sb.append('\n');
                        dumpNode(item, indent + 2, true);
                    } else {
                        sb.append(' ').append(formatScalar(item));
                    }
                }
                return;
            }
            // scalar at block position
            if (atLineStart) indentWrite(indent);
            sb.append(formatScalar(value));
        }

        /** Write value after {@code key:} — scalar inline, nested block on next line. */
        void writeMapValue(Object v, int keyIndent) {
            if (v == null) {
                sb.append(' ').append("null");
            } else if (isScalar(v)) {
                sb.append(' ').append(formatScalar(v));
            } else if (v instanceof Map<?, ?> m2 && m2.isEmpty()) {
                sb.append(' ').append("{}");
            } else if (v instanceof Collection<?> c2 && c2.isEmpty()) {
                sb.append(' ').append("[]");
            } else {
                sb.append('\n');
                dumpNode(v, keyIndent + 2, true);
            }
        }

        void indentWrite(int n) {
            for (int i = 0; i < n; i++) sb.append(' ');
        }

        static boolean isScalar(Object v) {
            return v instanceof String || v instanceof Number || v instanceof Boolean
                    || v instanceof Character;
        }

        static String formatKey(String key) {
            if (key == null) return "\"null\"";
            if (needsQuoting(key) || key.contains(": ") || key.contains("#")
                    || key.isEmpty() || looksLikeNumber(key) || isBooleanish(key)) {
                return quote(key);
            }
            return key;
        }

        static String formatScalar(Object v) {
            if (v == null) return "null";
            if (v instanceof Boolean b) return b ? "true" : "false";
            if (v instanceof Number n) {
                double d = n.doubleValue();
                if (Double.isNaN(d) || Double.isInfinite(d)) return "null";
                if (n instanceof Float || n instanceof Double) return Double.toString(d);
                return n.toString();
            }
            String s = String.valueOf(v);
            if (s.indexOf('\n') >= 0) {
                // literal block
                StringBuilder b = new StringBuilder("|\n");
                String[] lines = s.split("\n", -1);
                for (String line : lines) {
                    b.append("  ").append(line).append('\n');
                }
                // caller expects inline use mostly — for multi-line return quoted
                return quote(s);
            }
            if (needsQuoting(s) || s.isEmpty() || looksLikeNumber(s) || isBooleanish(s)
                    || "null".equalsIgnoreCase(s) || "~".equals(s)
                    || s.startsWith("{") || s.startsWith("[")
                    || s.contains(": ") || s.contains("#")) {
                return quote(s);
            }
            return s;
        }

        static boolean isBooleanish(String s) {
            String t = s.toLowerCase(Locale.ROOT);
            return "true".equals(t) || "false".equals(t) || "yes".equals(t) || "no".equals(t)
                    || "on".equals(t) || "off".equals(t);
        }

        static boolean looksLikeNumber(String s) {
            return Parser.isNumber(s);
        }

        static boolean needsQuoting(String s) {
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == ':' || c == '#' || c == '{' || c == '}' || c == '[' || c == ']'
                        || c == ',' || c == '&' || c == '*' || c == '!' || c == '|'
                        || c == '>' || c == '\'' || c == '"' || c == '%' || c == '@'
                        || c == '`' || c == '\t') {
                    return true;
                }
            }
            if (s.startsWith(" ") || s.endsWith(" ")) return true;
            return false;
        }

        static String quote(String s) {
            StringBuilder b = new StringBuilder(s.length() + 8);
            b.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '\\' -> b.append("\\\\");
                    case '"' -> b.append("\\\"");
                    case '\n' -> b.append("\\n");
                    case '\t' -> b.append("\\t");
                    case '\r' -> b.append("\\r");
                    default -> b.append(c);
                }
            }
            b.append('"');
            return b.toString();
        }
    }
}
