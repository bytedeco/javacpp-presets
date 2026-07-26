package org.bytedeco.pytorch.data.json;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Streaming-capable RFC 8259 JSON parser (pure Java).
 *
 * <p>Features:
 * <ul>
 *   <li>Full Unicode escapes including surrogate pairs</li>
 *   <li>Strict or lenient mode (trailing commas, comments, NaN/Infinity, single quotes)</li>
 *   <li>BOM strip, multi-value / JSON Lines (NDJSON) streaming</li>
 *   <li>Precise line/column error locations</li>
 *   <li>Duplicate-key policy (first / last / error)</li>
 *   <li>Depth and size limits for safety</li>
 * </ul>
 */
public final class JsonParser {
    private final Reader reader;
    private final JsonReadOptions options;
    private final char[] buf = new char[8192];
    private int bufLen = 0;
    private int bufPos = 0;
    private int peek = -2; // -2 = empty, -1 = EOF, else char
    private long offset = 0;
    private long line = 1;
    private long column = 0;
    private boolean eof = false;
    private int depth = 0;

    private JsonParser(Reader reader, JsonReadOptions options) {
        this.reader = reader instanceof BufferedReader ? reader : new BufferedReader(reader, 8192);
        this.options = options == null ? JsonReadOptions.defaults() : options;
    }

    // ---- public entry points ----

    public static JsonValue parse(String text) {
        return parse(text, JsonReadOptions.defaults());
    }

    public static JsonValue parse(String text, JsonReadOptions options) {
        if (text == null) throw new JsonException("null input");
        try {
            return new JsonParser(new StringReader(text), options).parseValueRoot();
        } catch (JsonException e) {
            throw e;
        } catch (IOException e) {
            throw new JsonException("I/O error", e);
        }
    }

    public static JsonValue parse(Reader reader) throws IOException {
        return parse(reader, JsonReadOptions.defaults());
    }

    public static JsonValue parse(Reader reader, JsonReadOptions options) throws IOException {
        return new JsonParser(reader, options).parseValueRoot();
    }

    public static JsonValue parse(InputStream in) throws IOException {
        return parse(in, JsonReadOptions.defaults());
    }

    public static JsonValue parse(InputStream in, JsonReadOptions options) throws IOException {
        JsonReadOptions opt = options == null ? JsonReadOptions.defaults() : options;
        InputStream pin = maybeStripBom(in, opt);
        return parse(new InputStreamReader(pin, opt.charset()), opt);
    }

    public static JsonValue parse(Path path) throws IOException {
        return parse(path, JsonReadOptions.defaults());
    }

    public static JsonValue parse(Path path, JsonReadOptions options) throws IOException {
        try (InputStream in = Files.newInputStream(path)) {
            return parse(in, options);
        }
    }

    public static JsonValue parseFile(String path) throws IOException {
        return parse(Path.of(path), JsonReadOptions.defaults());
    }

    public static JsonValue parseFile(String path, JsonReadOptions options) throws IOException {
        return parse(Path.of(path), options);
    }

    /**
     * Parse JSON Lines / NDJSON: one JSON value per line (blank lines skipped).
     * Returns a JSON array of all values.
     */
    public static JsonValue parseLines(Reader reader) throws IOException {
        return parseLines(reader, JsonReadOptions.defaults());
    }

    public static JsonValue parseLines(Reader reader, JsonReadOptions options) throws IOException {
        List<JsonValue> values = new ArrayList<>();
        parseLines(reader, options, values::add);
        return JsonValue.arrayOf(values);
    }

    public static JsonValue parseLines(InputStream in, JsonReadOptions options) throws IOException {
        JsonReadOptions opt = options == null ? JsonReadOptions.defaults() : options;
        InputStream pin = maybeStripBom(in, opt);
        return parseLines(new InputStreamReader(pin, opt.charset()), opt);
    }

    public static JsonValue parseLines(Path path, JsonReadOptions options) throws IOException {
        try (InputStream in = Files.newInputStream(path)) {
            return parseLines(in, options);
        }
    }

    public static JsonValue parseLinesFile(String path) throws IOException {
        return parseLines(Path.of(path), JsonReadOptions.defaults());
    }

    public static JsonValue parseLinesFile(String path, JsonReadOptions options) throws IOException {
        return parseLines(Path.of(path), options);
    }

    /** Stream JSONL records one-by-one without loading everything. */
    public static void parseLines(Reader reader, JsonReadOptions options, JsonValueConsumer consumer)
            throws IOException {
        JsonReadOptions opt = options == null ? JsonReadOptions.defaults() : options;
        BufferedReader br = reader instanceof BufferedReader
            ? (BufferedReader) reader
            : new BufferedReader(reader, 8192);
        long lineNo = 0;
        int count = 0;
        int max = opt.maxRows();
        StringBuilder carry = null; // for multi-line records when allowMultiLineJsonl
        while (true) {
            String line = br.readLine();
            if (line == null) break;
            lineNo++;
            if (opt.skipRows() > 0 && lineNo <= opt.skipRows()) continue;

            String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                if (opt.skipBlankLines()) continue;
            }
            if (opt.commentPrefix() != null && trimmed.startsWith(opt.commentPrefix())) continue;

            String payload;
            if (opt.allowMultiLineJsonl() && carry != null) {
                carry.append('\n').append(line);
                if (!isBalancedJson(carry.toString())) continue;
                payload = carry.toString();
                carry = null;
            } else if (opt.allowMultiLineJsonl() && !isBalancedJson(line)) {
                carry = new StringBuilder(line);
                continue;
            } else {
                payload = line;
            }

            if (payload.trim().isEmpty()) continue;
            try {
                JsonValue v = parse(payload, opt);
                consumer.accept(v);
                count++;
                if (max >= 0 && count >= max) break;
            } catch (JsonException e) {
                if (opt.strict()) {
                    throw new JsonException(e.getMessage() + " [jsonl physical line " + lineNo + "]",
                        lineNo, e.column(), e.offset(), e);
                }
                if (opt.onError() != null) {
                    opt.onError().accept(lineNo, payload, e);
                }
                // skip bad line in lenient mode
            }
        }
        if (carry != null && carry.length() > 0) {
            if (opt.strict()) {
                throw new JsonException("Unterminated multi-line JSONL record", lineNo, 1, -1);
            }
        }
    }

    @FunctionalInterface
    public interface JsonValueConsumer {
        void accept(JsonValue value) throws IOException;
    }

    @FunctionalInterface
    public interface JsonErrorHandler {
        void accept(long line, String raw, JsonException error);
    }

    // ---- root parse ----

    private JsonValue parseValueRoot() throws IOException {
        skipWsAndComments();
        if (peekChar() < 0) {
            if (options.allowEmpty()) return JsonValue.NULL;
            throw error("Unexpected end of input");
        }
        JsonValue v = parseValue();
        skipWsAndComments();
        if (peekChar() >= 0 && !options.allowTrailingContent()) {
            // allow multiple top-level values only if multiValue
            if (options.allowMultipleValues()) {
                JsonValue arr = JsonValue.array();
                arr.add(v);
                while (peekChar() >= 0) {
                    skipWsAndComments();
                    if (peekChar() < 0) break;
                    arr.add(parseValue());
                    skipWsAndComments();
                }
                return arr;
            }
            throw error("Trailing content after JSON value: '" + (char) peekChar() + "'");
        }
        return v;
    }

    private JsonValue parseValue() throws IOException {
        skipWsAndComments();
        int c = peekChar();
        if (c < 0) throw error("Unexpected end of input");
        switch (c) {
            case '{': return parseObject();
            case '[': return parseArray();
            case '"': return parseString(false);
            case '\'':
                if (options.allowSingleQuotes()) return parseString(true);
                throw error("Single quotes not allowed (enable allowSingleQuotes)");
            case 't': return parseLiteral("true", JsonValue.TRUE);
            case 'f': return parseLiteral("false", JsonValue.FALSE);
            case 'n': return parseLiteral("null", JsonValue.NULL);
            case 'N':
                if (options.allowNanInfinity()) return parseLiteral("NaN", JsonValue.ofNumberLex("NaN"));
                throw error("Unexpected 'N'");
            case 'I':
                if (options.allowNanInfinity()) return parseLiteral("Infinity", JsonValue.ofNumberLex("Infinity"));
                throw error("Unexpected 'I'");
            case '+':
                if (options.allowNanInfinity()) {
                    nextChar();
                    if (matchLiteral("Infinity")) return JsonValue.ofNumberLex("Infinity");
                    throw error("Unexpected '+'");
                }
                throw error("Unexpected '+'");
            case '-':
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                return parseNumber();
            default:
                throw error("Unexpected character: '" + (char) c + "'");
        }
    }

    private JsonValue parseObject() throws IOException {
        expect('{');
        enter();
        JsonValue obj = JsonValue.object();
        skipWsAndComments();
        if (peekChar() == '}') {
            nextChar();
            leave();
            return obj;
        }
        while (true) {
            skipWsAndComments();
            int c = peekChar();
            if (c < 0) throw error("Unterminated object");
            String key;
            if (c == '"') {
                key = parseString(false).asString();
            } else if (c == '\'' && options.allowSingleQuotes()) {
                key = parseString(true).asString();
            } else if (options.allowUnquotedKeys() && isIdentStart(c)) {
                key = parseIdent();
            } else {
                throw error("Expected object key");
            }
            skipWsAndComments();
            expect(':');
            skipWsAndComments();
            JsonValue val = parseValue();
            if (obj.has(key)) {
                switch (options.duplicateKeyPolicy()) {
                    case ERROR:
                        throw error("Duplicate key: " + key);
                    case FIRST:
                        // keep existing
                        break;
                    case LAST:
                    default:
                        obj.put(key, val);
                        break;
                }
            } else {
                obj.put(key, val);
            }
            if (obj.size() > options.maxObjectKeys()) {
                throw error("Object exceeds maxObjectKeys=" + options.maxObjectKeys());
            }
            skipWsAndComments();
            c = peekChar();
            if (c == ',') {
                nextChar();
                skipWsAndComments();
                if (peekChar() == '}' && options.allowTrailingCommas()) {
                    nextChar();
                    leave();
                    return obj;
                }
                continue;
            } else if (c == '}') {
                nextChar();
                leave();
                return obj;
            } else {
                throw error("Expected ',' or '}' in object");
            }
        }
    }

    private JsonValue parseArray() throws IOException {
        expect('[');
        enter();
        JsonValue arr = JsonValue.array();
        skipWsAndComments();
        if (peekChar() == ']') {
            nextChar();
            leave();
            return arr;
        }
        while (true) {
            skipWsAndComments();
            arr.add(parseValue());
            if (arr.size() > options.maxArrayLength()) {
                throw error("Array exceeds maxArrayLength=" + options.maxArrayLength());
            }
            skipWsAndComments();
            int c = peekChar();
            if (c == ',') {
                nextChar();
                skipWsAndComments();
                if (peekChar() == ']' && options.allowTrailingCommas()) {
                    nextChar();
                    leave();
                    return arr;
                }
                continue;
            } else if (c == ']') {
                nextChar();
                leave();
                return arr;
            } else {
                throw error("Expected ',' or ']' in array");
            }
        }
    }

    private JsonValue parseString(boolean singleQuote) throws IOException {
        int quote = nextChar();
        if (singleQuote) {
            if (quote != '\'') throw error("Expected single quote");
        } else {
            if (quote != '"') throw error("Expected double quote");
        }
        StringBuilder sb = new StringBuilder();
        while (true) {
            int c = nextChar();
            if (c < 0) throw error("Unterminated string");
            if (c == quote) break;
            if (c == '\\') {
                int e = nextChar();
                if (e < 0) throw error("Unterminated escape");
                switch (e) {
                    case '"': case '\'': case '\\': case '/': sb.append((char) e); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 'u': {
                        int cp = parseHex4();
                        if (Character.isHighSurrogate((char) cp)) {
                            // expect uXXXX low surrogate
                            if (peekChar() == '\\') {
                                nextChar();
                                if (nextChar() != 'u') throw error("Expected \\u after high surrogate");
                                int low = parseHex4();
                                if (!Character.isLowSurrogate((char) low)) {
                                    throw error("Invalid low surrogate");
                                }
                                sb.appendCodePoint(Character.toCodePoint((char) cp, (char) low));
                            } else {
                                sb.append((char) cp); // unpaired — keep as-is in lenient? strict error
                                if (options.strict()) throw error("Unpaired high surrogate");
                            }
                        } else {
                            sb.append((char) cp);
                        }
                        break;
                    }
                    default:
                        if (options.strict()) throw error("Invalid escape: \\" + (char) e);
                        sb.append((char) e);
                }
            } else if (c < 0x20 && options.strict()) {
                throw error("Unescaped control character U+" + Integer.toHexString(c));
            } else {
                sb.append((char) c);
            }
            if (sb.length() > options.maxStringLength()) {
                throw error("String exceeds maxStringLength=" + options.maxStringLength());
            }
        }
        return JsonValue.of(sb.toString());
    }

    private int parseHex4() throws IOException {
        int v = 0;
        for (int i = 0; i < 4; i++) {
            int c = nextChar();
            if (c < 0) throw error("Incomplete \\u escape");
            int d = Character.digit(c, 16);
            if (d < 0) throw error("Invalid hex in \\u escape");
            v = (v << 4) | d;
        }
        return v;
    }

    private JsonValue parseNumber() throws IOException {
        StringBuilder sb = new StringBuilder();
        int c = peekChar();
        if (c == '-') {
            sb.append((char) nextChar());
            c = peekChar();
        }
        if (c == 'I' && options.allowNanInfinity()) {
            if (matchLiteral("Infinity")) return JsonValue.ofNumberLex(sb.length() > 0 ? "-Infinity" : "Infinity");
        }
        if (c < '0' || c > '9') throw error("Invalid number");
        if (c == '0') {
            sb.append((char) nextChar());
            // no leading zeros
            int n = peekChar();
            if (n >= '0' && n <= '9' && options.strict()) {
                throw error("Leading zeros not allowed");
            }
        } else {
            while (c >= '0' && c <= '9') {
                sb.append((char) nextChar());
                c = peekChar();
            }
        }
        c = peekChar();
        if (c == '.') {
            sb.append((char) nextChar());
            c = peekChar();
            if (c < '0' || c > '9') throw error("Expected digit after decimal point");
            while (c >= '0' && c <= '9') {
                sb.append((char) nextChar());
                c = peekChar();
            }
        }
        c = peekChar();
        if (c == 'e' || c == 'E') {
            sb.append((char) nextChar());
            c = peekChar();
            if (c == '+' || c == '-') {
                sb.append((char) nextChar());
                c = peekChar();
            }
            if (c < '0' || c > '9') throw error("Expected digit in exponent");
            while (c >= '0' && c <= '9') {
                sb.append((char) nextChar());
                c = peekChar();
            }
        }
        String lex = sb.toString();
        if (lex.length() > options.maxStringLength()) {
            throw error("Number lexical form too long");
        }
        // validate by parsing
        try {
            if (lex.indexOf('.') >= 0 || lex.indexOf('e') >= 0 || lex.indexOf('E') >= 0) {
                Double.parseDouble(lex);
            } else {
                // may be big integer
                new java.math.BigInteger(lex);
            }
        } catch (Exception e) {
            throw error("Invalid number: " + lex);
        }
        return JsonValue.ofNumberLex(lex);
    }

    private JsonValue parseLiteral(String lit, JsonValue value) throws IOException {
        for (int i = 0; i < lit.length(); i++) {
            int c = nextChar();
            if (c != lit.charAt(i)) {
                throw error("Expected '" + lit + "'");
            }
        }
        return value;
    }

    private boolean matchLiteral(String lit) throws IOException {
        // assumes first char already matches or will be checked
        for (int i = 0; i < lit.length(); i++) {
            if (peekChar() != lit.charAt(i)) return false;
            nextChar();
        }
        return true;
    }

    private String parseIdent() throws IOException {
        StringBuilder sb = new StringBuilder();
        int c = peekChar();
        if (!isIdentStart(c)) throw error("Expected identifier");
        while (isIdentPart(peekChar())) {
            sb.append((char) nextChar());
        }
        return sb.toString();
    }

    private static boolean isIdentStart(int c) {
        return c == '_' || c == '$' || Character.isLetter(c);
    }

    private static boolean isIdentPart(int c) {
        return c >= 0 && (c == '_' || c == '$' || Character.isLetterOrDigit(c));
    }

    // ---- whitespace / comments ----

    private void skipWsAndComments() throws IOException {
        while (true) {
            int c = peekChar();
            if (c < 0) return;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                nextChar();
                continue;
            }
            if (options.allowComments() && c == '/') {
                nextChar();
                int n = peekChar();
                if (n == '/') {
                    // line comment
                    nextChar();
                    while (true) {
                        int x = nextChar();
                        if (x < 0 || x == '\n') break;
                    }
                    continue;
                } else if (n == '*') {
                    nextChar();
                    while (true) {
                        int x = nextChar();
                        if (x < 0) throw error("Unterminated block comment");
                        if (x == '*' && peekChar() == '/') {
                            nextChar();
                            break;
                        }
                    }
                    continue;
                } else {
                    // push back conceptually — not a comment
                    throw error("Unexpected '/'");
                }
            }
            // JSON5-style # comments
            if (options.allowHashComments() && c == '#') {
                nextChar();
                while (true) {
                    int x = nextChar();
                    if (x < 0 || x == '\n') break;
                }
                continue;
            }
            return;
        }
    }

    // ---- depth ----

    private void enter() {
        depth++;
        if (depth > options.maxDepth()) {
            throw error("Nesting depth exceeds maxDepth=" + options.maxDepth());
        }
    }

    private void leave() {
        depth--;
    }

    // ---- char IO ----

    private int peekChar() throws IOException {
        if (peek != -2) return peek;
        peek = readRaw();
        return peek;
    }

    private int nextChar() throws IOException {
        int c;
        if (peek != -2) {
            c = peek;
            peek = -2;
        } else {
            c = readRaw();
        }
        if (c >= 0) {
            offset++;
            if (c == '\n') {
                line++;
                column = 0;
            } else {
                column++;
            }
        }
        return c;
    }

    private int readRaw() throws IOException {
        if (eof) return -1;
        if (bufPos >= bufLen) {
            bufLen = reader.read(buf);
            bufPos = 0;
            if (bufLen < 0) {
                eof = true;
                return -1;
            }
        }
        return buf[bufPos++];
    }

    private void expect(int expected) throws IOException {
        int c = nextChar();
        if (c != expected) {
            throw error("Expected '" + (char) expected + "' but got "
                + (c < 0 ? "EOF" : ("'" + (char) c + "'")));
        }
    }

    private JsonException error(String msg) {
        return new JsonException(msg, line, column, offset);
    }

    // ---- helpers ----

    private static InputStream maybeStripBom(InputStream in, JsonReadOptions opt) throws IOException {
        if (!opt.stripBom()) return in;
        PushbackInputStream pin = new PushbackInputStream(in, 3);
        byte[] bom = new byte[3];
        int n = pin.read(bom);
        boolean hasBom = n == 3
            && (bom[0] & 0xFF) == 0xEF
            && (bom[1] & 0xFF) == 0xBB
            && (bom[2] & 0xFF) == 0xBF;
        if (n > 0 && !hasBom) {
            pin.unread(bom, 0, n);
        }
        return pin;
    }

    /** Heuristic: braces/brackets balanced and not inside incomplete string. */
    static boolean isBalancedJson(String s) {
        int depth = 0;
        boolean inStr = false;
        boolean escape = false;
        char q = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (inStr) {
                if (escape) { escape = false; continue; }
                if (c == '\\') { escape = true; continue; }
                if (c == q) inStr = false;
                continue;
            }
            if (c == '"' || c == '\'') { inStr = true; q = c; continue; }
            if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') depth--;
        }
        return !inStr && depth == 0 && s.trim().length() > 0;
    }
}
