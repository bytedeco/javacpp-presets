package org.bytedeco.pytorch.dataframe;

import java.util.*;

/**
 * Lightweight Pandas-style {@code DataFrame.query} / {@code DataFrame.eval}
 * string expression engine.
 *
 * <p>Supported (intentionally small, training-pipeline oriented):
 * <ul>
 *   <li>Comparisons: {@code == != < <= > >=}</li>
 *   <li>Boolean: {@code and / or / not} (also {@code & | ~})</li>
 *   <li>Arithmetic in eval: {@code + - * / %}</li>
 *   <li>Column names as bare identifiers or backtick-quoted {@code `col name`}</li>
 *   <li>Literals: numbers, {@code true/false/null}, single/double quoted strings</li>
 *   <li>Local vars: {@code @name} resolved from the provided local map</li>
 *   <li>Membership: {@code col in [a, b, c]} / {@code col not in [...]}</li>
 * </ul>
 *
 * <p>Not a full Python parser — complex Python expressions should use Expression API.
 */
public final class QueryEval {
    private QueryEval() {}

    /**
     * Filter rows where boolean expression is true (Pandas {@code query}).
     * @param localDict optional {@code @var} substitutions (may be null)
     */
    public static DataFrame query(DataFrame df, String expr, Map<String, Object> localDict) {
        if (expr == null || expr.isBlank()) return df.copy();
        Expression condition = parseBool(expr, localDict);
        return df.filter(condition);
    }

    public static DataFrame query(DataFrame df, String expr) {
        return query(df, expr, null);
    }

    /**
     * Evaluate an expression to a new column (Pandas {@code eval} for assignment-like use).
     * Returns a one-column frame named {@code result} unless {@code outName} is set.
     */
    public static Column evalColumn(DataFrame df, String expr, Map<String, Object> localDict) {
        Expression e = parseExpr(expr, localDict);
        return e.evaluate(df);
    }

    /**
     * Evaluate and attach as a new column (Pandas {@code df.eval("z = x + y")} subset).
     * Supports {@code "newcol = <expr>"} or bare {@code "<expr>"} → column {@code "result"}.
     */
    public static DataFrame eval(DataFrame df, String expr, Map<String, Object> localDict) {
        if (expr == null || expr.isBlank()) return df.copy();
        String s = expr.trim();
        String outName = "result";
        String rhs = s;
        int eq = findAssignment(s);
        if (eq > 0) {
            outName = s.substring(0, eq).trim();
            if (outName.startsWith("`") && outName.endsWith("`")) {
                outName = outName.substring(1, outName.length() - 1);
            }
            rhs = s.substring(eq + 1).trim();
        }
        Expression e = parseExpr(rhs, localDict);
        return df.withColumn(outName, e);
    }

    public static DataFrame eval(DataFrame df, String expr) {
        return eval(df, expr, null);
    }

    // ================================================================
    // Parser (recursive-descent over tokens)
    // ================================================================

    private static int findAssignment(String s) {
        // first '=' not part of == != <= >=
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '=') {
                char prev = i > 0 ? s.charAt(i - 1) : 0;
                char next = i + 1 < s.length() ? s.charAt(i + 1) : 0;
                if (prev != '!' && prev != '<' && prev != '>' && prev != '=' && next != '=') {
                    return i;
                }
            }
        }
        return -1;
    }

    private static Expression parseBool(String expr, Map<String, Object> locals) {
        Tokenizer t = new Tokenizer(expr, locals);
        Expression e = parseOr(t);
        t.expectEnd();
        return e;
    }

    private static Expression parseExpr(String expr, Map<String, Object> locals) {
        Tokenizer t = new Tokenizer(expr, locals);
        Expression e = parseOr(t); // or covers comparison + arith via precedence
        t.expectEnd();
        return e;
    }

    private static Expression parseOr(Tokenizer t) {
        Expression left = parseAnd(t);
        while (t.matchKeyword("or") || t.matchOp("|") || t.matchOp("||")) {
            left = left.or(parseAnd(t));
        }
        return left;
    }

    private static Expression parseAnd(Tokenizer t) {
        Expression left = parseNot(t);
        while (t.matchKeyword("and") || t.matchOp("&") || t.matchOp("&&")) {
            left = left.and(parseNot(t));
        }
        return left;
    }

    private static Expression parseNot(Tokenizer t) {
        if (t.matchKeyword("not") || t.matchOp("~") || t.matchOp("!")) {
            return parseNot(t).not();
        }
        return parseComparison(t);
    }

    private static Expression parseComparison(Tokenizer t) {
        Expression left = parseAdd(t);
        // in / not in
        if (t.matchKeyword("not") && t.lookaheadKeyword("in")) {
            t.matchKeyword("in");
            List<Object> vals = parseList(t);
            return left.isIn(vals.toArray()).not();
        }
        if (t.matchKeyword("in")) {
            List<Object> vals = parseList(t);
            return left.isIn(vals.toArray());
        }
        if (t.matchOp("==") || t.matchOp("=")) return left.eq(parseAdd(t));
        if (t.matchOp("!=")) return left.ne(parseAdd(t));
        if (t.matchOp("<=")) return left.le(parseAdd(t));
        if (t.matchOp(">=")) return left.ge(parseAdd(t));
        if (t.matchOp("<")) return left.lt(parseAdd(t));
        if (t.matchOp(">")) return left.gt(parseAdd(t));
        return left;
    }

    private static Expression parseAdd(Tokenizer t) {
        Expression left = parseMul(t);
        while (true) {
            if (t.matchOp("+")) left = left.plus(parseMul(t));
            else if (t.matchOp("-")) left = left.minus(parseMul(t));
            else break;
        }
        return left;
    }

    private static Expression parseMul(Tokenizer t) {
        Expression left = parseUnary(t);
        while (true) {
            if (t.matchOp("*")) left = left.multiply(parseUnary(t));
            else if (t.matchOp("/")) left = left.divide(parseUnary(t));
            else if (t.matchOp("%")) left = left.mod(parseUnary(t));
            else break;
        }
        return left;
    }

    private static Expression parseUnary(Tokenizer t) {
        if (t.matchOp("-")) return parseUnary(t).neg();
        if (t.matchOp("+")) return parseUnary(t);
        return parsePrimary(t);
    }

    private static Expression parsePrimary(Tokenizer t) {
        if (t.matchOp("(")) {
            Expression e = parseOr(t);
            t.expectOp(")");
            return e;
        }
        Token tok = t.next();
        if (tok == null) throw new IllegalArgumentException("unexpected end of expression");
        return switch (tok.type) {
            case IDENT -> Expression.col(tok.text);
            case NUMBER -> Expression.lit(tok.number);
            case STRING -> Expression.lit(tok.text);
            case BOOL -> Expression.lit(tok.boolVal);
            case NULL -> Expression.lit(null);
            case AT_VAR -> Expression.lit(tok.value);
            default -> throw new IllegalArgumentException("unexpected token: " + tok);
        };
    }

    private static List<Object> parseList(Tokenizer t) {
        t.expectOp("[");
        List<Object> vals = new ArrayList<>();
        if (!t.matchOp("]")) {
            do {
                Token tok = t.next();
                if (tok == null) throw new IllegalArgumentException("unterminated list");
                Object v = switch (tok.type) {
                    case NUMBER -> tok.number;
                    case STRING -> tok.text;
                    case BOOL -> tok.boolVal;
                    case NULL -> null;
                    case AT_VAR -> tok.value;
                    case IDENT -> tok.text; // bare word as string
                    default -> throw new IllegalArgumentException("bad list element: " + tok);
                };
                vals.add(v);
            } while (t.matchOp(","));
            t.expectOp("]");
        }
        return vals;
    }

    // ---- tokenizer ----

    enum TokType { IDENT, NUMBER, STRING, BOOL, NULL, AT_VAR, OP, END }

    static final class Token {
        final TokType type;
        final String text;
        final Number number;
        final Boolean boolVal;
        final Object value;
        Token(TokType type, String text) {
            this(type, text, null, null, null);
        }
        Token(TokType type, String text, Number number, Boolean boolVal, Object value) {
            this.type = type; this.text = text; this.number = number;
            this.boolVal = boolVal; this.value = value;
        }
        @Override public String toString() { return type + ":" + text; }
    }

    static final class Tokenizer {
        private final String s;
        private final Map<String, Object> locals;
        private int i;
        private Token peek;

        Tokenizer(String s, Map<String, Object> locals) {
            this.s = s;
            this.locals = locals == null ? Map.of() : locals;
            this.i = 0;
        }

        Token next() {
            if (peek != null) { Token t = peek; peek = null; return t; }
            return read();
        }

        Token peek() {
            if (peek == null) peek = read();
            return peek;
        }

        boolean matchOp(String op) {
            Token t = peek();
            if (t != null && t.type == TokType.OP && op.equals(t.text)) { next(); return true; }
            return false;
        }

        boolean matchKeyword(String kw) {
            Token t = peek();
            if (t != null && t.type == TokType.IDENT && kw.equalsIgnoreCase(t.text)) {
                next(); return true;
            }
            return false;
        }

        boolean lookaheadKeyword(String kw) {
            Token t = peek();
            return t != null && t.type == TokType.IDENT && kw.equalsIgnoreCase(t.text);
        }

        void expectOp(String op) {
            if (!matchOp(op)) throw new IllegalArgumentException("expected '" + op + "' at pos " + i);
        }

        void expectEnd() {
            Token t = peek();
            if (t != null && t.type != TokType.END) {
                throw new IllegalArgumentException("trailing input: " + t);
            }
        }

        private Token read() {
            skipWs();
            if (i >= s.length()) return new Token(TokType.END, "");
            char c = s.charAt(i);
            // @var
            if (c == '@') {
                i++;
                String name = readIdent();
                if (!locals.containsKey(name)) {
                    throw new IllegalArgumentException("unknown @var: " + name);
                }
                return new Token(TokType.AT_VAR, name, null, null, locals.get(name));
            }
            // backtick ident
            if (c == '`') {
                i++;
                int start = i;
                while (i < s.length() && s.charAt(i) != '`') i++;
                String name = s.substring(start, i);
                if (i < s.length()) i++;
                return new Token(TokType.IDENT, name);
            }
            // string
            if (c == '\'' || c == '"') {
                char q = c; i++;
                StringBuilder sb = new StringBuilder();
                while (i < s.length() && s.charAt(i) != q) {
                    if (s.charAt(i) == '\\' && i + 1 < s.length()) {
                        i++; sb.append(s.charAt(i++));
                    } else sb.append(s.charAt(i++));
                }
                if (i < s.length()) i++;
                return new Token(TokType.STRING, sb.toString());
            }
            // number
            if (Character.isDigit(c) || (c == '.' && i + 1 < s.length() && Character.isDigit(s.charAt(i + 1)))) {
                int start = i;
                while (i < s.length() && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '.' ||
                        s.charAt(i) == 'e' || s.charAt(i) == 'E' || s.charAt(i) == '+' || s.charAt(i) == '-')) {
                    // careful with + - only after e
                    char ch = s.charAt(i);
                    if ((ch == '+' || ch == '-') && i > start && (s.charAt(i - 1) != 'e' && s.charAt(i - 1) != 'E')) break;
                    i++;
                }
                String num = s.substring(start, i);
                Number n;
                if (num.contains(".") || num.contains("e") || num.contains("E")) n = Double.parseDouble(num);
                else {
                    long lv = Long.parseLong(num);
                    n = (lv >= Integer.MIN_VALUE && lv <= Integer.MAX_VALUE) ? (int) lv : lv;
                }
                return new Token(TokType.NUMBER, num, n, null, null);
            }
            // multi-char ops
            if (i + 1 < s.length()) {
                String two = s.substring(i, i + 2);
                if (two.equals("==") || two.equals("!=") || two.equals("<=") || two.equals(">=")
                        || two.equals("&&") || two.equals("||")) {
                    i += 2;
                    return new Token(TokType.OP, two);
                }
            }
            // single ops
            if ("+-*/%<>=!&|()[],~".indexOf(c) >= 0) {
                i++;
                return new Token(TokType.OP, String.valueOf(c));
            }
            // ident / keyword
            if (Character.isLetter(c) || c == '_') {
                String id = readIdent();
                if (id.equalsIgnoreCase("true") || id.equalsIgnoreCase("True"))
                    return new Token(TokType.BOOL, id, null, true, null);
                if (id.equalsIgnoreCase("false") || id.equalsIgnoreCase("False"))
                    return new Token(TokType.BOOL, id, null, false, null);
                if (id.equalsIgnoreCase("null") || id.equalsIgnoreCase("None"))
                    return new Token(TokType.NULL, id);
                return new Token(TokType.IDENT, id);
            }
            throw new IllegalArgumentException("bad char '" + c + "' at " + i);
        }

        private String readIdent() {
            int start = i;
            while (i < s.length()) {
                char ch = s.charAt(i);
                if (Character.isLetterOrDigit(ch) || ch == '_' || ch == '.') i++;
                else break;
            }
            return s.substring(start, i);
        }

        private void skipWs() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }
    }
}
