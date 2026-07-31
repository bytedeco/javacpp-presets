package distribute;

import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.tokenizers.Tiktoken;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-dimensional benchmark: pure-Java {@link Tiktoken} + transformers
 * {@link AutoTokenizer} / {@link FastTokenizer} interop.
 *
 * <p>Dimensions:
 * <ol>
 *   <li>Python-parity encode/decode across 6 encodings × multilingual/code/emoji</li>
 *   <li>Special-token policy (raise / allow / ordinary)</li>
 *   <li>{@code encoding_for_model} OpenAI model map</li>
 *   <li>{@code AutoTokenizer.fromPretrained("gpt-4o")} short-circuit (no Hub)</li>
 *   <li>FastTokenizer adapter id parity vs pure Tiktoken</li>
 *   <li>Batch + single-token + throughput</li>
 *   <li>Optional HF golden backends (BPE/WordPiece/Unigram/TiktokenBPE) if goldens present</li>
 * </ol>
 *
 * <pre>{@code
 * java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *   -cp "target/classes:target/tokenizer-compile-new:..." \
 *   distribute.BenchmarkTiktokenTransformers
 * }</pre>
 */
public final class BenchmarkTiktokenTransformers {

    static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("══ Tiktoken × Transformers multi-dimensional benchmark ══\n");

        // 1. Built-in encoding matrix vs hard-coded Python refs
        section("1. Core parity (hard refs)");
        Map<String, Map<String, int[]>> refs = new LinkedHashMap<>();
        refs.put("cl100k_base", Map.of(
                "Hello world", new int[]{9906, 1917},
                "日本語", new int[]{9080, 22656, 45918, 252},
                "🎉", new int[]{9468, 236, 231}));
        refs.put("o200k_base", Map.of(
                "Hello world", new int[]{13225, 2375},
                "Hello", new int[]{13225}));
        refs.put("p50k_base", Map.of("Hello world", new int[]{15496, 995}));
        refs.put("r50k_base", Map.of("Hello world", new int[]{15496, 995}));
        refs.put("gpt2", Map.of("Hello world", new int[]{15496, 995}));

        for (var encEntry : refs.entrySet()) {
            Tiktoken enc = Tiktoken.getEncoding(encEntry.getKey());
            for (var c : encEntry.getValue().entrySet()) {
                checkEq(encEntry.getKey() + " " + repr(c.getKey()),
                        c.getValue(), enc.encodeOrdinary(c.getKey()));
                checkEq(encEntry.getKey() + " rt " + repr(c.getKey()),
                        c.getKey(), enc.decode(enc.encodeOrdinary(c.getKey())));
            }
        }

        // 2. Specials policy
        section("2. Specials policy (cl100k_base)");
        Tiktoken cl = Tiktoken.getEncoding(Tiktoken.CL100K_BASE);
        checkEq("EOT id", 100257, cl.eotToken());
        checkEq("EOT allow-all", new int[]{100257}, cl.encode("<|endoftext|>", "all"));
        checkEq("EOT ordinary bytes", new int[]{27, 91, 8862, 728, 428, 91, 29},
                cl.encodeOrdinary("<|endoftext|>"));
        boolean raised = false;
        try {
            cl.encode("<|endoftext|>");
        } catch (IllegalArgumentException e) {
            raised = true;
        }
        check("default encode raises on EOT", raised);

        // 3. Model map + AutoTokenizer short-circuit
        section("3. encoding_for_model + AutoTokenizer");
        checkEq("gpt-4", "cl100k_base", Tiktoken.encodingNameForModel("gpt-4"));
        checkEq("gpt-4o", "o200k_base", Tiktoken.encodingNameForModel("gpt-4o"));
        checkEq("o1", "o200k_base", Tiktoken.encodingNameForModel("o1"));
        checkEq("text-davinci-003", "p50k_base", Tiktoken.encodingNameForModel("text-davinci-003"));

        FastTokenizer atGpt4 = AutoTokenizer.fromPretrained("gpt-4");
        FastTokenizer atGpt4o = AutoTokenizer.fromPretrained("gpt-4o");
        FastTokenizer atCl = AutoTokenizer.cl100kBase();
        FastTokenizer atO2 = AutoTokenizer.o200kBase();
        check("AutoTokenizer gpt-4 vocab", atGpt4.vocabSize() == 100277);
        check("AutoTokenizer gpt-4o vocab", atGpt4o.vocabSize() == 200019);
        checkEq("AutoTokenizer.cl100k Hello",
                new int[]{9906, 1917}, atCl.encode("Hello world", false).ids());
        checkEq("AutoTokenizer.o200k Hello",
                new int[]{13225, 2375}, atO2.encode("Hello world", false).ids());

        // 4. Pure Tiktoken vs FastTokenizer adapter
        section("4. Pure Tiktoken ↔ FastTokenizer adapter");
        String[] texts = {
                "Hello world",
                "你好世界",
                "def foo(x):\n    return x + 1\n",
                "🎉🌍",
                "The quick brown fox jumps over the lazy dog."
        };
        for (String name : List.of("cl100k_base", "o200k_base", "p50k_base", "gpt2")) {
            Tiktoken enc = Tiktoken.getEncoding(name);
            FastTokenizer ft = enc.toFastTokenizer();
            for (String t : texts) {
                int[] pure = enc.encodeOrdinary(t);
                int[] via = ft.encode(t, false).ids();
                checkEq(name + " adapter " + repr(t), pure, via);
            }
        }

        // 5. Batch / single-token / decode_bytes
        section("5. Batch / single-token / decode_bytes");
        List<String> batch = List.of("Hello", " world", "你好");
        List<int[]> ids = cl.encodeOrdinaryBatch(batch);
        List<String> back = cl.decodeBatch(ids);
        checkEq("batch roundtrip", batch, back);
        int hid = cl.encodeSingleToken("Hello");
        checkEq("single Hello", 9906, hid);
        byte[] hb = cl.decodeSingleTokenBytes(hid);
        checkEq("single bytes", "Hello", new String(hb, StandardCharsets.UTF_8));
        byte[] raw = cl.decodeBytes(cl.encodeOrdinary("Hello 世界"));
        checkEq("decode_bytes utf8", "Hello 世界", new String(raw, StandardCharsets.UTF_8));

        // 6. Throughput matrix
        section("6. Throughput matrix");
        String longText = "Hello world! 你好世界 🎉\n".repeat(100);
        System.out.printf(Locale.ROOT, "%-14s %12s %12s%n", "encoding", "chars/s", "tokens/s");
        for (String name : Tiktoken.listEncodingNames()) {
            if ("o200k_harmony".equals(name)) continue; // optional heavy specials
            Tiktoken enc = Tiktoken.getEncoding(name);
            for (int i = 0; i < 3; i++) enc.encodeOrdinary(longText);
            int iters = 30;
            long t0 = System.nanoTime();
            long toks = 0;
            for (int i = 0; i < iters; i++) toks += enc.encodeOrdinary(longText).length;
            double sec = (System.nanoTime() - t0) / 1e9;
            double cps = (longText.length() * (double) iters) / sec;
            double tps = toks / sec;
            System.out.printf(Locale.ROOT, "%-14s %12.0f %12.0f%n", name, cps, tps);
            check(name + " tps>0", tps > 0);
        }

        // 7. Optional HF goldens (transformers multi-backend) if present
        section("7. HF goldens presence (TokenizerBackendBenchmark data)");
        Path gold = Path.of("src/test/resources/tokenizers/goldens");
        String[] hf = {"bert_uncased", "gpt2", "qwen2.5", "t5_small", "glm4_chat", "llama32"};
        for (String g : hf) {
            Path p = gold.resolve(g).resolve("cases.jsonl");
            check("golden " + g, Files.isRegularFile(p));
        }
        // Tiktoken goldens
        for (String g : List.of("tiktoken_cl100k_base", "tiktoken_o200k_base", "tiktoken_gpt2")) {
            check("golden " + g, Files.isRegularFile(gold.resolve(g).resolve("cases_multidim.jsonl"))
                    || Files.isRegularFile(gold.resolve(g).resolve("cases.jsonl")));
        }

        // 8. Cross encode: Tiktoken ids can feed CausalLM-style int[] pipelines
        section("8. Transformers-shaped encode output");
        Encoding enc = cl.encode("Hello world", false);
        check("attentionMask len", enc.attentionMask().length == enc.size());
        check("ids len>0", enc.size() == 2);
        checkEq("typeIds default 0", 0, enc.typeIds()[0]);

        System.out.println();
        System.out.println("passed=" + passed + " failed=" + failed);
        if (failed > 0) {
            System.out.println("RESULT: FAIL");
            System.exit(1);
        }
        System.out.println("RESULT: ALL OK");
    }

    private static void section(String s) {
        System.out.println("── " + s + " ──");
    }

    private static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  OK  " + name);
        } else {
            failed++;
            System.out.println("  FAIL " + name);
        }
    }

    private static void checkEq(String name, Object expected, Object actual) {
        boolean ok;
        if (expected instanceof int[] ea && actual instanceof int[] aa) {
            ok = Arrays.equals(ea, aa);
        } else if (expected instanceof List<?> el && actual instanceof List<?> al) {
            ok = el.equals(al);
        } else {
            ok = Objects.equals(expected, actual);
        }
        if (ok) {
            passed++;
            System.out.println("  OK  " + name);
        } else {
            failed++;
            System.out.println("  FAIL " + name + " expected=" + fmt(expected) + " actual=" + fmt(actual));
        }
    }

    private static String fmt(Object o) {
        if (o instanceof int[] a) return Arrays.toString(a);
        return String.valueOf(o);
    }

    private static String repr(String s) {
        if (s == null) return "null";
        String one = s.replace("\n", "\\n");
        return one.length() > 40 ? one.substring(0, 37) + "..." : one;
    }
}
