package samples;

import org.bytedeco.pytorch.llm.ragas.EvaluationResult;
import org.bytedeco.pytorch.llm.ragas.Ragas;
import org.bytedeco.pytorch.llm.ragas.dataset.EvaluationDataset;
import org.bytedeco.pytorch.llm.ragas.dataset.MultiTurnSample;
import org.bytedeco.pytorch.llm.ragas.dataset.SingleTurnSample;
import org.bytedeco.pytorch.llm.ragas.llms.HeuristicJudge;
import org.bytedeco.pytorch.llm.ragas.llms.LlmJudge;
import org.bytedeco.pytorch.llm.ragas.metrics.AnswerCorrectness;
import org.bytedeco.pytorch.llm.ragas.metrics.AnswerRelevancy;
import org.bytedeco.pytorch.llm.ragas.metrics.AnswerSimilarity;
import org.bytedeco.pytorch.llm.ragas.metrics.ContextPrecision;
import org.bytedeco.pytorch.llm.ragas.metrics.ContextRecall;
import org.bytedeco.pytorch.llm.ragas.metrics.Faithfulness;
import org.bytedeco.pytorch.llm.ragas.metrics.Metric;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** D1 dataset schema | D2 faithfulness [0,1] | D3 relevancy | D4 precision/recall | D5 correctness/similarity | D6 evaluate aggregate | D7 custom Metric | D8 HeuristicJudge | D9 multi-sample stability | D10 empty-context edge */
public class BenchmarkRagas {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    public static void main(String[] args) {
        System.out.println("=== Ragas benchmark ===");
        d1DatasetSchema();
        d2Faithfulness();
        d3Relevancy();
        d4PrecisionRecall();
        d5CorrectnessSimilarity();
        d6EvaluateAggregate();
        d7CustomMetric();
        d8HeuristicJudge();
        d9MultiSampleStability();
        d10EmptyContext();
        d11BuilderMultiTurnDefaults();
        d12AllMetricsJudge();
        d13Stress();
        done();
    }

    static SingleTurnSample sample1() {
        return SingleTurnSample.of(
                "What is Java?",
                "Java is a programming language.",
                "Java is a language.",
                List.of("Java is a programming language used widely."));
    }

    static SingleTurnSample sample2() {
        return SingleTurnSample.of(
                "What is PyTorch?",
                "PyTorch is a deep learning framework.",
                "PyTorch is a ML framework.",
                List.of("PyTorch provides GPU acceleration.", "Deep learning is its domain."));
    }

    static void d1DatasetSchema() {
        section("D1 EvaluationDataset schema");
        EvaluationDataset ds = EvaluationDataset.of(List.of(sample1(), sample2()));
        check("size=2", ds.size() == 2);
        check("get(0) userInput", ds.get(0).userInput().contains("Java"));
        check("get(1) contexts size>0", ds.get(1).retrievedContexts().size() > 0);
        EvaluationDataset empty = EvaluationDataset.of(List.of());
        check("empty size=0", empty.size() == 0);
        check("of factory", EvaluationDataset.of(List.of(sample1())) != null);
    }

    static void d2Faithfulness() {
        section("D2 faithfulness range [0,1]");
        Faithfulness f = new Faithfulness();
        SingleTurnSample s1 = sample1();
        double sc = f.score(s1);
        check("faithfulness in [0,1]", sc >= 0 && sc <= 1);
        // identical response+context = max
        SingleTurnSample perfect = SingleTurnSample.of("q", "abc def", "abc def", List.of("abc def"));
        double p = f.score(perfect);
        check("perfect faithfulness ~= 1", p >= 0.9);
        // empty response
        SingleTurnSample emptyResp = SingleTurnSample.of("q", "", null, List.of("ctx"));
        double e = f.score(emptyResp);
        check("empty response faithfulness=0", e == 0.0);
    }

    static void d3Relevancy() {
        section("D3 answer_relevancy");
        AnswerRelevancy ar = new AnswerRelevancy();
        double s = ar.score(sample1());
        check("relevancy in [0,1]", s >= 0 && s <= 1);
        // identical question and answer
        SingleTurnSample self = SingleTurnSample.of("java", "java", null, null);
        double rs = ar.score(self);
        check("self relevancy high", rs > 0.0);
    }

    static void d4PrecisionRecall() {
        section("D4 context_precision / recall");
        ContextPrecision cp = new ContextPrecision();
        ContextRecall cr = new ContextRecall();
        double prec = cp.score(sample1());
        double rec = cr.score(sample1());
        check("precision in [0,1]", prec >= 0 && prec <= 1);
        check("recall in [0,1]", rec >= 0 && rec <= 1);
        // empty context
        SingleTurnSample noCtx = SingleTurnSample.of("q", "ans", null, List.of());
        check("empty ctx precision=0", cp.score(noCtx) == 0.0);
        check("empty ctx recall=0", cr.score(noCtx) == 0.0);
    }

    static void d5CorrectnessSimilarity() {
        section("D5 answer_correctness / similarity");
        AnswerCorrectness ac = new AnswerCorrectness();
        AnswerSimilarity as = new AnswerSimilarity();
        double acs = ac.score(sample1());
        double ass = as.score(sample1());
        check("correctness in [0,1]", acs >= 0 && acs <= 1);
        check("similarity in [0,1]", ass >= 0 && ass <= 1);
    }

    static void d6EvaluateAggregate() {
        section("D6 evaluate aggregate");
        EvaluationDataset ds = EvaluationDataset.of(List.of(sample1(), sample2()));
        var result = Ragas.evaluate(ds, List.of(new Faithfulness(), new AnswerRelevancy()));
        check("result has 2 metrics", result.metricNames().size() == 2);
        double f = result.mean("faithfulness");
        check("faithfulness mean in [0,1]", f >= 0 && f <= 1);
        Map<String, Double> m = result.toMap();
        check("toMap has faithfulness", m.containsKey("faithfulness"));
        check("toMap has answer_relevancy", m.containsKey("answer_relevancy"));
        check("numSamples=2", result.numSamples() == 2);
    }

    static void d7CustomMetric() {
        section("D7 custom Metric");
        Metric custom = new Metric() {
            public String name() { return "custom_score"; }
            public double score(SingleTurnSample s, LlmJudge j) { return 0.42; }
        };
        SingleTurnSample s = sample1();
        double sc = custom.score(s);
        check("custom score=0.42", Math.abs(sc - 0.42) < 1e-9);
        check("custom name=custom_score", custom.name().equals("custom_score"));
    }

    static void d8HeuristicJudge() {
        section("D8 HeuristicJudge");
        HeuristicJudge hj = new HeuristicJudge();
        check("HeuristicJudge available", hj.available());
        check("wordF1 self ~= 1", Math.abs(HeuristicJudge.wordF1("hello world", "hello world") - 1.0) < 1e-9);
        check("wordF1 diff < 1", HeuristicJudge.wordF1("hello world", "foo bar") < 1.0);
        check("jaccard self = 1", Math.abs(HeuristicJudge.jaccard("a b", "a b") - 1.0) < 1e-9);
        check("jaccard empty = 0", HeuristicJudge.jaccard("", "a") == 0.0);
        check("extractYesNo yes", hj.extractYesNo("yes").orElse(false));
        check("extractYesNo no", !hj.extractYesNo("no").orElse(true));
        check("embed returns 64-dim", hj.embed("test").length == 64);
    }

    static void d9MultiSampleStability() {
        section("D9 multi-sample stability");
        SingleTurnSample s = sample1();
        Faithfulness f = new Faithfulness();
        double sc1 = f.score(s);
        double sc2 = f.score(s);
        check("deterministic score", Math.abs(sc1 - sc2) < 1e-9);
        // 5 identical samples -> mean should equal single score
        List<SingleTurnSample> five = List.of(s, s, s, s, s);
        EvaluationDataset ds = EvaluationDataset.of(five);
        var result = Ragas.evaluate(ds, List.of(new Faithfulness()));
        double mean = result.mean("faithfulness");
        check("5-identical mean == single", Math.abs(mean - sc1) < 1e-9);
    }

    static void d10EmptyContext() {
        section("D10 empty-context edge");
        SingleTurnSample noCtx = SingleTurnSample.of("What is X?", "X is Y.", "X is Y.", List.of());
        Faithfulness f = new Faithfulness();
        check("faithfulness with no ctx >= 0", f.score(noCtx) >= 0);
        AnswerRelevancy ar = new AnswerRelevancy();
        check("relevancy with no ctx >= 0", ar.score(noCtx) >= 0);
        check("defaults list non-empty", Ragas.defaults().size() > 0);
        check("Ragas version", !Ragas.version().isEmpty());
    }

    static void d11BuilderMultiTurnDefaults() {
        section("D11 Builder / MultiTurn / defaults / EvaluationResult");
        HeuristicJudge hj = new HeuristicJudge();
        Ragas ragas = Ragas.builder().defaultJudge(hj).verbose(false).build();
        EvaluationDataset ds = EvaluationDataset.of(List.of(sample1()));
        EvaluationResult r = ragas.doEvaluate(ds, List.of(new Faithfulness(), new AnswerRelevancy()));
        check("builder doEvaluate metrics=2", r.metricNames().size() == 2);
        check("scores array length=1", r.scores("faithfulness").length == 1);
        check("EvaluationResult toString", r.toString() != null);

        Ragas verbose = Ragas.builder().verbose().defaultJudge(hj).build();
        check("verbose builder", verbose != null);

        // evaluate with explicit judge
        var r2 = Ragas.evaluate(ds, List.of(new AnswerSimilarity()), hj);
        check("evaluate with judge", r2.mean("answer_similarity") >= 0);

        // evaluate defaults
        var r3 = Ragas.evaluate(ds);
        check("evaluate defaults covers all", r3.metricNames().size() == Ragas.defaults().size());

        // SingleTurnSample overloads
        SingleTurnSample s1 = SingleTurnSample.of("q", "a");
        SingleTurnSample s2 = SingleTurnSample.of("q", "a", "ref");
        SingleTurnSample s3 = SingleTurnSample.of("q", "a", "ref", List.of("c1"));
        check("SingleTurn of 2", s1.userInput().equals("q") && s1.response().equals("a"));
        check("SingleTurn of 3", s2.reference() != null);
        check("SingleTurn of 4", s3.retrievedContexts().size() == 1);

        // EvaluationDataset mutable add + samples()
        EvaluationDataset mut = new EvaluationDataset(new ArrayList<>());
        mut.add(s1);
        mut.add(s2);
        check("mutable add size=2", mut.size() == 2);
        check("samples copy", mut.samples().size() == 2);

        // MultiTurnSample
        MultiTurnSample mt = MultiTurnSample.of(List.of(
                new MultiTurnSample.Turn("user", "hi"),
                new MultiTurnSample.Turn("assistant", "hello")
        ));
        check("MultiTurn messages=2", mt.messages().size() == 2);
        check("MultiTurn role", "user".equals(mt.messages().get(0).role()));
        MultiTurnSample mt2 = new MultiTurnSample(List.of(new MultiTurnSample.Turn("system", "x")));
        check("MultiTurn ctor", mt2.messages().size() == 1);

        check("VERSION const", Ragas.VERSION != null && Ragas.VERSION.equals(Ragas.version()));
    }

    static void d12AllMetricsJudge() {
        section("D12 all metrics with explicit HeuristicJudge");
        HeuristicJudge judge = new HeuristicJudge();
        SingleTurnSample s = sample1();
        Metric[] metrics = {
                new Faithfulness(), new AnswerRelevancy(), new ContextPrecision(),
                new ContextRecall(), new AnswerCorrectness(), new AnswerSimilarity()
        };
        for (Metric m : metrics) {
            double sc = m.score(s, judge);
            check(m.name() + " with judge in [0,1]", sc >= 0 && sc <= 1);
            check(m.name() + " name non-empty", m.name() != null && !m.name().isEmpty());
        }
        // LlmJudge defaults
        check("judge available", judge.available());
        check("judge generate non-null", judge.generate("Say yes") != null);
        check("judge embed dim", judge.embed("hello").length > 0);
    }

    static void d13Stress() {
        section("D13 multi-sample stress");
        List<SingleTurnSample> many = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            many.add(SingleTurnSample.of(
                    "Question " + i + " about topic?",
                    "Answer " + i + " with details about topic.",
                    "Reference " + i + " about topic.",
                    List.of("Context " + i + " topic facts.", "More context " + i)));
        }
        EvaluationDataset ds = EvaluationDataset.of(many);
        long t0 = System.nanoTime();
        EvaluationResult r = Ragas.evaluate(ds, Ragas.defaults());
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        check("stress 50 samples", r.numSamples() == 50);
        check("stress all metrics present", r.metricNames().size() == Ragas.defaults().size());
        for (String name : r.metricNames()) {
            double m = r.mean(name);
            check("stress mean " + name + " in [0,1]", m >= 0 && m <= 1);
        }
        System.out.println("  INFO  50 samples x " + r.metricNames().size() + " metrics took " + ms + " ms");
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("Ragas  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
