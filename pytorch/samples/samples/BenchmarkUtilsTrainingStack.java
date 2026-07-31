package samples;

import org.bytedeco.pytorch.llm.accelerate.Accelerator;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeed;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedConfig;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedEngine;
import org.bytedeco.pytorch.llm.nltk.Nltk;
import org.bytedeco.pytorch.llm.ragas.Ragas;
import org.bytedeco.pytorch.llm.ragas.dataset.EvaluationDataset;
import org.bytedeco.pytorch.llm.ragas.dataset.SingleTurnSample;
import org.bytedeco.pytorch.llm.ragas.metrics.Faithfulness;
import org.bytedeco.pytorch.llm.ragas.metrics.AnswerRelevancy;
import org.bytedeco.pytorch.llm.sentence.SentenceTransformer;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.unsloth.FastConfig;
import org.bytedeco.pytorch.llm.unsloth.FastLanguageModel;

import java.util.List;

/** Integration smoke: Accelerate + DeepSpeed + Unsloth + Sentence + NLTK + Ragas */
public class BenchmarkUtilsTrainingStack {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Training Stack integration smoke ===");

        // Accelerate
        try (Accelerator acc = Accelerator.builder().mixedPrecision("fp32").build()) {
            check("Accelerator version", acc.device() != null);
        }

        // DeepSpeed
        DeepSpeedConfig cfg = DeepSpeedConfig.builder().zeroStage(1).gradientClip(1.0).build();
        check("DeepSpeedConfig builder", cfg.zeroStage() == 1);
        try (DeepSpeedEngine eng = DeepSpeed.initialize(
                new org.bytedeco.pytorch.nn.modules.LinearImpl(16, 8), null, cfg)) {
            check("DeepSpeedEngine init", eng != null);
            check("DeepSpeed.memoryStats", eng.memoryStats().containsKey("zero_stage"));
        }

        // Unsloth
        PretrainedConfig pcfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(pcfg,
                FastConfig.builder().r(4).loadIn4bit(false).build()).getPeftModel();
        check("FastLanguageModel init", fm != null);
        check("Unsloth stats map", fm.stats().containsKey("total_params"));

        // SentenceTransformer
        SentenceTransformer st = SentenceTransformer.mini(512, 64);
        float[] e = st.encode("hello world");
        check("SentenceTransformer encode", e.length == 64);

        // NLTK
        List<String> toks = Nltk.wordTokenize("hello world");
        check("NLTK tokenize", toks.size() == 2);
        check("NLTK bleu", Nltk.sentenceBleu(toks, List.of("hello", "world")) > 0.9);

        // Ragas
        SingleTurnSample ragasSample = SingleTurnSample.of(
                "What is Java?", "Java is a language.", "Java is a language.",
                java.util.List.of("Java is a programming language."));
        var ragasResult = Ragas.evaluate(
                EvaluationDataset.of(java.util.List.of(ragasSample)),
                java.util.List.of(new Faithfulness(), new AnswerRelevancy()));
        check("Ragas evaluate", ragasResult.metricNames().size() == 2);

        System.out.println("\n============================================================");
        System.out.println("Stack  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL GREEN");
    }
}
