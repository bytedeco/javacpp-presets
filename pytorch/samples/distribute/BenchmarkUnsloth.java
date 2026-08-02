package distribute;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.unsloth.FastConfig;
import org.bytedeco.pytorch.llm.unsloth.FastLanguageModel;
import org.bytedeco.pytorch.llm.unsloth.UnslothTrainer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.tensor;

public class BenchmarkUnsloth {
    static int passed = 0, failed = 0;

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; System.out.println("  FAIL  " + name); }
    }
    static void section(String t) { System.out.println("\n=== " + t + " ==="); }

    static void d1FromPretrained() throws Exception {
        section("D1 fromPretrained + getPeftModel");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastConfig fc = FastConfig.builder().r(8).loadIn4bit(false).build();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg, fc);
        check("fm not null", fm != null);
        check("fastConfig r=8", fm.fastConfig().r() == 8);
        fm = fm.getPeftModel();
        check("after getPeftModel peftApplied", fm.isPeftApplied());
        check("forTraining initial", !fm.isInferenceMode());
    }

    static void d2QuantState() {
        section("D2 4bit quant state");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().loadIn4bit(true).r(4).build()).getPeftModel();
        check("stats has load_in_4bit", fm.stats().containsKey("load_in_4bit"));
        check("load_in_4bit=true", (Boolean) fm.stats().get("load_in_4bit"));
        FastLanguageModel fm8 = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().loadIn8bit(true).r(4).build()).getPeftModel();
        check("load_in_8bit=true", (Boolean) fm8.stats().get("load_in_8bit"));
    }

    static void d3TrainableRatio() {
        section("D3 Trainable param ratio << total");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(8).build()).getPeftModel();
        long total = fm.totalParameters();
        long train = fm.trainableParameters();
        check("total>0", total > 0);
        check("trainable<=total", train <= total);
        double ratio = total == 0 ? 0 : (double) train / (double) total;
        check("ratio<1 (LoRA subset)", ratio < 1.0);
        check("ratio>0 (LoRA active)", ratio > 0);
    }

    static void d4TrainStep() throws Exception {
        section("D4 trainStep finite");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(4).build()).getPeftModel();
        int[] ids = new int[8];
        int vocab = Math.max(1, cfg.vocabSize());
        for (int i = 0; i < ids.length; i++) ids[i] = i % vocab;
        Tensor input = tensor(ids).reshape(1, 8);
        fm.trainStep(input);
        check("stepCount=1", fm.stepCount() == 1);
        double loss = Double.NaN;
        try { loss = fm.trainStep(input).item_double(); } catch (Exception e) { /* ignore */ }
        check("trainStep finite loss", !Double.isNaN(loss));
    }

    static void d5ForInference() {
        section("D5 forInference merge/unmerge");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(4).build()).getPeftModel();
        fm.forInference();
        check("isInferenceMode=true", fm.isInferenceMode());
        fm.forTraining();
        check("isInferenceMode=false after forTraining", !fm.isInferenceMode());
    }

    static void d6SaveLoad() throws Exception {
        section("D6 save_pretrained");
        Path tmp = Files.createTempDirectory("unsloth_ckpt");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(4).build()).getPeftModel();
        fm.forInference();
        fm.savePretrained(tmp);
        check("adapter file exists", Files.exists(tmp.resolve("adapter.pt")));
        Files.walk(tmp).sorted(Comparator.reverseOrder()).forEach(p -> {
            try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
    }

    static void d7Generate() {
        section("D7 generate");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(4).build()).getPeftModel();
        int vocab = Math.max(1, cfg.vocabSize());
        int[] ids = new int[4];
        for (int i = 0; i < ids.length; i++) ids[i] = i % vocab;
        int[] gen = fm.generate(ids, 8);
        check("generate output len>input", gen.length > ids.length);
    }

    static void d8TrainerBridge() {
        section("D8 UnslothTrainer bridge");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(4).build()).getPeftModel();
        Adam opt = new Adam(fm.model().parameters(), new AdamOptions());
        try (UnslothTrainer ut = UnslothTrainer.create(fm, opt)) {
            check("UnslothTrainer created", ut != null);
            int vocab = Math.max(1, cfg.vocabSize());
            int[] ids = new int[8];
            for (int i = 0; i < ids.length; i++) ids[i] = i % vocab;
            double sl = ut.trainStep(tensor(ids).reshape(1, 8));
            check("trainStep loss finite", !Double.isNaN(sl));
            Map<String, Object> stats = ut.stats();
            check("stats has sft_max_seq_length", stats.containsKey("sft_max_seq_length"));
        }
    }

    static void d9Flags() {
        section("D9 rslora / checkpointing flags");
        FastConfig fc = FastConfig.builder()
                .r(16).useRslora(true)
                .gradientCheckpointing(true)
                .useGradientCheckpointingUnsloth(true)
                .fullFinetuning(false).build();
        check("useRslora=true", fc.useRslora());
        check("gradientCheckpointing=true", fc.gradientCheckpointing());
        check("fullFinetuning=false", !fc.fullFinetuning());
        check("toLoraConfig works", fc.toLoraConfig() != null);
        check("toBnbConfig works", fc.toBnbConfig() != null);
    }

    static void d10StatsMap() {
        section("D10 stats map");
        PretrainedConfig cfg = PretrainedConfig.tinyGpt2();
        FastLanguageModel fm = FastLanguageModel.fromPretrained(cfg,
                FastConfig.builder().r(8).build()).getPeftModel();
        Map<String, Object> stats = fm.stats();
        String[] keys = {"total_params", "trainable_params", "r", "max_seq_length",
                "load_in_4bit", "gradient_checkpointing", "trainable_ratio", "use_rslora"};
        for (String k : keys) check("stats has " + k, stats.containsKey(k));
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Unsloth benchmark ===");
        d1FromPretrained();
        d2QuantState();
        d3TrainableRatio();
        d4TrainStep();
        d5ForInference();
        d6SaveLoad();
        d7Generate();
        d8TrainerBridge();
        d9Flags();
        d10StatsMap();
        done();
    }

    static void done() {
        System.out.println("\n============================================================");
        System.out.println("Unsloth  passed=" + passed + "  failed=" + failed);
        System.out.println("============================================================");
        if (failed > 0) System.exit(1);
        System.out.println("ALL DIMENSIONS GREEN");
    }
}
