package org.bytedeco.pytorch.data.dataframe.ai;

import java.util.Collection;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Global registry of {@link EmbeddingModel} instances keyed by model id
 * (Daft {@code model="bge-small-zh"} / {@code clip-vit-base-patch32"}).
 *
 * <pre>
 *   EmbeddingRegistry.register(ClipStyleEmbeddingModel.open("clip-vit-base-patch32"));
 *   EmbeddingModel m = EmbeddingRegistry.get("clip-vit-base-patch32");
 * </pre>
 *
 * <p>Built-in defaults (hash, real torchvision/torchaudio towers, CLIP-style dual
 * towers) are installed on first use so offline pipelines always resolve a model.
 */
public final class EmbeddingRegistry {
    private static final Map<String, EmbeddingModel> MODELS = new ConcurrentHashMap<>();
    private static final Map<String, Function<ModelSpec, EmbeddingModel>> FACTORIES = new ConcurrentHashMap<>();
    private static volatile boolean builtinsInstalled = false;

    private EmbeddingRegistry() {}

    /** Register (or replace) a live model instance. */
    public static void register(EmbeddingModel model) {
        Objects.requireNonNull(model, "model");
        ensureBuiltins();
        MODELS.put(normalize(model.spec().id()), model);
    }

    /** Internal put without re-entering {@link #ensureBuiltins()}. */
    private static void putBuiltin(EmbeddingModel model) {
        MODELS.put(normalize(model.spec().id()), model);
    }

    /** Best-effort builtin registration — neural towers may fail to init on some platforms. */
    private static void safePutBuiltin(java.util.function.Supplier<EmbeddingModel> supplier, String label) {
        try {
            EmbeddingModel m = supplier.get();
            if (m != null) putBuiltin(m);
        } catch (Throwable t) {
            System.err.println("[EmbeddingRegistry] skip builtin '" + label + "': "
                    + t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    /** Register a factory invoked lazily on {@link #get(String)}. */
    public static void registerFactory(String familyOrId, Function<ModelSpec, EmbeddingModel> factory) {
        Objects.requireNonNull(factory, "factory");
        FACTORIES.put(normalize(familyOrId), factory);
    }

    /** Resolve by model id; creates from factory / builtins if needed. */
    public static EmbeddingModel get(String modelId) {
        ensureBuiltins();
        String key = normalize(modelId == null ? "hash-text" : modelId);
        EmbeddingModel existing = MODELS.get(key);
        if (existing != null) return existing;

        ModelSpec spec = ModelSpec.parse(modelId);
        // try exact factory, then family factory
        Function<ModelSpec, EmbeddingModel> factory = FACTORIES.get(key);
        if (factory == null) factory = FACTORIES.get(normalize(spec.family()));
        if (factory == null) factory = FACTORIES.get("hash");

        EmbeddingModel created = factory.apply(spec);
        EmbeddingModel prev = MODELS.putIfAbsent(key, created);
        return prev != null ? prev : created;
    }

    public static EmbeddingModel get(ModelSpec spec) {
        if (spec == null) return get("hash-text");
        EmbeddingModel m = MODELS.get(normalize(spec.id()));
        if (m != null) return m;
        return get(spec.id());
    }

    public static boolean contains(String modelId) {
        ensureBuiltins();
        return MODELS.containsKey(normalize(modelId));
    }

    public static Set<String> ids() {
        ensureBuiltins();
        return Set.copyOf(MODELS.keySet());
    }

    public static Collection<EmbeddingModel> models() {
        ensureBuiltins();
        return MODELS.values();
    }

    public static void clear() {
        for (EmbeddingModel m : MODELS.values()) {
            try { m.close(); } catch (Exception ignored) {}
        }
        MODELS.clear();
        builtinsInstalled = false;
    }

    private static String normalize(String id) {
        return id == null ? "" : id.trim().toLowerCase(Locale.ROOT);
    }

    private static void ensureBuiltins() {
        if (builtinsInstalled) return;
        synchronized (EmbeddingRegistry.class) {
            if (builtinsInstalled) return;
            // mark early to prevent re-entrancy if a constructor somehow touches the registry
            builtinsInstalled = true;

            // family factories
            registerFactory("hash", HashEmbeddingModel::fromSpec);
            registerFactory("clip", ClipStyleEmbeddingModel::new);
            registerFactory("bge", spec -> new HashEmbeddingModel(
                ModelSpec.of(spec.id(), Modality.TEXT, spec.defaultDim(), "bge", true)));
            registerFactory("wav2vec", TorchAudioEmbeddingModel::fromSpec);
            registerFactory("audio", TorchAudioEmbeddingModel::fromSpec);
            registerFactory("vision", TorchVisionEmbeddingModel::fromSpec);
            registerFactory("video", spec -> {
                // Video uses vision backbone over temporally-pooled frames
                return TorchVisionEmbeddingModel.fromSpec(
                    ModelSpec.of(spec.id(), Modality.VIDEO, spec.defaultDim(), "vision", true));
            });
            registerFactory("custom", HashEmbeddingModel::fromSpec);
            registerFactory("torch", spec -> TorchScriptEmbeddingModel.builder(spec).build());

            // well-known instances (use putBuiltin — never register() here)
            putBuiltin(HashEmbeddingModel.forText(384));
            putBuiltin(HashEmbeddingModel.forImage(384));
            putBuiltin(HashEmbeddingModel.forAudio(384));
            putBuiltin(HashEmbeddingModel.forVideo(384));
            // Real neural vision / audio towers (random-init, eval mode)
            safePutBuiltin(TorchVisionEmbeddingModel::resnet18, "resnet18");
            safePutBuiltin(TorchVisionEmbeddingModel::resnet34, "resnet34");
            safePutBuiltin(TorchVisionEmbeddingModel::mobilenetV2, "mobilenet_v2");
            safePutBuiltin(TorchAudioEmbeddingModel::m5, "m5");
            safePutBuiltin(TorchAudioEmbeddingModel::wav2letter, "wav2letter-lite");
            // CLIP dual-encoder with neural image tower + hash text (aligned dims)
            safePutBuiltin(() -> ClipStyleEmbeddingModel.open("clip-vit-base-patch32")
                    .withImageTower(TorchVisionEmbeddingModel.fromSpec(
                        ModelSpec.of("clip-vit-base-patch32/image", Modality.IMAGE, 512, "vision", true))),
                    "clip-vit-base-patch32+resnet");
            putBuiltin(ClipStyleEmbeddingModel.open("clip-vit-large-patch14"));
            putBuiltin(new HashEmbeddingModel(ModelSpec.BGE_SMALL_ZH));
            putBuiltin(new HashEmbeddingModel(ModelSpec.BGE_BASE_EN));
            // wav2vec2-base → real M5 audio tower projected to 768
            safePutBuiltin(() -> TorchAudioEmbeddingModel.fromSpec(ModelSpec.WAV2VEC2_BASE), "wav2vec2-base");
            safePutBuiltin(() -> TorchVisionEmbeddingModel.fromSpec(ModelSpec.VIDEO_MAE), "videomae-base");
        }
    }
}
