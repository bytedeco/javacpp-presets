/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.vllm.multimodal;

import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;
import org.bytedeco.pytorch.llm.vllm.multimodal.encoders.MediaEncoder;
import org.bytedeco.pytorch.llm.vllm.multimodal.encoders.MediaEncoderRegistry;
import org.bytedeco.pytorch.llm.vllm.multimodal.encoders.VideoEncoder;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Routes mixed text + image/audio/video/embedding parts into a single token-id
 * stream for the vLLM engine.
 *
 * <p>When a {@link MediaEncoderRegistry} is attached:
 * <ul>
 *   <li><b>IMAGE</b> → DINOv2 / CLIP / SmolVLM / Qwen-VL real forward → feature-hash token ids</li>
 *   <li><b>AUDIO / ASR</b> → Whisper / AsrEncoder real forward → feature-hash token ids</li>
 *   <li><b>VIDEO</b> → VideoEncoder (multi-frame sample + image tower) when available</li>
 *   <li><b>OCR</b> → OcrEncoder via {@link #encodeOcrFeatures} (image specialization)</li>
 * </ul>
 * Falls back to fixed placeholders if no encoder is loaded.
 *
 * <p>Last encode features are cached for {@link #lastFeatures()} / embedding APIs.
 */
public final class CompositeMultimodalProcessor implements MultimodalProcessor {

    private final FastTokenizer tokenizer;
    private final ChatTemplate chatTemplate;
    private final int imageBudget;
    private final int audioBudget;
    private final int videoBudget;
    private final int placeholderTokenId;
    private final MediaEncoderRegistry encoders;

    /** Last real encode results keyed by modality (IMAGE/AUDIO/VIDEO). */
    private final Map<MediaType, MediaEncoder.EncoderFeatures> lastFeatures = new LinkedHashMap<>();
    private final List<String> encodeLog = new ArrayList<>();

    /**
     * Default budgets tuned for Mac CPU + small KV caches (no real encoders).
     */
    public CompositeMultimodalProcessor(FastTokenizer tokenizer, ChatTemplate chatTemplate) {
        this(tokenizer, chatTemplate, 64, 128, 256, 0, null);
    }

    public CompositeMultimodalProcessor(FastTokenizer tokenizer, ChatTemplate chatTemplate,
                                        MediaEncoderRegistry encoders) {
        this(tokenizer, chatTemplate, 64, 128, 256, 0, encoders);
    }

    public CompositeMultimodalProcessor(FastTokenizer tokenizer, ChatTemplate chatTemplate,
                                        int imageBudget, int audioBudget, int videoBudget,
                                        int placeholderTokenId) {
        this(tokenizer, chatTemplate, imageBudget, audioBudget, videoBudget, placeholderTokenId, null);
    }

    public CompositeMultimodalProcessor(FastTokenizer tokenizer, ChatTemplate chatTemplate,
                                        int imageBudget, int audioBudget, int videoBudget,
                                        int placeholderTokenId,
                                        MediaEncoderRegistry encoders) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.chatTemplate = chatTemplate;
        this.imageBudget = Math.max(1, imageBudget);
        this.audioBudget = Math.max(1, audioBudget);
        this.videoBudget = Math.max(1, videoBudget);
        int vocab = Math.max(1, tokenizer.vocabSize());
        this.placeholderTokenId = placeholderTokenId >= 0 && placeholderTokenId < vocab
                && placeholderTokenId < 151_000 // avoid Qwen special/EOS band
                ? placeholderTokenId
                : Math.min(100, Math.max(1, vocab / 100));
        this.encoders = encoders;
    }

    public static CompositeMultimodalProcessor of(FastTokenizer tok, ChatTemplate ct) {
        return new CompositeMultimodalProcessor(tok, ct);
    }

    /** Build with real encoders discovered under {@code modelsRoot}. */
    public static CompositeMultimodalProcessor withEncoders(FastTokenizer tok, ChatTemplate ct,
                                                            Path modelsRoot) {
        MediaEncoderRegistry reg = MediaEncoderRegistry.loadDefault(modelsRoot);
        reg.printStatus();
        return new CompositeMultimodalProcessor(tok, ct, reg);
    }

    public static CompositeMultimodalProcessor withEncoders(FastTokenizer tok, ChatTemplate ct,
                                                            MediaEncoderRegistry reg) {
        return new CompositeMultimodalProcessor(tok, ct, reg);
    }

    public MediaEncoderRegistry encoders() {
        return encoders;
    }

    public Map<MediaType, MediaEncoder.EncoderFeatures> lastFeatures() {
        return Map.copyOf(lastFeatures);
    }

    public List<String> encodeLog() {
        return List.copyOf(encodeLog);
    }

    public boolean hasRealImageEncoder() {
        return encoders != null && encoders.hasImage();
    }

    public boolean hasRealAudioEncoder() {
        return encoders != null && encoders.hasAudio();
    }

    public boolean hasRealVideoEncoder() {
        return encoders != null && encoders.hasVideo();
    }

    public boolean hasRealOcrEncoder() {
        return encoders != null && encoders.hasOcr();
    }

    public boolean hasRealAsrEncoder() {
        return encoders != null && encoders.hasAsr();
    }

    @Override
    public int[] process(MultimodalPrompt prompt, List<Map<String, String>> messages) {
        Objects.requireNonNull(prompt, "prompt");
        List<Integer> ids = new ArrayList<>();
        lastFeatures.clear();
        encodeLog.clear();

        // Collect user-visible text from the multimodal prompt (questions / captions).
        StringBuilder userText = new StringBuilder();
        for (MediaInput part : prompt.parts()) {
            if (part != null && part.type == MediaType.TEXT && part.text != null && !part.text.isEmpty()) {
                if (userText.length() > 0) userText.append('\n');
                userText.append(part.text);
            }
        }

        // Prefer explicit chat history; otherwise synthesize ChatML so Instruct LMs
        // (Qwen/DeepSeek/…) actually generate assistant content. Without this wrapper
        // the model often emits EOS / blank after media hash tokens.
        List<Map<String, String>> chatMsgs = messages;
        if ((chatMsgs == null || chatMsgs.isEmpty()) && chatTemplate != null) {
            String q = userText.length() > 0 ? userText.toString()
                    : "Describe the provided media briefly.";
            chatMsgs = List.of(Map.of("role", "user", "content", q));
        }

        if (chatMsgs != null && !chatMsgs.isEmpty() && chatTemplate != null) {
            String chat = chatTemplate.apply(chatMsgs, /*addGenerationPrompt=*/true);
            // Encode chat without re-adding specials; template already has im_start/im_end.
            appendEncode(ids, chat, false);
            encodeLog.add("CHAT template applied (" + chatMsgs.size() + " msgs, chars=" + chat.length() + ")");
        }

        int vocab = Math.max(2, tokenizer.vocabSize());
        // Inject media feature-hash tokens BEFORE the assistant turn when possible.
        // If chat template already ends with assistant header, media tokens are still
        // useful as a prefix context block.
        List<Integer> mediaIds = new ArrayList<>();
        for (MediaInput part : prompt.parts()) {
            if (part == null || part.type == null) continue;
            switch (part.type) {
                case TEXT -> { /* text already in chat template content */ }
                case IMAGE -> processImage(mediaIds, part, vocab);
                case AUDIO -> processAudio(mediaIds, part, vocab);
                case VIDEO -> processVideo(mediaIds, part, vocab);
                case EMBEDDING -> encodeLog.add("EMBEDDING part skipped in token stream");
            }
        }

        if (!mediaIds.isEmpty()) {
            // Prefer insert media tokens just before trailing assistant header when present.
            // Fallback: prepend media block so model sees media context first.
            int insertAt = findAssistantHeaderInsertIndex(ids);
            if (insertAt >= 0) {
                ids.addAll(insertAt, mediaIds);
            } else if (chatMsgs != null && !chatMsgs.isEmpty()) {
                // After chat: still append media then re-add a soft assistant cue
                ids.addAll(0, mediaIds);
            } else {
                ids.addAll(mediaIds);
                // No chat path: append raw text parts
                if (userText.length() > 0) {
                    appendEncode(ids, userText.toString(), true);
                }
            }
            encodeLog.add("MEDIA tokens injected n=" + mediaIds.size()
                    + " insertAt=" + insertAt);
        } else if (chatMsgs == null || chatMsgs.isEmpty()) {
            // Pure-text fallback without template
            if (userText.length() > 0) {
                appendEncode(ids, userText.toString(), true);
            }
        }

        if (ids.isEmpty()) {
            appendEncode(ids, "", true);
        }

        int[] out = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) out[i] = ids.get(i);
        return out;
    }

    /**
     * Heuristic: locate start of trailing {@code <|im_start|>assistant} / similar
     * generation header so media tokens can sit inside the user turn.
     * Returns index to insert at, or -1 if not found.
     */
    private int findAssistantHeaderInsertIndex(List<Integer> ids) {
        if (ids == null || ids.size() < 2 || tokenizer == null) return -1;
        try {
            // Encode known header fragments and search from the end
            int[][] needles = new int[][] {
                    safeEncode("<|im_start|>assistant"),
                    safeEncode("<|assistant|>"),
                    safeEncode("assistant\n"),
            };
            for (int[] needle : needles) {
                if (needle == null || needle.length == 0) continue;
                outer:
                for (int i = ids.size() - needle.length; i >= 0; i--) {
                    for (int j = 0; j < needle.length; j++) {
                        if (!ids.get(i + j).equals(needle[j])) continue outer;
                    }
                    return i; // insert media before assistant header
                }
            }
        } catch (Throwable ignored) {}
        return -1;
    }

    private int[] safeEncode(String s) {
        try {
            Encoding enc = tokenizer.encode(s, false);
            return enc == null ? null : enc.ids();
        } catch (Throwable t) {
            return null;
        }
    }

    private void processImage(List<Integer> ids, MediaInput part, int vocab) {
        int budget = estimateTokenBudget(part);
        MediaEncoder enc = encoders != null ? encoders.preferredImage() : null;
        if (enc != null) {
            MediaEncoder.EncoderFeatures feat = enc.encode(part);
            lastFeatures.put(MediaType.IMAGE, feat);
            if (!feat.isEmpty()) {
                int[] toks = feat.toTokenIds(budget, vocab);
                for (int t : toks) ids.add(t);
                encodeLog.add(String.format(Locale.ROOT,
                        "IMAGE real %s dim=%d ms=%.1f tokens=%d",
                        feat.source, feat.dim(), feat.encodeMs, toks.length));
                appendEncode(ids, " [image:encoded] ", true);
                return;
            }
            encodeLog.add("IMAGE encoder returned empty — fallback placeholders");
        } else {
            encodeLog.add("IMAGE no encoder — placeholders");
        }
        appendPlaceholders(ids, budget);
        if (part.path != null) {
            appendEncode(ids, " [image:" + part.path.getFileName() + "] ", true);
        } else if (part.width > 0 && part.height > 0) {
            appendEncode(ids, " [image:" + part.width + "x" + part.height + "] ", true);
        } else {
            appendEncode(ids, " [image] ", true);
        }
    }

    private void processAudio(List<Integer> ids, MediaInput part, int vocab) {
        int budget = estimateTokenBudget(part);
        // Prefer ASR wrapper (Whisper + energy cue) when available
        MediaEncoder enc = encoders != null ? encoders.preferredAsr() : null;
        if (enc == null && encoders != null) enc = encoders.primaryAudio();
        if (enc != null) {
            MediaEncoder.EncoderFeatures feat = enc.encode(part);
            lastFeatures.put(MediaType.AUDIO, feat);
            if (!feat.isEmpty()) {
                int[] toks = feat.toTokenIds(budget, vocab);
                for (int t : toks) ids.add(t);
                encodeLog.add(String.format(Locale.ROOT,
                        "AUDIO real %s dim=%d ms=%.1f tokens=%d",
                        feat.source, feat.dim(), feat.encodeMs, toks.length));
                appendEncode(ids, " [audio:encoded] ", true);
                return;
            }
            encodeLog.add("AUDIO encoder returned empty — fallback placeholders");
        } else {
            encodeLog.add("AUDIO no encoder — placeholders");
        }
        appendPlaceholders(ids, budget);
        if (part.path != null) {
            appendEncode(ids, " [audio:" + part.path.getFileName() + "] ", true);
        } else if (part.durationMs > 0) {
            appendEncode(ids, " [audio:" + part.durationMs + "ms] ", true);
        } else {
            appendEncode(ids, " [audio] ", true);
        }
    }

    private void processVideo(List<Integer> ids, MediaInput part, int vocab) {
        int budget = estimateTokenBudget(part);
        // Prefer dedicated VideoEncoder (multi-frame sample + image tower)
        MediaEncoder enc = encoders != null ? encoders.primaryVideo() : null;
        if (enc == null && encoders != null) enc = encoders.preferredImage();
        if (enc != null) {
            MediaEncoder.EncoderFeatures feat = enc.encode(part);
            lastFeatures.put(MediaType.VIDEO, feat);
            if (!feat.isEmpty()) {
                int[] toks = feat.toTokenIds(budget, vocab);
                for (int t : toks) ids.add(t);
                encodeLog.add(String.format(Locale.ROOT,
                        "VIDEO real %s dim=%d ms=%.1f tokens=%d",
                        feat.source, feat.dim(), feat.encodeMs, toks.length));
                appendEncode(ids, " [video:encoded] ", true);
                return;
            }
            encodeLog.add("VIDEO encoder returned empty — fallback placeholders");
        } else {
            encodeLog.add("VIDEO no encoder — placeholders");
        }
        appendPlaceholders(ids, budget);
        if (part.path != null) {
            appendEncode(ids, " [video:" + part.path.getFileName() + "] ", true);
        } else {
            appendEncode(ids, " [video] ", true);
        }
    }

    /**
     * Encode image features only (does not affect token stream). Useful for
     * embedding / retrieval benches with real DINOv2/CLIP/SmolVLM.
     */
    public MediaEncoder.EncoderFeatures encodeImageFeatures(MediaInput image) {
        MediaEncoder enc = encoders != null ? encoders.preferredImage() : null;
        if (enc == null) return MediaEncoder.EncoderFeatures.empty("none");
        MediaEncoder.EncoderFeatures f = enc.encode(image);
        lastFeatures.put(MediaType.IMAGE, f);
        return f;
    }

    public MediaEncoder.EncoderFeatures encodeAudioFeatures(MediaInput audio) {
        MediaEncoder enc = encoders != null ? encoders.primaryAudio() : null;
        if (enc == null) return MediaEncoder.EncoderFeatures.empty("none");
        MediaEncoder.EncoderFeatures f = enc.encode(audio);
        lastFeatures.put(MediaType.AUDIO, f);
        return f;
    }

    /** Encode video via {@link VideoEncoder}. */
    public MediaEncoder.EncoderFeatures encodeVideoFeatures(MediaInput video) {
        MediaEncoder enc = encoders != null ? encoders.primaryVideo() : null;
        if (enc == null && encoders != null) enc = encoders.preferredImage();
        if (enc == null) return MediaEncoder.EncoderFeatures.empty("none");
        MediaEncoder.EncoderFeatures f = enc.encode(video);
        lastFeatures.put(MediaType.VIDEO, f);
        return f;
    }

    /**
     * OCR-oriented encode (document/UI/text-in-image). Uses {@code ocr} wrapper when present.
     */
    public MediaEncoder.EncoderFeatures encodeOcrFeatures(MediaInput image) {
        MediaEncoder enc = encoders != null ? encoders.preferredOcr() : null;
        if (enc == null) return MediaEncoder.EncoderFeatures.empty("none");
        MediaEncoder.EncoderFeatures f = enc.encode(image);
        lastFeatures.put(MediaType.IMAGE, f);
        encodeLog.add(String.format(Locale.ROOT, "OCR %s dim=%d ms=%.1f",
                f.source, f.dim(), f.encodeMs));
        return f;
    }

    /**
     * ASR-oriented encode (speech). Uses {@code asr} wrapper (Whisper + energy cue) when present.
     */
    public MediaEncoder.EncoderFeatures encodeAsrFeatures(MediaInput audio) {
        MediaEncoder enc = encoders != null ? encoders.preferredAsr() : null;
        if (enc == null) return MediaEncoder.EncoderFeatures.empty("none");
        MediaEncoder.EncoderFeatures f = enc.encode(audio);
        lastFeatures.put(MediaType.AUDIO, f);
        encodeLog.add(String.format(Locale.ROOT, "ASR %s dim=%d ms=%.1f",
                f.source, f.dim(), f.encodeMs));
        return f;
    }

    /** Cosine similarity between two pooled feature vectors. */
    public static double cosine(float[] a, float[] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) return 0;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double d = Math.sqrt(na) * Math.sqrt(nb);
        return d < 1e-12 ? 0 : dot / d;
    }

    @Override
    public int estimateTokenBudget(MediaInput input) {
        if (input == null || input.type == null) return 0;
        return switch (input.type) {
            case TEXT -> 0;
            case IMAGE -> imageBudget;
            case AUDIO -> {
                if (input.durationMs > 0) {
                    yield Math.max(audioBudget / 4, Math.min(audioBudget * 4, input.durationMs / 40));
                }
                yield audioBudget;
            }
            case VIDEO -> {
                if (input.durationMs > 0) {
                    yield Math.max(videoBudget / 4, Math.min(videoBudget * 2, input.durationMs / 20));
                }
                yield videoBudget;
            }
            case EMBEDDING -> 0;
        };
    }

    public static String mediaSummary(MultimodalPrompt prompt) {
        if (prompt == null) return "empty";
        int t = 0, i = 0, a = 0, v = 0, e = 0;
        for (MediaInput p : prompt.parts()) {
            if (p == null || p.type == null) continue;
            switch (p.type) {
                case TEXT -> t++;
                case IMAGE -> i++;
                case AUDIO -> a++;
                case VIDEO -> v++;
                case EMBEDDING -> e++;
            }
        }
        return "text=" + t + " image=" + i + " audio=" + a + " video=" + v + " emb=" + e;
    }

    private void appendEncode(List<Integer> ids, String text, boolean addSpecials) {
        Encoding enc = tokenizer.encode(text == null ? "" : text, addSpecials);
        int[] arr = enc.ids();
        if (arr != null) {
            for (int id : arr) ids.add(id);
        }
    }

    private void appendPlaceholders(List<Integer> ids, int n) {
        for (int i = 0; i < n; i++) ids.add(placeholderTokenId);
    }
}
