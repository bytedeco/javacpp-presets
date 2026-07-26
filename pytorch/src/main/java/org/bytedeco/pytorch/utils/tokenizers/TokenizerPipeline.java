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
package org.bytedeco.pytorch.utils.tokenizers;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.utils.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.utils.tokenizers.models.Model;
import org.bytedeco.pytorch.utils.tokenizers.models.Token;
import org.bytedeco.pytorch.utils.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.utils.tokenizers.pretokenizers.PreToken;
import org.bytedeco.pytorch.utils.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.utils.tokenizers.processors.PostProcessor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * Full HuggingFace tokenizers-rs pipeline:
 * AddedVocabulary → Normalizer → PreTokenizer → Model → PostProcessor → Trunc/Pad
 * and reverse decode via Decoder.
 */
public final class TokenizerPipeline {

    private final Normalizer normalizer;
    private final PreTokenizer preTokenizer;
    private final Model model;
    private final PostProcessor postProcessor;
    private final Decoder decoder;
    private final AddedVocabulary added;
    private final FastTokenizer.Padding padding;
    private final FastTokenizer.Truncation truncation;
    private final String unkToken;
    private final String padToken;
    private final String clsToken;
    private final String sepToken;
    private final String bosToken;
    private final String eosToken;
    private final String maskToken;
    private final int modelMaxLength;
    private final boolean addPrefixSpace;

    /** Combined vocab: model + added tokens (added wins on conflict by id ownership). */
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;

    public TokenizerPipeline(Normalizer normalizer,
                             PreTokenizer preTokenizer,
                             Model model,
                             PostProcessor postProcessor,
                             Decoder decoder,
                             AddedVocabulary added,
                             FastTokenizer.Padding padding,
                             FastTokenizer.Truncation truncation,
                             String unkToken, String padToken, String clsToken,
                             String sepToken, String bosToken, String eosToken,
                             String maskToken,
                             int modelMaxLength,
                             boolean addPrefixSpace) {
        this.normalizer = normalizer == null ? Normalizer.NOP : normalizer;
        this.preTokenizer = preTokenizer == null ? PreTokenizer.NOP : preTokenizer;
        this.model = Objects.requireNonNull(model, "model");
        this.postProcessor = postProcessor == null ? PostProcessor.NOP : postProcessor;
        this.decoder = decoder == null ? Decoder.FUSE : decoder;
        this.added = added == null ? AddedVocabulary.empty() : added;
        this.padding = padding;
        this.truncation = truncation;
        this.unkToken = unkToken;
        this.padToken = padToken;
        this.clsToken = clsToken;
        this.sepToken = sepToken;
        this.bosToken = bosToken;
        this.eosToken = eosToken;
        this.maskToken = maskToken;
        this.modelMaxLength = modelMaxLength;
        this.addPrefixSpace = addPrefixSpace;

        this.vocab = new LinkedHashMap<>(model.getVocab());
        this.idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> e : this.vocab.entrySet()) {
            idToToken.put(e.getValue(), e.getKey());
        }
        for (AddedToken t : this.added.tokens()) {
            this.vocab.put(t.content(), t.id());
            idToToken.put(t.id(), t.content());
        }
    }

    public TokenizerPipeline withSpecials(String unk, String pad, String cls, String sep,
                                          String bos, String eos, String mask, int modelMax) {
        return new TokenizerPipeline(
                normalizer, preTokenizer, model, postProcessor, decoder, added,
                padding, truncation,
                unk, pad, cls, sep, bos, eos, mask,
                modelMax, addPrefixSpace);
    }

    public TokenizerPipeline withPadding(FastTokenizer.Padding padding) {
        return new TokenizerPipeline(
                normalizer, preTokenizer, model, postProcessor, decoder, added,
                padding, truncation,
                unkToken, padToken, clsToken, sepToken, bosToken, eosToken, maskToken,
                modelMaxLength, addPrefixSpace);
    }

    public TokenizerPipeline withTruncation(FastTokenizer.Truncation truncation) {
        return new TokenizerPipeline(
                normalizer, preTokenizer, model, postProcessor, decoder, added,
                padding, truncation,
                unkToken, padToken, clsToken, sepToken, bosToken, eosToken, maskToken,
                modelMaxLength, addPrefixSpace);
    }

    // ---- encode -------------------------------------------------------------

    public Encoding encode(String text, boolean addSpecialTokens) {
        return encodePair(text, null, addSpecialTokens);
    }

    public Encoding encodePair(String textA, String textB, boolean addSpecialTokens) {
        List<Token> tokensA = encodeToTokens(textA);
        List<Token> tokensB = textB == null ? null : encodeToTokens(textB);

        Function<String, Integer> idLookup = this::tokenToIdOrNeg;
        Encoding enc = postProcessor.process(tokensA, tokensB, addSpecialTokens, idLookup);

        if (truncation != null && truncation.maxLength > 0 && enc.size() > truncation.maxLength) {
            enc = enc.truncate(truncation.maxLength, truncation.direction);
        }
        if (padding != null && padding.strategy == FastTokenizer.Padding.Strategy.MAX_LENGTH
                && padding.maxLength > 0) {
            enc = enc.padTo(padding.maxLength, padId(), 0, padding.direction);
        }
        return enc;
    }

    /**
     * Run AddedVocabulary → Normalizer → PreTokenizer → Model for one sequence
     * (no post-processor).
     */
    public List<Token> encodeToTokens(String text) {
        String src = text == null ? "" : text;
        if (addPrefixSpace && !src.isEmpty() && src.charAt(0) != ' ') {
            src = " " + src;
        }

        List<AddedVocabulary.Segment> segments = added.splitForEncode(src, normalizer);
        List<Token> out = new ArrayList<>();
        for (AddedVocabulary.Segment seg : segments) {
            if (seg.added) {
                out.add(new Token(seg.addedId, seg.value, seg.start, seg.end, true));
                continue;
            }
            // ordinary: pretokenize then model
            List<PreToken> pretokens = preTokenizer.preTokenize(seg.value);
            // shift offsets to segment start
            List<PreToken> shifted = new ArrayList<>(pretokens.size());
            for (PreToken p : pretokens) {
                if (p.added()) {
                    shifted.add(p);
                } else {
                    shifted.add(new PreToken(p.value(),
                            seg.start + p.start(),
                            seg.start + p.end(),
                            false, -1));
                }
            }
            out.addAll(model.tokenize(shifted));
        }
        return out;
    }

    // ---- decode -------------------------------------------------------------

    public String decode(int[] ids, boolean skipSpecialTokens) {
        if (ids == null || ids.length == 0) return "";
        List<String> pieces = new ArrayList<>(ids.length);
        for (int id : ids) {
            if (skipSpecialTokens && added.isSpecialId(id)) continue;
            // also skip configured specials by content
            String tok = idToToken.get(id);
            if (tok == null) continue;
            if (skipSpecialTokens && isConfiguredSpecial(tok)) continue;
            pieces.add(tok);
        }
        return decoder.decode(pieces);
    }

    private boolean isConfiguredSpecial(String tok) {
        if (tok == null) return false;
        return tok.equals(unkToken) && false // unk usually kept? HF skips only special flag
                || tok.equals(padToken)
                || tok.equals(clsToken)
                || tok.equals(sepToken)
                || tok.equals(bosToken)
                || tok.equals(eosToken)
                || tok.equals(maskToken)
                || added.isSpecialContent(tok);
    }

    // ---- vocab helpers ------------------------------------------------------

    public int tokenToId(String token) {
        if (token == null) return -1;
        Integer id = vocab.get(token);
        if (id != null) return id;
        if (unkToken != null) {
            Integer u = vocab.get(unkToken);
            return u == null ? -1 : u;
        }
        return -1;
    }

    private Integer tokenToIdOrNeg(String token) {
        if (token == null) return -1;
        Integer id = vocab.get(token);
        return id == null ? -1 : id;
    }

    public String idToToken(int id) {
        return idToToken.get(id);
    }

    public int vocabSize() {
        int max = -1;
        for (Integer v : vocab.values()) {
            if (v != null && v > max) max = v;
        }
        return max + 1;
    }

    public Map<String, Integer> getVocab() {
        return vocab;
    }

    public int padId() {
        if (padToken != null) {
            Integer id = vocab.get(padToken);
            if (id != null) return id;
        }
        // common fallbacks
        for (String c : new String[]{"<pad>", "[PAD]", "<|endoftext|>", padToken}) {
            if (c == null) continue;
            Integer id = vocab.get(c);
            if (id != null) return id;
        }
        return 0;
    }

    public int unkId() {
        if (unkToken != null) {
            Integer id = vocab.get(unkToken);
            if (id != null) return id;
        }
        return 0;
    }

    public int bosId() {
        return tokenToId(bosToken != null ? bosToken : clsToken);
    }

    public int eosId() {
        return tokenToId(eosToken != null ? eosToken : sepToken);
    }

    // ---- accessors ----------------------------------------------------------

    public Normalizer normalizer() { return normalizer; }
    public PreTokenizer preTokenizer() { return preTokenizer; }
    public Model model() { return model; }
    public PostProcessor postProcessor() { return postProcessor; }
    public Decoder decoder() { return decoder; }
    public AddedVocabulary added() { return added; }
    public FastTokenizer.Padding padding() { return padding; }
    public FastTokenizer.Truncation truncation() { return truncation; }
    public String unkToken() { return unkToken; }
    public String padToken() { return padToken; }
    public String clsToken() { return clsToken; }
    public String sepToken() { return sepToken; }
    public String bosToken() { return bosToken; }
    public String eosToken() { return eosToken; }
    public String maskToken() { return maskToken; }
    public int modelMaxLength() { return modelMaxLength; }
    public boolean addPrefixSpace() { return addPrefixSpace; }
}
