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
package org.bytedeco.pytorch.utils.text.transforms;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.utils.text.vocab.Vocab;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Torchtext-style text transforms: Truncate, Pad, Vocab, Sequential, ToTensor.
 */
public final class TextTransforms {

    private TextTransforms() {}

    /** Functional transform interface. */
    @FunctionalInterface
    public interface Transform<I, O> {
        O apply(I input);

        default <R> Transform<I, R> andThen(Transform<? super O, ? extends R> after) {
            Objects.requireNonNull(after);
            return in -> after.apply(apply(in));
        }
    }

    /** Truncate a token list to {@code maxLen} (from the front). */
    public static final class Truncate implements Transform<List<String>, List<String>> {
        private final int maxLen;

        public Truncate(int maxLen) {
            this.maxLen = Math.max(0, maxLen);
        }

        @Override
        public List<String> apply(List<String> input) {
            if (input == null) {
                return List.of();
            }
            if (input.size() <= maxLen) {
                return new ArrayList<>(input);
            }
            return new ArrayList<>(input.subList(0, maxLen));
        }
    }

    /** Truncate integer ids. */
    public static final class TruncateIds implements Transform<int[], int[]> {
        private final int maxLen;

        public TruncateIds(int maxLen) {
            this.maxLen = Math.max(0, maxLen);
        }

        @Override
        public int[] apply(int[] input) {
            if (input == null) {
                return new int[0];
            }
            if (input.length <= maxLen) {
                return Arrays.copyOf(input, input.length);
            }
            return Arrays.copyOf(input, maxLen);
        }
    }

    /**
     * Pad (or truncate) a list of token ids to fixed length.
     * Pads on the right by default.
     */
    public static final class PadTransform implements Transform<int[], int[]> {
        private final int maxLength;
        private final int padValue;
        private final boolean padRight;

        public PadTransform(int maxLength, int padValue) {
            this(maxLength, padValue, true);
        }

        public PadTransform(int maxLength, int padValue, boolean padRight) {
            this.maxLength = Math.max(0, maxLength);
            this.padValue = padValue;
            this.padRight = padRight;
        }

        @Override
        public int[] apply(int[] input) {
            int[] out = new int[maxLength];
            if (input == null || maxLength == 0) {
                Arrays.fill(out, padValue);
                return out;
            }
            int copy = Math.min(input.length, maxLength);
            if (padRight) {
                System.arraycopy(input, 0, out, 0, copy);
                for (int i = copy; i < maxLength; i++) {
                    out[i] = padValue;
                }
            } else {
                int start = maxLength - copy;
                Arrays.fill(out, 0, start, padValue);
                System.arraycopy(input, 0, out, start, copy);
            }
            return out;
        }

        public int[] applyTokensAsIds(List<Integer> ids) {
            if (ids == null) {
                return apply(null);
            }
            int[] arr = new int[ids.size()];
            for (int i = 0; i < ids.size(); i++) {
                arr[i] = ids.get(i);
            }
            return apply(arr);
        }
    }

    /** Map tokens → ids via {@link Vocab}. */
    public static final class VocabTransform implements Transform<List<String>, int[]> {
        private final Vocab vocab;

        public VocabTransform(Vocab vocab) {
            this.vocab = Objects.requireNonNull(vocab, "vocab");
        }

        @Override
        public int[] apply(List<String> input) {
            return vocab.encode(input);
        }
    }

    /** Compose transforms sequentially. */
    public static final class Sequential<I, O> implements Transform<I, O> {
        private final List<Transform<?, ?>> transforms;

        @SafeVarargs
        public Sequential(Transform<?, ?>... transforms) {
            this.transforms = new ArrayList<>(Arrays.asList(transforms));
        }

        public Sequential(List<Transform<?, ?>> transforms) {
            this.transforms = new ArrayList<>(transforms);
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        @Override
        public O apply(I input) {
            Object cur = input;
            for (Transform t : transforms) {
                cur = t.apply(cur);
            }
            return (O) cur;
        }

        public Sequential<I, O> add(Transform<?, ?> t) {
            transforms.add(t);
            return this;
        }
    }

    /** Convert int ids to a 1-D Long Tensor. */
    public static final class ToTensor implements Transform<int[], Tensor> {
        @Override
        public Tensor apply(int[] input) {
            if (input == null || input.length == 0) {
                return torch.tensor(new long[0]);
            }
            long[] data = new long[input.length];
            for (int i = 0; i < input.length; i++) {
                data[i] = input[i];
            }
            return torch.tensor(data);
        }
    }

    /** Convert long ids to a 1-D Long Tensor. */
    public static final class ToTensorLong implements Transform<long[], Tensor> {
        @Override
        public Tensor apply(long[] input) {
            if (input == null) {
                return torch.tensor(new long[0]);
            }
            return torch.tensor(input);
        }
    }

    /** Lowercase string. */
    public static final class Lowercase implements Transform<String, String> {
        @Override
        public String apply(String input) {
            return input == null ? "" : input.toLowerCase(java.util.Locale.ROOT);
        }
    }

    /** Tokenize string with a Function (e.g. tokenizer::tokenize). */
    public static final class Tokenize implements Transform<String, List<String>> {
        private final Function<String, List<String>> tokenizer;

        public Tokenize(Function<String, List<String>> tokenizer) {
            this.tokenizer = Objects.requireNonNull(tokenizer);
        }

        @Override
        public List<String> apply(String input) {
            return tokenizer.apply(input == null ? "" : input);
        }
    }

    // -------------------------------------------------------------------------
    // torchtext.transforms extras (historical; prefer HF tokenizers for new code)
    // -------------------------------------------------------------------------

    /**
     * Prepend or append a special token string to a token list
     * (torchtext.transforms.AddToken).
     */
    public static final class AddToken implements Transform<List<String>, List<String>> {
        private final String token;
        private final boolean begin;

        public AddToken(String token) {
            this(token, true);
        }

        public AddToken(String token, boolean begin) {
            this.token = Objects.requireNonNull(token, "token");
            this.begin = begin;
        }

        @Override
        public List<String> apply(List<String> input) {
            List<String> out = new ArrayList<>();
            if (begin) {
                out.add(token);
            }
            if (input != null) {
                out.addAll(input);
            }
            if (!begin) {
                out.add(token);
            }
            return out;
        }
    }

    /** Prepend/append an integer id to an id sequence. */
    public static final class AddTokenId implements Transform<int[], int[]> {
        private final int tokenId;
        private final boolean begin;

        public AddTokenId(int tokenId) {
            this(tokenId, true);
        }

        public AddTokenId(int tokenId, boolean begin) {
            this.tokenId = tokenId;
            this.begin = begin;
        }

        @Override
        public int[] apply(int[] input) {
            int n = input == null ? 0 : input.length;
            int[] out = new int[n + 1];
            if (begin) {
                out[0] = tokenId;
                if (n > 0) {
                    System.arraycopy(input, 0, out, 1, n);
                }
            } else {
                if (n > 0) {
                    System.arraycopy(input, 0, out, 0, n);
                }
                out[n] = tokenId;
            }
            return out;
        }
    }

    /**
     * Regex replace on raw text (torchtext.transforms.RegexReplace).
     */
    public static final class RegexReplace implements Transform<String, String> {
        private final java.util.regex.Pattern pattern;
        private final String replacement;

        public RegexReplace(String pattern, String replacement) {
            this.pattern = java.util.regex.Pattern.compile(Objects.requireNonNull(pattern, "pattern"));
            this.replacement = replacement == null ? "" : replacement;
        }

        public RegexReplace(java.util.regex.Pattern pattern, String replacement) {
            this.pattern = Objects.requireNonNull(pattern, "pattern");
            this.replacement = replacement == null ? "" : replacement;
        }

        @Override
        public String apply(String input) {
            if (input == null) {
                return "";
            }
            return pattern.matcher(input).replaceAll(replacement);
        }
    }

    /**
     * Character n-gram features over a string or token list
     * (torchtext.transforms.CharNGram-style).
     * <p>
     * For a string {@code "hello"} with n=3 → {@code ["hel","ell","llo"]}.
     * For a token list, n-grams are computed per token and concatenated.
     */
    public static final class CharNGram implements Transform<Object, List<String>> {
        private final int n;

        public CharNGram(int n) {
            if (n < 1) {
                throw new IllegalArgumentException("n must be >= 1");
            }
            this.n = n;
        }

        @Override
        public List<String> apply(Object input) {
            List<String> out = new ArrayList<>();
            if (input == null) {
                return out;
            }
            if (input instanceof String s) {
                addNgrams(out, s);
            } else if (input instanceof List<?> list) {
                for (Object o : list) {
                    if (o != null) {
                        addNgrams(out, o.toString());
                    }
                }
            } else {
                addNgrams(out, input.toString());
            }
            return out;
        }

        private void addNgrams(List<String> out, String s) {
            if (s == null || s.isEmpty()) {
                return;
            }
            if (s.length() < n) {
                out.add(s);
                return;
            }
            for (int i = 0; i <= s.length() - n; i++) {
                out.add(s.substring(i, i + n));
            }
        }
    }

    /**
     * Apply a transform element-wise over a batch (list) of inputs
     * (torchtext.transforms.BatchTransform).
     */
    public static final class BatchTransform<I, O> implements Transform<List<I>, List<O>> {
        private final Transform<? super I, ? extends O> transform;

        public BatchTransform(Transform<? super I, ? extends O> transform) {
            this.transform = Objects.requireNonNull(transform, "transform");
        }

        @Override
        public List<O> apply(List<I> input) {
            if (input == null) {
                return List.of();
            }
            List<O> out = new ArrayList<>(input.size());
            for (I item : input) {
                out.add(transform.apply(item));
            }
            return out;
        }
    }

    /** Strip leading/trailing whitespace. */
    public static final class Strip implements Transform<String, String> {
        @Override
        public String apply(String input) {
            return input == null ? "" : input.strip();
        }
    }

    // ---- factories ----

    public static Truncate truncate(int maxLen) {
        return new Truncate(maxLen);
    }

    public static TruncateIds truncateIds(int maxLen) {
        return new TruncateIds(maxLen);
    }

    public static PadTransform pad(int maxLength, int padValue) {
        return new PadTransform(maxLength, padValue);
    }

    public static VocabTransform vocab(Vocab vocab) {
        return new VocabTransform(vocab);
    }

    public static ToTensor toTensor() {
        return new ToTensor();
    }

    public static Tokenize tokenize(Function<String, List<String>> tokenizer) {
        return new Tokenize(tokenizer);
    }

    public static AddToken addToken(String token, boolean begin) {
        return new AddToken(token, begin);
    }

    public static AddTokenId addTokenId(int tokenId, boolean begin) {
        return new AddTokenId(tokenId, begin);
    }

    public static RegexReplace regexReplace(String pattern, String replacement) {
        return new RegexReplace(pattern, replacement);
    }

    public static CharNGram charNGram(int n) {
        return new CharNGram(n);
    }

    public static <I, O> BatchTransform<I, O> batch(Transform<? super I, ? extends O> transform) {
        return new BatchTransform<>(transform);
    }

    @SafeVarargs
    public static <I, O> Sequential<I, O> sequential(Transform<?, ?>... transforms) {
        return new Sequential<>(transforms);
    }
}
