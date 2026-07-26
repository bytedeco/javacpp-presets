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
package org.bytedeco.pytorch.utils.text.tokenizer;
import org.bytedeco.pytorch.jit.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Lite Jieba-style Chinese segmenter using dictionary maximum matching (forward).
 * Embeds a small built-in dictionary and accepts an external dict path.
 */
public final class JiebaSegmenter implements Tokenizer {

    private final Set<String> dictionary;
    private int maxWordLen;
    private final boolean keepPunct;

    public JiebaSegmenter() {
        this(defaultDict(), true);
    }

    public JiebaSegmenter(Set<String> dictionary) {
        this(dictionary, true);
    }

    public JiebaSegmenter(Set<String> dictionary, boolean keepPunct) {
        this.dictionary = new HashSet<>(dictionary == null ? Set.of() : dictionary);
        int max = 1;
        for (String w : this.dictionary) {
            if (w != null && w.length() > max) {
                max = w.length();
            }
        }
        this.maxWordLen = Math.max(1, max);
        this.keepPunct = keepPunct;
        // ensure single CJK chars are always valid
        for (char c = 0x4e00; c <= 0x4e20; c++) {
            this.dictionary.add(String.valueOf(c));
        }
    }

    /** Load dict file: one word per line, optional {@code word freq tag}. */
    public static JiebaSegmenter fromDictFile(Path dictPath) {
        Set<String> dict = new HashSet<>(defaultDict());
        try (BufferedReader br = Files.newBufferedReader(dictPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String word = line.split("\\s+")[0];
                if (!word.isEmpty()) {
                    dict.add(word);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new JiebaSegmenter(dict);
    }

    /** Add user words at runtime. */
    public JiebaSegmenter addWord(String word) {
        if (word != null && !word.isEmpty()) {
            dictionary.add(word);
            if (word.length() > maxWordLen) {
                maxWordLen = word.length();
            }
        }
        return this;
    }

    public JiebaSegmenter addWords(Iterable<String> words) {
        if (words != null) {
            for (String w : words) {
                addWord(w);
            }
        }
        return this;
    }

    /** Small built-in Chinese dictionary for demos / fallbacks. */
    public static Set<String> defaultDict() {
        String[] words = {
                "中国", "人民", "共和国", "北京", "上海", "深圳", "广州", "杭州",
                "我们", "你们", "他们", "自己", "什么", "怎么", "为什么", "因为", "所以",
                "可以", "应该", "需要", "已经", "还是", "或者", "但是", "如果", "虽然",
                "今天", "明天", "昨天", "现在", "以后", "以前", "时候", "时间",
                "学习", "工作", "生活", "技术", "科学", "计算机", "软件", "硬件",
                "人工智能", "机器学习", "深度学习", "自然语言", "处理", "神经网络",
                "数据", "模型", "训练", "测试", "验证", "结果", "问题", "方法",
                "一个", "这个", "那个", "一些", "所有", "没有", "不是", "就是",
                "进行", "实现", "开发", "应用", "系统", "网络", "信息", "服务",
                "公司", "大学", "研究", "发展", "经济", "社会", "文化", "历史",
                "你好", "谢谢", "再见", "请", "对不起", "没关系",
                "自然语言处理", "卷积神经网络", "循环神经网络", "注意力机制",
                "分词", "词向量", "语言模型", "预训练", "微调"
        };
        Set<String> set = new HashSet<>();
        Collections.addAll(set, words);
        return set;
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return tokens;
        }
        int i = 0;
        int n = text.length();
        while (i < n) {
            char ch = text.charAt(i);
            if (Character.isWhitespace(ch)) {
                i++;
                continue;
            }
            // non-CJK: take run of alnum or single punct
            if (!isCjk(ch)) {
                if (Character.isLetterOrDigit(ch)) {
                    int j = i + 1;
                    while (j < n && Character.isLetterOrDigit(text.charAt(j)) && !isCjk(text.charAt(j))) {
                        j++;
                    }
                    tokens.add(text.substring(i, j).toLowerCase(Locale.ROOT));
                    i = j;
                } else {
                    if (keepPunct) {
                        tokens.add(String.valueOf(ch));
                    }
                    i++;
                }
                continue;
            }
            // Forward maximum matching: longest dictionary hit, else single char
            int matchedLen = 1;
            int maxTry = Math.min(maxWordLen, n - i);
            for (int len = maxTry; len >= 1; len--) {
                String cand = text.substring(i, i + len);
                if (dictionary.contains(cand) || len == 1) {
                    matchedLen = len;
                    break;
                }
            }
            tokens.add(text.substring(i, i + matchedLen));
            i += matchedLen;
        }
        return tokens;
    }

    private static boolean isCjk(char ch) {
        Character.UnicodeBlock block = Character.UnicodeBlock.of(ch);
        return block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS
                || block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A
                || block == Character.UnicodeBlock.CJK_COMPATIBILITY_IDEOGRAPHS
                || block == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B
                || (ch >= 0x4e00 && ch <= 0x9fff);
    }

    public Set<String> dictionary() {
        return Collections.unmodifiableSet(dictionary);
    }

    /** Alias matching jieba API. */
    public List<String> cut(String text) {
        return tokenize(text);
    }

    /** Return (word, start, end) triples. */
    public List<Map<String, Object>> tokenizeWithOffsets(String text) {
        List<Map<String, Object>> out = new ArrayList<>();
        // re-run with offset tracking
        if (text == null) {
            return out;
        }
        List<String> toks = tokenize(text);
        int pos = 0;
        for (String t : toks) {
            int idx = text.indexOf(t, pos);
            if (idx < 0) {
                idx = pos;
            }
            Map<String, Object> m = new HashMap<>();
            m.put("word", t);
            m.put("start", idx);
            m.put("end", idx + t.length());
            out.add(m);
            pos = idx + t.length();
        }
        return out;
    }
}
