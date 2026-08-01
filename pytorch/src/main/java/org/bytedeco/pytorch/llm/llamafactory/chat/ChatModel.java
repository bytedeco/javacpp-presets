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
package org.bytedeco.pytorch.llm.llamafactory.chat;

import org.bytedeco.pytorch.llm.llamafactory.data.SimpleTokenizer;
import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;
import org.bytedeco.pytorch.llm.llamafactory.hparams.GeneratingArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.InferArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.transformers.CausalLM;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * In-process chat over a loaded {@link CausalLM} + {@link SimpleTokenizer}.
 *
 * <p>Production hosts may swap the tokenizer for FastTokenizer; generation uses
 * {@link CausalLM#generate(int[], int, CausalLM.GenerationConfig)}.
 */
public final class ChatModel implements ChatEngine {

    private final LoadedModel loaded;
    private final CausalLM causal;
    private final SimpleTokenizer tokenizer;
    private final Template template;
    private final GeneratingArgs generating;
    private final List<Turn> history = new ArrayList<>();
    private final String defaultSystem;

    public ChatModel(LoadedModel loaded, SimpleTokenizer tokenizer, Template template,
                     GeneratingArgs generating, String defaultSystem) {
        this.loaded = Objects.requireNonNull(loaded, "loaded");
        this.causal = Objects.requireNonNull(loaded.causalLM(), "causalLM");
        this.tokenizer = tokenizer == null ? SimpleTokenizer.defaults() : tokenizer;
        this.template = template == null ? TemplateRegistry.getOrDefault("default") : template;
        this.generating = generating == null ? GeneratingArgs.defaults() : generating;
        this.defaultSystem = defaultSystem;
    }

    public static ChatModel fromLoaded(LoadedModel loaded, InferArgs infer) {
        Objects.requireNonNull(loaded, "loaded");
        InferArgs ia = infer == null ? InferArgs.defaults() : infer;
        Template tpl = TemplateRegistry.getOrDefault(ia.template());
        GeneratingArgs gen = ia.generating() == null ? GeneratingArgs.defaults() : ia.generating();
        return new ChatModel(loaded, SimpleTokenizer.defaults(), tpl, gen, gen.defaultSystem());
    }

    public static ChatModel fromLoaded(LoadedModel loaded, GeneratingArgs gen, String templateName) {
        Template tpl = TemplateRegistry.getOrDefault(templateName);
        GeneratingArgs g = gen == null ? GeneratingArgs.defaults() : gen;
        return new ChatModel(loaded, SimpleTokenizer.defaults(), tpl, g, g.defaultSystem());
    }

    public LoadedModel loaded() { return loaded; }
    public CausalLM causalLM() { return causal; }
    public SimpleTokenizer tokenizer() { return tokenizer; }
    public List<Turn> history() { return List.copyOf(history); }

    public void clearHistory() {
        history.clear();
    }

    @Override
    public String chat(String userMessage) {
        return chat(defaultSystem, userMessage);
    }

    @Override
    public String chat(String system, String userMessage) {
        Objects.requireNonNull(userMessage, "userMessage");
        String prompt = buildPrompt(system, userMessage);
        long[] ids = tokenizer.encode(prompt, false);
        int[] promptIds = toIntIds(ids);
        int maxNew = generating.maxNewTokens() > 0 ? generating.maxNewTokens() : 64;
        if (generating.maxLength() > 0) {
            maxNew = Math.max(1, generating.maxLength() - promptIds.length);
        }
        CausalLM.GenerationConfig cfg = toGenConfig(generating);
        int[] out = causal.generate(promptIds, maxNew, cfg);
        // strip prompt prefix if model echoes it
        int[] genOnly = stripPrompt(out, promptIds);
        String text = tokenizer.decode(toLongIds(genOnly));
        if (generating.skipSpecialTokens()) {
            text = text.trim();
        }
        history.add(new Turn("user", userMessage));
        history.add(new Turn("assistant", text));
        return text;
    }

    private String buildPrompt(String system, String userMessage) {
        String sys = system == null || system.isBlank() ? defaultSystem : system;
        List<Template.Message> msgs = new ArrayList<>();
        int from = Math.max(0, history.size() - 8);
        for (int i = from; i < history.size(); i++) {
            Turn t = history.get(i);
            if ("assistant".equalsIgnoreCase(t.role)) {
                msgs.add(Template.Message.assistant(t.content));
            } else if ("system".equalsIgnoreCase(t.role)) {
                msgs.add(Template.Message.system(t.content));
            } else {
                msgs.add(Template.Message.user(t.content));
            }
        }
        msgs.add(Template.Message.user(userMessage));
        try {
            // encodePrompt leaves the assistant prefix open for generation
            return template.encodePrompt(msgs, sys);
        } catch (Throwable t) {
            StringBuilder sb = new StringBuilder();
            if (sys != null && !sys.isBlank()) {
                sb.append(sys).append('\n');
            }
            for (Template.Message m : msgs) {
                sb.append(m.role()).append(": ").append(m.content()).append('\n');
            }
            sb.append("assistant:");
            return sb.toString();
        }
    }

    private static CausalLM.GenerationConfig toGenConfig(GeneratingArgs g) {
        if (!g.doSample()) {
            return CausalLM.GenerationConfig.greedy();
        }
        return CausalLM.GenerationConfig.builder()
                .doSample(true)
                .temperature(g.temperature() > 0 ? g.temperature() : 0.95)
                .topK(g.topK() > 0 ? g.topK() : 50)
                .topP(g.topP() > 0 ? g.topP() : 0.9)
                .eosStop(true)
                .build();
    }

    private static int[] toIntIds(long[] ids) {
        int[] out = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            out[i] = (int) ids[i];
        }
        return out;
    }

    private static long[] toLongIds(int[] ids) {
        long[] out = new long[ids.length];
        for (int i = 0; i < ids.length; i++) {
            out[i] = ids[i];
        }
        return out;
    }

    private static int[] stripPrompt(int[] full, int[] prompt) {
        if (full == null) return new int[0];
        if (prompt == null || prompt.length == 0 || full.length <= prompt.length) {
            return full;
        }
        boolean prefix = true;
        for (int i = 0; i < prompt.length; i++) {
            if (full[i] != prompt[i]) {
                prefix = false;
                break;
            }
        }
        if (!prefix) return full;
        int[] gen = new int[full.length - prompt.length];
        System.arraycopy(full, prompt.length, gen, 0, gen.length);
        return gen;
    }

    @Override
    public void close() {
        // model ownership stays with LoadedModel / job
    }

    /** One chat turn. */
    public static final class Turn {
        public final String role;
        public final String content;

        public Turn(String role, String content) {
            this.role = role == null ? "" : role;
            this.content = content == null ? "" : content;
        }
    }
}
