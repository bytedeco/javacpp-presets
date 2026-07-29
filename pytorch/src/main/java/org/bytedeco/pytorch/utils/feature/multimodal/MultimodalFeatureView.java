/*
 * Multimodal feature view — binds tabular + text/image/audio/embedding fields
 * into a standard FeatureView with modality tags (short-video / news / pharma).
 */
package org.bytedeco.pytorch.utils.feature.multimodal;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureTable;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.Project;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Multimodal feature view binder for industrial multi-modal stores
 * (cover emb + ASR + watch seq, news title emb, product image emb, molecule emb).
 */
public final class MultimodalFeatureView {

    private final String name;
    private final String project;
    private final List<Entity> entities;
    private final List<Field> tabular;
    private final List<EmbeddingFeatureSpec> embeddings;
    private final List<TextFeatureSpec> texts;
    private final List<ImageFeatureSpec> images;
    private final List<AudioFeatureSpec> audios;
    private final FeatureTable source;
    private final Duration ttl;
    private final boolean online;
    private final String description;
    private final Map<String, String> tags;

    private MultimodalFeatureView(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.project = b.project != null ? b.project : Project.DEFAULT;
        this.entities = Collections.unmodifiableList(new ArrayList<>(b.entities));
        this.tabular = Collections.unmodifiableList(new ArrayList<>(b.tabular));
        this.embeddings = Collections.unmodifiableList(new ArrayList<>(b.embeddings));
        this.texts = Collections.unmodifiableList(new ArrayList<>(b.texts));
        this.images = Collections.unmodifiableList(new ArrayList<>(b.images));
        this.audios = Collections.unmodifiableList(new ArrayList<>(b.audios));
        this.source = b.source != null ? b.source : FeatureTable.lance(name, "lance://" + name);
        this.ttl = b.ttl != null ? b.ttl : Duration.ofDays(7);
        this.online = b.online;
        this.description = b.description != null ? b.description : "";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public String name() { return name; }
    public String project() { return project; }
    public List<Entity> entities() { return entities; }
    public List<EmbeddingFeatureSpec> embeddings() { return embeddings; }
    public List<TextFeatureSpec> texts() { return texts; }
    public List<ImageFeatureSpec> images() { return images; }
    public List<AudioFeatureSpec> audios() { return audios; }
    public FeatureTable source() { return source; }
    public Duration ttl() { return ttl; }
    public boolean online() { return online; }
    public String description() { return description; }
    public Map<String, String> tags() { return tags; }

    /** Flatten all fields across modalities. */
    public List<Field> allFields() {
        List<Field> fields = new ArrayList<>(tabular);
        for (EmbeddingFeatureSpec e : embeddings) fields.add(e.toField());
        for (TextFeatureSpec t : texts) fields.addAll(t.fields());
        for (ImageFeatureSpec i : images) fields.addAll(i.fields());
        for (AudioFeatureSpec a : audios) fields.addAll(a.fields());
        return fields;
    }

    /** Project as a standard batch FeatureView for registry / materialize / PIT. */
    public FeatureView toFeatureView() {
        Map<String, String> t = new LinkedHashMap<>(tags);
        t.putIfAbsent("multimodal", "true");
        if (!texts.isEmpty()) t.putIfAbsent("has_text", "true");
        if (!images.isEmpty()) t.putIfAbsent("has_image", "true");
        if (!audios.isEmpty()) t.putIfAbsent("has_audio", "true");
        if (!embeddings.isEmpty()) t.putIfAbsent("has_embedding", "true");
        return FeatureView.builder(name)
                .project(project)
                .entities(entities)
                .schema(allFields())
                .source(source)
                .ttl(ttl)
                .online(online)
                .description(description)
                .tags(t)
                .build();
    }

    public static final class Builder {
        private final String name;
        private String project = Project.DEFAULT;
        private final List<Entity> entities = new ArrayList<>();
        private final List<Field> tabular = new ArrayList<>();
        private final List<EmbeddingFeatureSpec> embeddings = new ArrayList<>();
        private final List<TextFeatureSpec> texts = new ArrayList<>();
        private final List<ImageFeatureSpec> images = new ArrayList<>();
        private final List<AudioFeatureSpec> audios = new ArrayList<>();
        private FeatureTable source;
        private Duration ttl = Duration.ofDays(7);
        private boolean online = true;
        private String description;
        private final Map<String, String> tags = new LinkedHashMap<>();

        private Builder(String name) { this.name = name; }

        public Builder project(String project) { this.project = project; return this; }

        public Builder entities(Entity... es) {
            if (es != null) entities.addAll(Arrays.asList(es));
            return this;
        }

        public Builder entities(List<Entity> es) {
            if (es != null) entities.addAll(es);
            return this;
        }

        public Builder tabular(Field... fields) {
            if (fields != null) tabular.addAll(Arrays.asList(fields));
            return this;
        }

        public Builder embedding(EmbeddingFeatureSpec spec) {
            if (spec != null) embeddings.add(spec);
            return this;
        }

        public Builder embedding(String name, int dim) {
            embeddings.add(EmbeddingFeatureSpec.of(name, dim));
            return this;
        }

        public Builder text(TextFeatureSpec spec) {
            if (spec != null) texts.add(spec);
            return this;
        }

        public Builder text(String name, int embDim) {
            texts.add(TextFeatureSpec.of(name, embDim));
            return this;
        }

        public Builder image(ImageFeatureSpec spec) {
            if (spec != null) images.add(spec);
            return this;
        }

        public Builder image(String name, int embDim) {
            images.add(ImageFeatureSpec.of(name, embDim));
            return this;
        }

        public Builder audio(AudioFeatureSpec spec) {
            if (spec != null) audios.add(spec);
            return this;
        }

        public Builder audio(String name, int embDim) {
            audios.add(AudioFeatureSpec.of(name, embDim));
            return this;
        }

        public Builder source(FeatureTable source) { this.source = source; return this; }
        public Builder ttl(Duration ttl) { this.ttl = ttl; return this; }
        public Builder ttlDays(long days) { this.ttl = Duration.ofDays(days); return this; }
        public Builder online(boolean online) { this.online = online; return this; }
        public Builder description(String description) { this.description = description; return this; }
        public Builder tag(String k, String v) {
            if (k != null && v != null) tags.put(k, v);
            return this;
        }

        public MultimodalFeatureView build() {
            return new MultimodalFeatureView(this);
        }
    }
}
