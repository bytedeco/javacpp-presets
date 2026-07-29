/*
 * Serving-time ranking pipeline for multi-stage recommendation.
 *
 * Canonical cascade used by YouTube, TikTok, Taobao, Meta Feed, Netflix:
 *
 *   Request
 *     -> Recall (multi-channel candidate generation, ~thousands-millions)
 *     -> Coarse rank (lightweight model / formula, cut to hundreds)
 *     -> Fine rank (heavy CTR/CVR model, score remaining)
 *     -> Re-rank (diversity, freshness, business rules, listwise)
 *     -> Mix / merge (multi-queue insert, ads/organic, force-insert)
 *     -> Response
 *
 * Each stage has: timeout budget, quota (max candidates out), fallback,
 * and AB-parameter overlays from LayeredExperimentManager.
 */
package org.bytedeco.pytorch.utils.recommend.serving.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** A single candidate item flowing through the ranking cascade. */
public final class Candidate {

    private final String itemId;
    private double score;
    private final Map<String, Double> scores;
    private final Map<String, String> tags;
    private final List<String> recallChannels;
    private int rank;

    public Candidate(String itemId) {
        this(itemId, 0.0);
    }

    public Candidate(String itemId, double score) {
        if (itemId == null || itemId.isEmpty()) {
            throw new IllegalArgumentException("itemId required");
        }
        this.itemId = itemId;
        this.score = score;
        this.scores = new LinkedHashMap<>();
        this.tags = new LinkedHashMap<>();
        this.recallChannels = new ArrayList<>();
        this.rank = -1;
    }

    public String itemId() {
        return itemId;
    }

    public double score() {
        return score;
    }

    public Candidate score(double score) {
        this.score = score;
        return this;
    }

    public Candidate putScore(String name, double value) {
        scores.put(name, value);
        return this;
    }

    public double getScore(String name, double defaultValue) {
        return scores.getOrDefault(name, defaultValue);
    }

    public Map<String, Double> scores() {
        return Collections.unmodifiableMap(scores);
    }

    public Candidate tag(String key, String value) {
        tags.put(key, value);
        return this;
    }

    public String tag(String key) {
        return tags.get(key);
    }

    public Map<String, String> tags() {
        return Collections.unmodifiableMap(tags);
    }

    public Candidate addRecallChannel(String channel) {
        if (channel != null && !recallChannels.contains(channel)) {
            recallChannels.add(channel);
        }
        return this;
    }

    public List<String> recallChannels() {
        return Collections.unmodifiableList(recallChannels);
    }

    public int rank() {
        return rank;
    }

    public Candidate rank(int rank) {
        this.rank = rank;
        return this;
    }

    /** Shallow copy for branch / shadow scoring. */
    public Candidate copy() {
        Candidate c = new Candidate(itemId, score);
        c.scores.putAll(scores);
        c.tags.putAll(tags);
        c.recallChannels.addAll(recallChannels);
        c.rank = rank;
        return c;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Candidate)) return false;
        return itemId.equals(((Candidate) o).itemId);
    }

    @Override
    public int hashCode() {
        return itemId.hashCode();
    }

    @Override
    public String toString() {
        return "Candidate{id=" + itemId + ", score=" + score + ", rank=" + rank
                + ", channels=" + recallChannels + "}";
    }
}
