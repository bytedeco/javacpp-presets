package org.bytedeco.pytorch.dataframe.temporal;

import org.bytedeco.pytorch.utils.hudi.HudiTimeline;

import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Unified temporal timeline over Hudi (and future Iceberg) commit instants.
 *
 * <p>Wraps {@link HudiTimeline} without re-parsing; exposes Instant-based API
 * for Point-in-Time joins and audit trails.
 */
public final class TemporalTimeline {

    public enum Source {
        HUDI,
        INLINE  // synthetic / test timeline
    }

    public record Entry(String instantTime, Instant instant, String action, String state) {}

    private final Source source;
    private final List<Entry> entries;
    private final Path tablePath; // may be null for INLINE

    private TemporalTimeline(Source source, List<Entry> entries, Path tablePath) {
        this.source = source;
        this.entries = List.copyOf(entries);
        this.tablePath = tablePath;
    }

    /** Load from a Hudi table path via {@link HudiTimeline}. */
    public static TemporalTimeline fromHudi(Path tablePath) {
        HudiTimeline ht = HudiTimeline.load(tablePath);
        List<Entry> list = new ArrayList<>();
        for (HudiTimeline.Instant inst : ht.instants()) {
            Instant parsed = parseHudiInstant(inst.instantTime());
            list.add(new Entry(
                    inst.instantTime(),
                    parsed,
                    inst.action() == null ? "UNKNOWN" : inst.action().name(),
                    inst.state() == null ? "UNKNOWN" : inst.state().name()));
        }
        return new TemporalTimeline(Source.HUDI, list, tablePath);
    }

    /** Build an inline timeline from epoch-millis commit times (tests / demos). */
    public static TemporalTimeline inline(long... epochMillis) {
        List<Entry> list = new ArrayList<>();
        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
        for (long ms : epochMillis) {
            Instant in = Instant.ofEpochMilli(ms);
            String tag = LocalDateTime.ofInstant(in, ZoneOffset.UTC).format(fmt);
            list.add(new Entry(tag, in, "COMMIT", "COMPLETED"));
        }
        return new TemporalTimeline(Source.INLINE, list, null);
    }

    public Source source() {
        return source;
    }

    public Path tablePath() {
        return tablePath;
    }

    public List<Entry> entries() {
        return entries;
    }

    public int size() {
        return entries.size();
    }

    public boolean isEmpty() {
        return entries.isEmpty();
    }

    /** All completed commit instants as DateTimeIndex (UTC). */
    public DateTimeIndex toDateTimeIndex() {
        List<Instant> instants = new ArrayList<>(entries.size());
        for (Entry e : entries) {
            if (e.instant() != null) instants.add(e.instant());
        }
        return DateTimeIndex.of(instants);
    }

    /**
     * Point-in-time: latest entry with {@code instant <= asOf}.
     * Empty if none.
     */
    public Optional<Entry> asOf(Instant asOf) {
        Objects.requireNonNull(asOf, "asOf");
        Entry best = null;
        for (Entry e : entries) {
            if (e.instant() == null) continue;
            if (!e.instant().isAfter(asOf)) {
                if (best == null || e.instant().isAfter(best.instant())) best = e;
            }
        }
        return Optional.ofNullable(best);
    }

    public Optional<Entry> latest() {
        Entry best = null;
        for (Entry e : entries) {
            if (e.instant() == null) continue;
            if (best == null || e.instant().isAfter(best.instant())) best = e;
        }
        return Optional.ofNullable(best);
    }

    private static Instant parseHudiInstant(String instantTime) {
        if (instantTime == null || instantTime.isBlank()) return null;
        String s = instantTime.trim();
        try {
            // yyyyMMddHHmmss or yyyyMMddHHmmssSSS
            DateTimeFormatter fmt = s.length() >= 17
                    ? DateTimeFormatter.ofPattern("yyyyMMddHHmmssSSS")
                    : DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
            String use = s.length() > 17 ? s.substring(0, 17) : (s.length() > 14 ? s.substring(0, 14) : s);
            if (use.length() == 14) {
                LocalDateTime ldt = LocalDateTime.parse(use, DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
                return ldt.toInstant(ZoneOffset.UTC);
            }
            if (use.length() >= 17) {
                LocalDateTime ldt = LocalDateTime.parse(use.substring(0, 17),
                        DateTimeFormatter.ofPattern("yyyyMMddHHmmssSSS"));
                return ldt.toInstant(ZoneOffset.UTC);
            }
        } catch (Exception ignored) {}
        return null;
    }

    @Override
    public String toString() {
        return "TemporalTimeline{source=" + source + ", size=" + entries.size() + "}";
    }
}
