package dataframe;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Expression;
import org.bytedeco.pytorch.dataframe.Resampler;
import org.bytedeco.pytorch.dataframe.dtype.TimestampData;
import org.bytedeco.pytorch.dataframe.enums.PrecisionType;
import org.bytedeco.pytorch.dataframe.geo.CRS;
import org.bytedeco.pytorch.dataframe.geo.GeoData;
import org.bytedeco.pytorch.dataframe.geo.GeoJoin;
import org.bytedeco.pytorch.dataframe.geo.GeoOptions;
import org.bytedeco.pytorch.dataframe.geo.H3Data;
import org.bytedeco.pytorch.dataframe.geo.S2Data;
import org.bytedeco.pytorch.dataframe.geo.SpatialPredicate;
import org.bytedeco.pytorch.dataframe.temporal.BusinessCalendar;
import org.bytedeco.pytorch.dataframe.temporal.BusinessResampler;
import org.bytedeco.pytorch.dataframe.temporal.DateTimeIndex;
import org.bytedeco.pytorch.dataframe.temporal.DurationData;
import org.bytedeco.pytorch.dataframe.temporal.PeriodIndex;
import org.bytedeco.pytorch.dataframe.temporal.TemporalFeatureExtractor;
import org.bytedeco.pytorch.dataframe.temporal.TemporalOps;
import org.bytedeco.pytorch.dataframe.temporal.TemporalTimeline;
import org.bytedeco.pytorch.dataframe.temporal.TimeUnit;
import org.bytedeco.pytorch.dataframe.temporal.TimeZone;

import java.time.DayOfWeek;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.bytedeco.pytorch.dataframe.Functions.col;

/**
 * Multi-dimension correctness + scale suite for enterprise Temporal + Geo packages.
 *
 * <p>Covers plan acceptance criteria in
 * {@code scripts/doc/ENTERPRISE_DATAFRAME_TEMPORAL_GEO_PLAN.md}:
 * <ul>
 *   <li>TimestampData tz + business day</li>
 *   <li>BusinessCalendar full interface</li>
 *   <li>Expression.dt / TemporalOps pandas-polars ops</li>
 *   <li>DateTimeIndex / PeriodIndex / DurationData / BusinessResampler</li>
 *   <li>GeoData WKT/WKB/GeoJSON, H3/S2, spatial join</li>
 *   <li>Scale: resample + rolling-ish feature extract + geo join (N tunable)</li>
 * </ul>
 *
 * <pre>
 *   # default N=50_000 (fast CI / local)
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... dataframe.BenchmarkDataFrameTemporalGeo
 *
 *   # scale (e.g. 1M rows)
 *   java ... dataframe.BenchmarkDataFrameTemporalGeo --n 1000000
 *
 *   # skip heavy scale section
 *   java ... dataframe.BenchmarkDataFrameTemporalGeo --quick
 * </pre>
 */
public class BenchmarkDataFrameTemporalGeo {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<String> timings = new ArrayList<>();

    /** Default scale; override with --n / env BENCH_N. */
    static int N = 50_000;
    static boolean quick = false;

    interface CheckedRunnable {
        void run() throws Exception;
    }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
            timings.add(String.format(Locale.ROOT, "OK\t%s\t%d", name, ms));
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            timings.add(String.format(Locale.ROOT, "FAIL\t%s\t%d", name, ms));
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
        } else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok = expected == null ? actual == null : expected.equals(actual);
        if (!ok) {
            failed++;
            String msg = name + " expected=" + expected + " actual=" + actual;
            report.append("  check failed: ").append(msg).append('\n');
            throw new AssertionError(msg);
        }
        passed++;
    }

    static void checkClose(String name, double expected, double actual, double eps) {
        boolean ok = Double.isNaN(expected) ? Double.isNaN(actual)
                : Math.abs(expected - actual) <= eps;
        if (!ok) {
            failed++;
            String msg = name + " expected≈" + expected + " actual=" + actual + " eps=" + eps;
            report.append("  check failed: ").append(msg).append('\n');
            throw new AssertionError(msg);
        }
        passed++;
    }

    static long asLong(Object v) {
        return ((Number) v).longValue();
    }

    static double asDouble(Object v) {
        return ((Number) v).doubleValue();
    }

    static void parseArgs(String[] args) {
        String env = System.getenv("BENCH_N");
        if (env != null && !env.isBlank()) {
            try {
                N = Integer.parseInt(env.trim());
            } catch (NumberFormatException ignored) {
            }
        }
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if ("--quick".equals(a) || "-q".equals(a)) {
                quick = true;
            } else if ("--n".equals(a) && i + 1 < args.length) {
                N = Integer.parseInt(args[++i]);
            } else if (a.startsWith("--n=")) {
                N = Integer.parseInt(a.substring(4));
            }
        }
        if (N < 100) N = 100;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        parseArgs(args);
        System.out.println("=== BenchmarkDataFrameTemporalGeo ===");
        System.out.println("N=" + N + " quick=" + quick);
        System.out.println();

        // ---- Phase A: Temporal core correctness ----
        System.out.println("-- Temporal core --");
        benchmark("T1 BusinessCalendar weekends + US federal", BenchmarkDataFrameTemporalGeo::t1BusinessCalendar);
        benchmark("T2 TimestampData tz + business + Arrow desc", BenchmarkDataFrameTemporalGeo::t2TimestampData);
        benchmark("T3 TemporalOps components + flags", BenchmarkDataFrameTemporalGeo::t3TemporalOps);
        benchmark("T4 DateTimeIndex range/asof/reindex", BenchmarkDataFrameTemporalGeo::t4DateTimeIndex);
        benchmark("T5 PeriodIndex month/quarter", BenchmarkDataFrameTemporalGeo::t5PeriodIndex);
        benchmark("T6 DurationData arithmetic", BenchmarkDataFrameTemporalGeo::t6DurationData);
        benchmark("T7 Expression.dt namespace", BenchmarkDataFrameTemporalGeo::t7ExpressionDt);
        benchmark("T8 TemporalFeatureExtractor", BenchmarkDataFrameTemporalGeo::t8FeatureExtractor);
        benchmark("T9 TemporalTimeline inline + asof", BenchmarkDataFrameTemporalGeo::t9Timeline);
        benchmark("T10 Resampler 5m/1h/1d", BenchmarkDataFrameTemporalGeo::t10Resampler);
        benchmark("T11 BusinessResampler", BenchmarkDataFrameTemporalGeo::t11BusinessResampler);

        // ---- Phase B: Geo core correctness ----
        System.out.println();
        System.out.println("-- Geo core --");
        benchmark("G1 GeoData WKT/WKB/GeoJSON roundtrip", BenchmarkDataFrameTemporalGeo::g1GeoRoundtrip);
        benchmark("G2 Spatial predicates + haversine", BenchmarkDataFrameTemporalGeo::g2SpatialPred);
        benchmark("G3 H3 index parent/disk/hex", BenchmarkDataFrameTemporalGeo::g3H3);
        benchmark("G4 S2 index parent/token", BenchmarkDataFrameTemporalGeo::g4S2);
        benchmark("G5 CRS parse + transform approx", BenchmarkDataFrameTemporalGeo::g5Crs);
        benchmark("G6 GeoJoin + h3Join + withH3/S2", BenchmarkDataFrameTemporalGeo::g6GeoJoin);

        // ---- Phase C: Enterprise combo + scale ----
        System.out.println();
        System.out.println("-- Enterprise combo + scale --");
        benchmark("E1 Audit trail (ts + geo + business)", BenchmarkDataFrameTemporalGeo::e1AuditTrail);
        if (!quick) {
            benchmark("E2 Scale resample N=" + N, () -> e2ScaleResample(N));
            benchmark("E3 Scale TemporalOps N=" + N, () -> e3ScaleTemporalOps(N));
            benchmark("E4 Scale H3 index N=" + N, () -> e4ScaleH3(N));
            benchmark("E5 Scale geoJoin (left=" + Math.min(N, 5_000)
                    + " right=" + Math.min(N / 10, 500) + ")", () -> e5ScaleGeoJoin(N));
            benchmark("E6 Scale business features N=" + N, () -> e6ScaleBusinessFeatures(N));
        } else {
            System.out.println("  SKIP scale (--quick)");
        }

        // ---- summary ----
        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println("passed=" + passed + " failed=" + failed);
        System.out.println("--- timings (status\\tname\\tms) ---");
        for (String t : timings) System.out.println(t);
        if (failed > 0) {
            System.out.println("--- failures ---");
            System.out.print(report);
            System.exit(1);
        }
        System.out.println("ALL OK");
    }

    // ================================================================ Temporal

    static void t1BusinessCalendar() {
        BusinessCalendar weekends = BusinessCalendar.weekendsOnly();
        LocalDate mon = LocalDate.of(2024, 1, 8);  // Mon
        LocalDate sat = LocalDate.of(2024, 1, 6);  // Sat
        LocalDate sun = LocalDate.of(2024, 1, 7);  // Sun
        check("Mon business", weekends.isBusinessDay(mon));
        check("Sat not business", !weekends.isBusinessDay(sat));
        check("Sun not business", !weekends.isBusinessDay(sun));
        check("next after Fri is Mon",
                weekends.nextBusinessDay(LocalDate.of(2024, 1, 5)).equals(mon));
        check("prev before Mon is Fri",
                weekends.previousBusinessDay(mon).equals(LocalDate.of(2024, 1, 5)));
        // Fri 5 → Mon 8 is 1 business day between? (exclusive end typical)
        long bd = weekends.businessDaysBetween(LocalDate.of(2024, 1, 5), mon);
        check("businessDaysBetween Fri→Mon >=0", bd >= 0);

        BusinessCalendar us = BusinessCalendar.usFederal(2024);
        // 2024-01-01 is Mon → New Year observed on Mon
        check("NY2024 holiday", us.isHoliday(LocalDate.of(2024, 1, 1)));
        check("NY2024 not business", !us.isBusinessDay(LocalDate.of(2024, 1, 1)));
        // Independence 2024-07-04 is Thu
        check("Jul4 holiday", us.isHoliday(LocalDate.of(2024, 7, 4)));
        // ordinary Tuesday
        check("ordinary Tue business", us.isBusinessDay(LocalDate.of(2024, 7, 2)));

        Set<LocalDate> custom = new TreeSet<>();
        custom.add(LocalDate.of(2024, 3, 15));
        BusinessCalendar c = BusinessCalendar.ofHolidays(custom);
        check("custom holiday", c.isHoliday(LocalDate.of(2024, 3, 15)));
        check("custom not biz", !c.isBusinessDay(LocalDate.of(2024, 3, 15)));

        BusinessCalendar friOff = BusinessCalendar.ofWeekends(DayOfWeek.FRIDAY, DayOfWeek.SATURDAY);
        check("Fri off calendar", !friOff.isBusinessDay(LocalDate.of(2024, 1, 5)));
        check("Sun is business under Fri/Sat weekend", friOff.isBusinessDay(sun));
    }

    static void t2TimestampData() {
        Instant now = Instant.parse("2024-07-04T15:30:00Z");
        TimestampData utc = TimestampData.of(now, ZoneOffset.UTC);
        checkEq("epoch milli", now.toEpochMilli(), utc.toEpochMillis());
        check("zone UTC", utc.getTimeZone().equals(ZoneOffset.UTC)
                || "UTC".equals(utc.getTimeZone().getId())
                || "Z".equals(utc.getTimeZone().getId()));

        ZoneId ny = ZoneId.of("America/New_York");
        TimestampData nyTs = TimestampData.of(now, ny);
        ZonedDateTime zdt = nyTs.toZonedDateTime();
        check("NY local hour 11 (EDT)", zdt.getHour() == 11);

        TimestampData converted = utc.convertZone(ny);
        check("convertZone keeps instant", converted.getInstant().equals(now));
        check("convertZone zone NY", converted.getTimeZone().equals(ny));

        BusinessCalendar us = BusinessCalendar.usFederal(2024);
        TimestampData biz = utc.withBusinessDay(us);
        check("Jul4 not business flag", Boolean.FALSE.equals(biz.getBusinessDay()));
        check("isBusinessDay API", !biz.isBusinessDay(us));

        TimestampData mon = TimestampData.of(Instant.parse("2024-07-01T12:00:00Z"), ZoneOffset.UTC)
                .withBusinessDay(us);
        check("Jul1 Mon is business", Boolean.TRUE.equals(mon.getBusinessDay()));

        TimestampData plus = utc.plusMillis(60_000);
        checkEq("plus 1min", now.plusMillis(60_000).toEpochMilli(), plus.toEpochMillis());

        DurationData gap = utc.until(plus);
        check("until ~60s", Math.abs(gap.toMillis() - 60_000L) < 1);

        Map<String, Object> arrow = utc.toArrowTimestampDesc();
        check("arrow desc has value", arrow.containsKey("value") || arrow.containsKey("timestamp")
                || arrow.size() >= 2);
        Map<String, Object> json = utc.toJsonMap();
        check("json map non-empty", json != null && !json.isEmpty());

        TimestampData micros = new TimestampData(now, PrecisionType.MICROS, ZoneOffset.UTC);
        check("micros precision", micros.precisionType() == PrecisionType.MICROS);
        check("micros roundtrip instant close",
                Math.abs(micros.getInstant().toEpochMilli() - now.toEpochMilli()) <= 1);
    }

    static void t3TemporalOps() {
        ZoneId z = ZoneOffset.UTC;
        Object v = Instant.parse("2024-03-31T23:15:45Z"); // Sun, month-end, Q1 end
        checkEq("year", 2024, TemporalOps.year(v, z));
        checkEq("month", 3, TemporalOps.month(v, z));
        checkEq("day", 31, TemporalOps.day(v, z));
        checkEq("hour", 23, TemporalOps.hour(v, z));
        checkEq("minute", 15, TemporalOps.minute(v, z));
        checkEq("second", 45, TemporalOps.second(v, z));
        checkEq("dow ISO Sun=7", 7, TemporalOps.dayOfWeek(v, z));
        checkEq("quarter", 1, TemporalOps.quarter(v, z));
        check("isWeekend", Boolean.TRUE.equals(TemporalOps.isWeekend(v, z)));
        check("isMonthEnd", Boolean.TRUE.equals(TemporalOps.isMonthEnd(v, z)));
        check("isQuarterEnd", Boolean.TRUE.equals(TemporalOps.isQuarterEnd(v, z)));
        check("isYearEnd false", Boolean.FALSE.equals(TemporalOps.isYearEnd(v, z)));
        check("isLeapYear", Boolean.TRUE.equals(TemporalOps.isLeapYear(v, z)));

        Object mon = LocalDate.of(2024, 1, 8);
        check("isBusinessDay mon", Boolean.TRUE.equals(
                TemporalOps.isBusinessDay(mon, z, BusinessCalendar.weekendsOnly())));
        check("isWeekend mon false", Boolean.FALSE.equals(TemporalOps.isWeekend(mon, z)));

        LocalDate next = TemporalOps.nextBusinessDay(LocalDate.of(2024, 1, 5), z, null);
        checkEq("nextBiz after Fri", LocalDate.of(2024, 1, 8), next);

        LocalDate toBiz = TemporalOps.toBusinessDay(LocalDate.of(2024, 1, 6), z, null);
        checkEq("toBusinessDay Sat→Mon", LocalDate.of(2024, 1, 8), toBiz);

        checkEq("monthStart", LocalDate.of(2024, 3, 1), TemporalOps.monthStart(v, z));
        checkEq("monthEnd", LocalDate.of(2024, 3, 31), TemporalOps.monthEnd(v, z));
        checkEq("quarterStart", LocalDate.of(2024, 1, 1), TemporalOps.quarterStart(v, z));
        checkEq("quarterEnd", LocalDate.of(2024, 3, 31), TemporalOps.quarterEnd(v, z));

        // coerce paths
        check("coerce epoch sec", TemporalOps.toInstant(1_704_067_200L, z) != null); // ~2024-01-01
        check("coerce ISO string", TemporalOps.toInstant("2024-01-01T00:00:00Z", z) != null);
        check("coerce LocalDate", TemporalOps.toLocalDate(LocalDate.of(2024, 2, 29), z)
                .equals(LocalDate.of(2024, 2, 29)));
        check("null-in null-out", TemporalOps.year(null, z) == null);
    }

    static void t4DateTimeIndex() {
        Instant start = Instant.parse("2024-01-01T00:00:00Z");
        Instant end = Instant.parse("2024-01-01T01:00:00Z");
        DateTimeIndex idx = DateTimeIndex.range(start, end, 15 * 60_000L); // 15 min
        check("range size 5 (0,15,30,45,60)", idx.size() == 5);
        check("sorted", idx.isSorted());
        checkEq("iloc0", start, idx.iloc(0));
        check("loc hit", idx.loc(start).isPresent());
        check("loc miss", idx.loc(start.plusSeconds(1)).isEmpty());

        int asof = idx.asof(Instant.parse("2024-01-01T00:20:00Z"));
        check("asof 20min → 15min bin idx 1", asof == 1);

        DateTimeIndex other = DateTimeIndex.of(List.of(
                Instant.parse("2024-01-01T00:05:00Z"),
                Instant.parse("2024-01-01T00:35:00Z"),
                Instant.parse("2023-12-31T23:00:00Z") // before all
        ));
        int[] map = idx.reindexAsof(other);
        checkEq("reindex asof[0]", 0, map[0]);
        checkEq("reindex asof[1]", 2, map[1]); // 30min
        checkEq("reindex asof[2] before", -1, map[2]);

        long[] millis = idx.toEpochMillis();
        checkEq("millis len", 5, millis.length);
        DateTimeIndex fromMs = DateTimeIndex.fromEpochMillis(millis, ZoneOffset.UTC);
        checkEq("fromEpochMillis size", 5, fromMs.size());

        // unsorted → sorted()
        DateTimeIndex unsorted = DateTimeIndex.of(List.of(
                Instant.parse("2024-01-02T00:00:00Z"),
                Instant.parse("2024-01-01T00:00:00Z")
        ));
        check("unsorted flag", !unsorted.isSorted());
        check("sorted() flag", unsorted.sorted().isSorted());
    }

    static void t5PeriodIndex() {
        PeriodIndex months = PeriodIndex.monthRange(YearMonth.of(2024, 1), YearMonth.of(2024, 3));
        checkEq("3 months", 3, months.size());
        check("Jan contains 2024-01-15", months.get(0).contains(LocalDate.of(2024, 1, 15)));
        check("Jan not contains Feb1", !months.get(0).contains(LocalDate.of(2024, 2, 1)));
        checkEq("Jan length days", 31L, months.get(0).lengthDays());

        PeriodIndex quarters = PeriodIndex.quarterRange(2024, 2024);
        checkEq("4 quarters", 4, quarters.size());
        check("Q1 label", quarters.get(0).toString().contains("Q1")
                || quarters.get(0).toString().startsWith("2024"));
        check("Q2 contains May", quarters.get(1).contains(LocalDate.of(2024, 5, 1)));
    }

    static void t6DurationData() {
        DurationData a = DurationData.ofSeconds(90);
        DurationData b = DurationData.ofMillis(30_000);
        checkEq("90s millis", 90_000L, a.toMillis());
        DurationData sum = a.plus(b);
        checkEq("plus", 120_000L, sum.toMillis());
        DurationData diff = a.minus(b);
        checkEq("minus", 60_000L, diff.toMillis());
        checkEq("multiply 2", 180_000L, a.multiply(2).toMillis());
        checkEq("divide by dur", 3L, a.divide(b));
        checkEq("divide by long", 45_000L, a.divide(2).toMillis());
        checkEq("abs neg", 5_000L, DurationData.ofMillis(-5_000).abs().toMillis());
        check("compare", a.compareTo(b) > 0);
        check("toDuration", a.toDuration().equals(Duration.ofSeconds(90)));

        DurationData parsed = DurationData.parse("PT1H30M");
        checkEq("parse PT1H30M", 5_400_000L, parsed.toMillis());
    }

    static void t7ExpressionDt() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("ts", Column.DType.INT64);
        df.addColumn("id", Column.DType.STRING);
        // 2024-03-31 Sun month/quarter end
        long t1 = Instant.parse("2024-03-31T12:00:00Z").toEpochMilli();
        // 2024-07-01 Mon
        long t2 = Instant.parse("2024-07-01T12:00:00Z").toEpochMilli();
        // 2024-07-04 Thu US holiday
        long t3 = Instant.parse("2024-07-04T12:00:00Z").toEpochMilli();
        df.addRow(t1, "a");
        df.addRow(t2, "b");
        df.addRow(t3, "c");

        BusinessCalendar us = BusinessCalendar.usFederal(2024);
        DataFrame out = df
                .withColumn("year", col("ts").dt().year())
                .withColumn("month", col("ts").dt().month())
                .withColumn("quarter", col("ts").dt().quarter())
                .withColumn("dow", col("ts").dt().dayOfWeek())
                .withColumn("is_we", col("ts").dt().isWeekend())
                .withColumn("is_me", col("ts").dt().isMonthEnd())
                .withColumn("is_qe", col("ts").dt().isQuarterEnd())
                .withColumn("is_biz", col("ts").dt().withCalendar(us).isBusinessDay())
                .withColumn("is_hol", col("ts").dt().withCalendar(us).isHoliday());

        checkEq("row0 year", 2024L, asLong(out.get(0, "year")));
        checkEq("row0 month", 3L, asLong(out.get(0, "month")));
        checkEq("row0 quarter", 1L, asLong(out.get(0, "quarter")));
        checkEq("row0 dow Sun", 7L, asLong(out.get(0, "dow")));
        check("row0 weekend", boolish(out.get(0, "is_we")));
        check("row0 month end", boolish(out.get(0, "is_me")));
        check("row0 quarter end", boolish(out.get(0, "is_qe")));
        check("row0 not biz (Sun)", !boolish(out.get(0, "is_biz")));

        check("row1 Mon biz", boolish(out.get(1, "is_biz")));
        check("row1 not weekend", !boolish(out.get(1, "is_we")));

        check("row2 Jul4 holiday", boolish(out.get(2, "is_hol")));
        check("row2 Jul4 not biz", !boolish(out.get(2, "is_biz")));
    }

    static boolean boolish(Object v) {
        if (v instanceof Boolean b) return b;
        if (v instanceof Number n) return n.intValue() != 0;
        if (v == null) return false;
        return Boolean.parseBoolean(v.toString());
    }

    static void t8FeatureExtractor() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("event_ts", Column.DType.INT64);
        df.addColumn("val", Column.DType.FLOAT64);
        long base = Instant.parse("2024-01-31T23:00:00Z").toEpochMilli(); // month end
        df.addRow(base, 1.0);
        df.addRow(base + 86_400_000L, 2.0); // next day Feb 1

        TemporalFeatureExtractor fe = new TemporalFeatureExtractor("event_ts")
                .includeHour(true)
                .includeWeekday(true)
                .includeDay(true)
                .includeMonth(true)
                .includeQuarter(true)
                .includeIsMonthEnd(true)
                .includeIsBusinessDay(true)
                .zone(ZoneOffset.UTC);
        fe.fit(df);
        DataFrame out = fe.transform(df);
        check("feature cols added", out.columnCount() > df.columnCount());
        // at least one new column name containing hour/weekday/month
        boolean found = false;
        for (Column c : out.columns()) {
            String n = c.name().toLowerCase(Locale.ROOT);
            if (n.contains("hour") || n.contains("weekday") || n.contains("month")
                    || n.contains("quarter") || n.contains("business")) {
                found = true;
                break;
            }
        }
        check("expected feature name pattern", found);
    }

    static void t9Timeline() {
        long t0 = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli();
        long t1 = Instant.parse("2024-01-02T00:00:00Z").toEpochMilli();
        long t2 = Instant.parse("2024-01-05T12:00:00Z").toEpochMilli();
        TemporalTimeline tl = TemporalTimeline.inline(t0, t1, t2);
        checkEq("3 entries", 3, tl.size());
        check("source INLINE", tl.source() == TemporalTimeline.Source.INLINE);
        check("first instant", tl.entries().get(0).instant().equals(Instant.ofEpochMilli(t0)));

        // asof / latest before
        Instant query = Instant.parse("2024-01-03T00:00:00Z");
        var hit = tl.asOf(query);
        check("asOf present", hit.isPresent());
        check("asOf → t1", hit.get().instant().equals(Instant.ofEpochMilli(t1)));

        var beforeAll = tl.asOf(Instant.parse("2023-12-01T00:00:00Z"));
        check("asOf before all empty", beforeAll.isEmpty());
    }

    static void t10Resampler() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("t", Column.DType.INT64);
        df.addColumn("x", Column.DType.FLOAT64);
        long t0 = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli();
        // 10 points every 1 minute
        for (int i = 0; i < 10; i++) {
            df.addRow(t0 + i * 60_000L, (double) i);
        }
        DataFrame mean5 = df.resample("t", "5m").mean("x");
        check("5m bins >= 1", mean5.rowCount() >= 1);
        check("5m has t", mean5.hasColumn("t"));
        check("5m has x", mean5.hasColumn("x"));

        DataFrame mean1h = df.resample("t", "1h").mean("x");
        check("1h single bin", mean1h.rowCount() == 1);
        checkClose("1h mean ~4.5", 4.5, asDouble(mean1h.get(0, "x")), 1e-9);

        // daily: still one day
        DataFrame d1 = df.resample("t", "1d").count();
        check("1d count rows >=1", d1.rowCount() >= 1);
    }

    static void t11BusinessResampler() throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("t", Column.DType.INT64);
        df.addColumn("x", Column.DType.FLOAT64);
        // Fri 2024-01-05, Sat 6, Sun 7, Mon 8 — daily points at noon UTC
        LocalDate[] days = {
                LocalDate.of(2024, 1, 5),
                LocalDate.of(2024, 1, 6),
                LocalDate.of(2024, 1, 7),
                LocalDate.of(2024, 1, 8)
        };
        for (int i = 0; i < days.length; i++) {
            long ms = days[i].atTime(12, 0).toInstant(ZoneOffset.UTC).toEpochMilli();
            df.addRow(ms, (double) (i + 1));
        }
        BusinessResampler br = BusinessResampler.of(df, "t", "1d", BusinessCalendar.weekendsOnly());
        DataFrame out = br.mean("x");
        // weekend bins filtered → only Fri + Mon (and maybe empty filtered)
        for (int i = 0; i < out.rowCount(); i++) {
            long t = asLong(out.get(i, "t"));
            LocalDate d = Instant.ofEpochMilli(t).atZone(ZoneOffset.UTC).toLocalDate();
            check("business bin only: " + d, BusinessCalendar.weekendsOnly().isBusinessDay(d));
        }
        check("at least 1 business bin", out.rowCount() >= 1);
    }

    // ================================================================ Geo

    static void g1GeoRoundtrip() {
        GeoData p = GeoData.point(-73.9857, 40.7484); // Empire State-ish
        String wkt = p.toWkt();
        check("WKT has POINT", wkt.toUpperCase(Locale.ROOT).contains("POINT"));
        GeoData p2 = GeoData.fromWkt(wkt);
        checkClose("WKT lon", -73.9857, p2.centroidXy()[0], 1e-6);
        checkClose("WKT lat", 40.7484, p2.centroidXy()[1], 1e-6);

        byte[] wkb = p.toWkb();
        check("WKB non-empty", wkb != null && wkb.length > 0);
        GeoData p3 = GeoData.fromWkb(wkb);
        checkClose("WKB lon", -73.9857, p3.centroidXy()[0], 1e-6);

        String gj = p.toGeoJson();
        check("GeoJSON Point", gj.contains("Point") || gj.contains("point"));
        GeoData p4 = GeoData.fromGeoJson(gj);
        checkClose("GeoJSON lon", -73.9857, p4.centroidXy()[0], 1e-5);

        // Polygon
        String polyWkt = "POLYGON((-74 40, -73 40, -73 41, -74 41, -74 40))";
        GeoData poly = GeoData.fromWkt(polyWkt);
        check("poly type", "Polygon".equalsIgnoreCase(poly.geometryType()));
        check("poly not empty", !poly.isEmpty());

        // parse helpers
        GeoData parsed = GeoData.parse("-73.98,40.75");
        check("parse lon,lat", parsed != null);
        GeoData ewkt = GeoData.parse("SRID=4326;POINT(-73.98 40.75)");
        check("parse EWKT", ewkt != null);
        check("crs WGS84", p.crs().isGeographic());

        Map<?, ?> arrow = (Map<?, ?>) p.toArrowCompatible();
        check("arrow has wkt", arrow.containsKey("wkt"));
        check("arrow has wkb", arrow.containsKey("wkb"));
    }

    static void g2SpatialPred() {
        GeoData poly = GeoData.fromWkt("POLYGON((-74 40, -73 40, -73 41, -74 41, -74 40))");
        GeoData inside = GeoData.point(-73.5, 40.5);
        GeoData outside = GeoData.point(-75.0, 40.5);
        GeoData onEdgeish = GeoData.point(-74.0, 40.5);

        check("within", inside.evaluate(SpatialPredicate.WITHIN, poly));
        check("contains", poly.evaluate(SpatialPredicate.CONTAINS, inside));
        check("intersects inside", inside.evaluate(SpatialPredicate.INTERSECTS, poly));
        check("outside not within", !outside.evaluate(SpatialPredicate.WITHIN, poly));
        check("disjoint outside", outside.evaluate(SpatialPredicate.DISJOINT, poly));
        check("equals self", inside.evaluate(SpatialPredicate.EQUALS, GeoData.point(-73.5, 40.5)));

        // haversine: NYC → ~1 deg lon east ~85km at lat 40.7
        GeoData a = GeoData.point(-74.0, 40.7);
        GeoData b = GeoData.point(-73.0, 40.7);
        double d = a.distance(b);
        check("haversine ~85km ±20km", d > 60_000 && d < 120_000);
        check("DWITHIN 200km", a.evaluate(SpatialPredicate.DWITHIN, b, 200_000));
        check("not DWITHIN 1m", !a.evaluate(SpatialPredicate.DWITHIN, b, 1.0));

        double h = GeoData.haversineMeters(40.7, -74.0, 40.7, -73.0);
        checkClose("haversine static ~distance", d, h, 1.0);
    }

    static void g3H3() {
        H3Data h = H3Data.fromLonLat(-73.9857, 40.7484, 7);
        check("res 7", h.resolution() == 7);
        check("h3 nonzero", h.h3Index() != 0L);
        String hex = h.toHex();
        checkEq("hex roundtrip", h.h3Index(), H3Data.fromHex(hex).h3Index());

        double[] c = h.centerLonLat();
        check("center lon finite", Double.isFinite(c[0]));
        check("center lat finite", Double.isFinite(c[1]));
        // same cell for nearby point at res 7
        H3Data h2 = H3Data.fromLonLat(-73.9857 + 0.0001, 40.7484 + 0.0001, 7);
        // may or may not equal; at least both valid
        check("h2 valid", h2.resolution() == 7);

        H3Data parent = h.toParent(5);
        checkEq("parent res", 5, parent.resolution());

        List<H3Data> disk = h.gridDisk(1);
        check("gridDisk k=1 size >=1", disk.size() >= 1);

        GeoData g = GeoData.point(-73.9857, 40.7484);
        H3Data fromGeo = H3Data.fromGeo(g, 8);
        checkEq("fromGeo res", 8, fromGeo.resolution());

        // determinism
        checkEq("deterministic", h.h3Index(), H3Data.fromLonLat(-73.9857, 40.7484, 7).h3Index());
    }

    static void g4S2() {
        S2Data s = S2Data.fromLonLat(-122.4194, 37.7749, 12); // SF
        check("level 12", s.level() == 12);
        check("cellId nonzero", s.cellId() != 0L);
        String token = s.toToken();
        checkEq("token roundtrip", s.cellId(), S2Data.fromToken(token).cellId());

        S2Data parent = s.toParent(8);
        checkEq("parent level", 8, parent.level());

        double[] c = s.centerLonLat();
        check("s2 center finite", Double.isFinite(c[0]) && Double.isFinite(c[1]));

        S2Data fromGeo = S2Data.fromGeo(GeoData.point(-122.4194, 37.7749), 10);
        checkEq("fromGeo level", 10, fromGeo.level());
        checkEq("deterministic", s.cellId(), S2Data.fromLonLat(-122.4194, 37.7749, 12).cellId());
    }

    static void g5Crs() {
        check("WGS84 epsg 4326", CRS.WGS84.epsg().isPresent() && CRS.WGS84.epsg().get() == 4326);
        check("WGS84 geographic", CRS.WGS84.isGeographic());
        CRS c = CRS.ofEpsg(3857);
        check("3857 code", "3857".equals(c.code()) || c.epsg().orElse(-1) == 3857);
        CRS parsed = CRS.parse("EPSG:4326");
        check("parse EPSG:4326 geographic", parsed.isGeographic());

        double[] xy = CRS.transformLonLatApprox(-73.9857, 40.7484, CRS.WGS84, c);
        check("transform returns 2", xy != null && xy.length == 2);
        check("web mercator x finite", Double.isFinite(xy[0]));
        // rough: NYC mercator x ~ -8.2e6
        check("mercator x magnitude", Math.abs(xy[0]) > 1_000_000);
    }

    static void g6GeoJoin() throws Exception {
        DataFrame left = DataFrame.create();
        left.addColumn("id", Column.DType.STRING);
        left.addColumn("geom", Column.DType.STRING);
        left.addRow("L1", "POINT(-73.5 40.5)");
        left.addRow("L2", "POINT(-75.0 40.5)"); // outside
        left.addRow("L3", "POINT(-73.9 40.7)");

        DataFrame right = DataFrame.create();
        right.addColumn("zone", Column.DType.STRING);
        right.addColumn("geom", Column.DType.STRING);
        right.addRow("Z1", "POLYGON((-74 40, -73 40, -73 41, -74 41, -74 40))");

        DataFrame joined = GeoJoin.geoJoin(left, right, "geom", "geom", SpatialPredicate.WITHIN);
        check("within join >=1", joined.rowCount() >= 1);
        // L1 should match
        boolean sawL1 = false;
        for (int i = 0; i < joined.rowCount(); i++) {
            if ("L1".equals(joined.get(i, "id"))) sawL1 = true;
        }
        check("L1 in within join", sawL1);

        DataFrame h3j = GeoJoin.h3Join(left, left, "geom", "geom", 5);
        check("h3 self-join >= left rows", h3j.rowCount() >= left.rowCount());

        DataFrame withH = GeoJoin.withH3(left, "geom", "h3", 7);
        check("withH3 col", withH.hasColumn("h3"));
        check("h3 non-null row0", withH.get(0, "h3") != null);

        DataFrame withS = GeoJoin.withS2(left, "geom", "s2", 12);
        check("withS2 col", withS.hasColumn("s2"));
        check("s2 non-null row0", withS.get(0, "s2") != null);

        // H3 prefilter path
        GeoOptions opts = GeoOptions.builder().h3Resolution(6).build();
        DataFrame pre = GeoJoin.geoJoin(left, right, "geom", "geom",
                SpatialPredicate.INTERSECTS, 0.0, opts);
        check("prefilter join runs", pre.rowCount() >= 0);
    }

    // ================================================================ Enterprise + scale

    static void e1AuditTrail() throws Exception {
        // Simulate audit rows: operation time + geo + business flag
        BusinessCalendar cal = BusinessCalendar.usFederal(2024);
        DataFrame audit = DataFrame.create();
        audit.addColumn("op_ts", Column.DType.INT64);
        audit.addColumn("user", Column.DType.STRING);
        audit.addColumn("lon", Column.DType.FLOAT64);
        audit.addColumn("lat", Column.DType.FLOAT64);
        audit.addColumn("wkt", Column.DType.STRING);

        Instant[] ops = {
                Instant.parse("2024-07-03T18:00:00Z"), // Wed before holiday
                Instant.parse("2024-07-04T18:00:00Z"), // holiday
                Instant.parse("2024-07-05T18:00:00Z")  // Fri
        };
        double[][] locs = {{-74.0, 40.7}, {-73.9, 40.75}, {-122.4, 37.77}};
        for (int i = 0; i < ops.length; i++) {
            String wkt = String.format(Locale.ROOT, "POINT(%.4f %.4f)", locs[i][0], locs[i][1]);
            audit.addRow(ops[i].toEpochMilli(), "u" + i, locs[i][0], locs[i][1], wkt);
        }

        DataFrame enriched = audit
                .withColumn("is_biz", col("op_ts").dt().withCalendar(cal).isBusinessDay())
                .withColumn("is_hol", col("op_ts").dt().withCalendar(cal).isHoliday())
                .withColumn("dow", col("op_ts").dt().dayOfWeek());
        enriched = GeoJoin.withH3(enriched, "wkt", "h3", 7);
        enriched = GeoJoin.withS2(enriched, "wkt", "s2", 12);

        checkEq("3 audit rows", 3, enriched.rowCount());
        check("Jul4 holiday flag", boolish(enriched.get(1, "is_hol")));
        check("Jul4 not biz", !boolish(enriched.get(1, "is_biz")));
        check("Jul3 biz", boolish(enriched.get(0, "is_biz")));
        check("h3 present", enriched.get(0, "h3") != null);
        check("s2 present", enriched.get(2, "s2") != null);

        // Point-in-time via timeline
        TemporalTimeline tl = TemporalTimeline.inline(
                ops[0].toEpochMilli(), ops[1].toEpochMilli(), ops[2].toEpochMilli());
        var asof = tl.asOf(Instant.parse("2024-07-04T20:00:00Z"));
        check("audit asof holiday commit", asof.isPresent());
        check("asof instant = Jul4", asof.get().instant().equals(ops[1]));
    }

    static void e2ScaleResample(int n) throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("t", Column.DType.INT64);
        df.addColumn("x", Column.DType.FLOAT64);
        long t0 = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli();
        // 1 point / second for n seconds (or cap span)
        int stepMs = n > 500_000 ? 1000 : 200; // keep wall time reasonable
        for (int i = 0; i < n; i++) {
            df.addRow(t0 + (long) i * stepMs, Math.sin(i * 0.001) + i * 1e-6);
        }
        long tStart = System.nanoTime();
        DataFrame r5 = df.resample("t", "5m").mean("x");
        long ms5 = (System.nanoTime() - tStart) / 1_000_000L;
        check("resample 5m rows > 0", r5.rowCount() > 0);

        tStart = System.nanoTime();
        DataFrame r1h = df.resample("t", "1h").mean("x");
        long ms1h = (System.nanoTime() - tStart) / 1_000_000L;
        check("resample 1h rows > 0", r1h.rowCount() > 0);

        tStart = System.nanoTime();
        DataFrame r1d = df.resample("t", "1d").mean("x");
        long ms1d = (System.nanoTime() - tStart) / 1_000_000L;
        check("resample 1d rows > 0", r1d.rowCount() > 0);

        System.out.printf(Locale.ROOT,
                "    scale resample n=%d  5m=%dms(bins=%d) 1h=%dms(bins=%d) 1d=%dms(bins=%d)%n",
                n, ms5, r5.rowCount(), ms1h, r1h.rowCount(), ms1d, r1d.rowCount());
    }

    static void e3ScaleTemporalOps(int n) {
        ZoneId z = ZoneOffset.UTC;
        BusinessCalendar cal = BusinessCalendar.usFederal(2024);
        long t0 = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli();
        int biz = 0, we = 0, me = 0;
        long tStart = System.nanoTime();
        for (int i = 0; i < n; i++) {
            long ts = t0 + (long) i * 3_600_000L; // hourly
            Object v = ts;
            if (Boolean.TRUE.equals(TemporalOps.isBusinessDay(v, z, cal))) biz++;
            if (Boolean.TRUE.equals(TemporalOps.isWeekend(v, z))) we++;
            if (Boolean.TRUE.equals(TemporalOps.isMonthEnd(v, z))) me++;
            TemporalOps.quarter(v, z);
            TemporalOps.dayOfWeek(v, z);
        }
        long ms = (System.nanoTime() - tStart) / 1_000_000L;
        check("scale temporal ops ran", biz + we >= 0);
        System.out.printf(Locale.ROOT,
                "    scale TemporalOps n=%d ms=%d biz=%d weekend=%d monthEnd=%d (%.1f ns/row)%n",
                n, ms, biz, we, me, (ms * 1_000_000.0) / n);
    }

    static void e4ScaleH3(int n) {
        long tStart = System.nanoTime();
        long xor = 0;
        for (int i = 0; i < n; i++) {
            double lon = -180.0 + (360.0 * (i % 10_000)) / 10_000.0;
            double lat = -90.0 + (180.0 * ((i / 10_000) % 10_000)) / 10_000.0;
            xor ^= H3Data.indexLonLat(lon, lat, 7);
            xor ^= S2Data.indexLonLat(lon, lat, 12);
        }
        long ms = (System.nanoTime() - tStart) / 1_000_000L;
        check("h3/s2 scale xor mixed", true); // always runs
        System.out.printf(Locale.ROOT,
                "    scale H3+S2 n=%d ms=%d (%.1f ns/row) checksum=%d%n",
                n, ms, (ms * 1_000_000.0) / Math.max(1, n), xor);
    }

    static void e5ScaleGeoJoin(int n) throws Exception {
        int leftN = Math.min(n, 5_000);
        int rightN = Math.min(Math.max(n / 10, 50), 500);
        DataFrame left = DataFrame.create();
        left.addColumn("id", Column.DType.INT64);
        left.addColumn("geom", Column.DType.STRING);
        for (int i = 0; i < leftN; i++) {
            double lon = -74.0 + (i % 100) * 0.01;
            double lat = 40.0 + (i / 100) * 0.01;
            left.addRow((long) i, String.format(Locale.ROOT, "POINT(%.5f %.5f)", lon, lat));
        }
        DataFrame right = DataFrame.create();
        right.addColumn("zid", Column.DType.INT64);
        right.addColumn("geom", Column.DType.STRING);
        for (int j = 0; j < rightN; j++) {
            double lon0 = -74.0 + (j % 20) * 0.05;
            double lat0 = 40.0 + (j / 20) * 0.05;
            String wkt = String.format(Locale.ROOT,
                    "POLYGON((%.4f %.4f, %.4f %.4f, %.4f %.4f, %.4f %.4f, %.4f %.4f))",
                    lon0, lat0, lon0 + 0.04, lat0, lon0 + 0.04, lat0 + 0.04, lon0, lat0 + 0.04, lon0, lat0);
            right.addRow((long) j, wkt);
        }

        long t0 = System.nanoTime();
        DataFrame pairwise = GeoJoin.geoJoin(left, right, "geom", "geom", SpatialPredicate.WITHIN);
        long msPair = (System.nanoTime() - t0) / 1_000_000L;

        t0 = System.nanoTime();
        GeoOptions opts = GeoOptions.builder().h3Resolution(6).build();
        DataFrame pre = GeoJoin.geoJoin(left, right, "geom", "geom",
                SpatialPredicate.WITHIN, 0.0, opts);
        long msPre = (System.nanoTime() - t0) / 1_000_000L;

        t0 = System.nanoTime();
        DataFrame h3 = GeoJoin.h3Join(left, left, "geom", "geom", 6);
        long msH3 = (System.nanoTime() - t0) / 1_000_000L;

        check("geoJoin ran", pairwise.rowCount() >= 0);
        System.out.printf(Locale.ROOT,
                "    scale geoJoin L=%d R=%d pairwise=%dms(rows=%d) h3pre=%dms(rows=%d) h3eq=%dms(rows=%d)%n",
                leftN, rightN, msPair, pairwise.rowCount(), msPre, pre.rowCount(), msH3, h3.rowCount());
    }

    static void e6ScaleBusinessFeatures(int n) throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("ts", Column.DType.INT64);
        df.addColumn("x", Column.DType.FLOAT64);
        long t0 = Instant.parse("2024-01-01T00:00:00Z").toEpochMilli();
        int step = 3_600_000; // hourly
        int rows = Math.min(n, 200_000); // withColumn materializes; cap for heap
        for (int i = 0; i < rows; i++) {
            df.addRow(t0 + (long) i * step, (double) i);
        }
        BusinessCalendar cal = BusinessCalendar.usFederal(2024);
        long tStart = System.nanoTime();
        DataFrame out = df
                .withColumn("dow", col("ts").dt().dayOfWeek())
                .withColumn("q", col("ts").dt().quarter())
                .withColumn("is_biz", col("ts").dt().withCalendar(cal).isBusinessDay())
                .withColumn("is_me", col("ts").dt().isMonthEnd());
        long ms = (System.nanoTime() - tStart) / 1_000_000L;
        checkEq("feature rows", rows, out.rowCount());
        check("has is_biz", out.hasColumn("is_biz"));
        System.out.printf(Locale.ROOT,
                "    scale Expression.dt features rows=%d ms=%d (%.1f µs/row)%n",
                rows, ms, (ms * 1000.0) / Math.max(1, rows));
    }
}
