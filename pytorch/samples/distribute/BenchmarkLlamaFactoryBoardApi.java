/*
 * BenchmarkLlamaFactoryBoardApi — LlamaBoard HTTP + OpenAI API smoke
 *
 * Run: java -cp ... distribute.BenchmarkLlamaFactoryBoardApi
 */
package distribute;

import org.bytedeco.pytorch.llm.llamafactory.api.ApiAuth;
import org.bytedeco.pytorch.llm.llamafactory.api.OpenAiTypes;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.llamafactory.webui.LlamaBoard;

import java.io.IOException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * D1  BoardState status / metric / stop API
 * D2  LlamaBoard HTTP server starts
 * D3  GET /api/state
 * D4  GET /api/runs
 * D5  ApiAuth enabled / disabled
 * D6  OpenAiTypes request/response roundtrip
 * D7  BoardState stopRequested cooperative cancel
 * D8  LlamaBoard close is clean
 */
public class BenchmarkLlamaFactoryBoardApi {

    static int passed = 0, failed = 0;
    static final List<String> failures = new ArrayList<>();
    static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkLlamaFactoryBoardApi ===\n");
        d1BoardState();
        d2BoardStart();
        d3BoardStateApi();
        d4BoardRunsApi();
        d5ApiAuth();
        d6OpenAiTypesRoundtrip();
        d7BoardStop();
        d8BoardClose();
        done();
    }

    // ── D1 ───────────────────────────────────────────────────────────────────
    static void d1BoardState() {
        section("D1 BoardState API");
        benchmark("BoardState initial status IDLE", () -> {
            BoardState s = new BoardState();
            check("initial status IDLE", s.status() == BoardState.Status.IDLE);
            check("initial step=0", s.globalStep() == 0);
            check("stopRequested=false", !s.stopRequested());
        });

        benchmark("BoardState putMetric / recordLoss", () -> {
            BoardState s = new BoardState();
            s.putMetric("loss", 0.5);
            s.recordLoss(0.7);
            check("metrics contains loss", s.metrics().containsKey("loss"));
            check("metrics loss=0.5", s.metrics().get("loss") == 0.5);
            check("lossHistory non-empty", !s.lossHistory().isEmpty());
            check("lossHistory last=0.7", s.lossHistory().get(s.lossHistory().size() - 1) == 0.7);
        });

        benchmark("BoardState snapshot", () -> {
            BoardState s = new BoardState();
            s.putMetric("lr", 1e-4);
            s.setGlobalStep(10);
            java.util.Map<String, Object> snap = s.snapshot();
            check("snapshot has status", snap.containsKey("status"));
            check("snapshot has global_step", snap.containsKey("global_step"));
            check("snapshot has metrics", snap.containsKey("metrics"));
            check("snapshot global_step=10", ((Number) snap.get("global_step")).intValue() == 10);
        });
    }

    // ── D2 ───────────────────────────────────────────────────────────────────
    static void d2BoardStart() throws Exception {
        section("D2 LlamaBoard HTTP server starts");
        benchmark("LlamaBoard.start returns handle", () -> {
            try (LlamaBoard board = LlamaBoard.start(0)) {
                check("port > 0", board.port() > 0);
                check("uiUrl non-null", board.uiUrl() != null);
                check("uiUrl contains port", board.uiUrl().contains(String.valueOf(board.port())));
                check("state() non-null", board.state() != null);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        benchmark("GET /health", () -> {
            try (LlamaBoard board = LlamaBoard.start(0)) {
                String url = "http://127.0.0.1:" + board.port() + "/health";
                String body = get(url);
                check("health returns ok", body.contains("\"ok\":true"));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    // ── D3 ───────────────────────────────────────────────────────────────────
    static void d3BoardStateApi() throws Exception {
        section("D3 GET /api/state");
        benchmark("GET /api/state returns snapshot", () -> {
            try (LlamaBoard board = LlamaBoard.start(0)) {
                String url = "http://127.0.0.1:" + board.port() + "/api/state";
                String body = get(url);
                check("api/state has status", body.contains("status"));
                check("api/state has global_step", body.contains("global_step"));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    // ── D4 ───────────────────────────────────────────────────────────────────
    static void d4BoardRunsApi() throws Exception {
        section("D4 GET /api/runs");
        benchmark("GET /api/runs returns list", () -> {
            try (LlamaBoard board = LlamaBoard.start(0)) {
                String url = "http://127.0.0.1:" + board.port() + "/api/runs";
                String body = get(url);
                check("api/runs has runs", body.contains("runs"));
            }catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    // ── D5 ───────────────────────────────────────────────────────────────────
    static void d5ApiAuth() {
        section("D5 ApiAuth");
        benchmark("ApiAuth disabled (null key)", () -> {
            ApiAuth auth = ApiAuth.disabled();
            check("enabled=false", !auth.enabled());
            check("allow all", auth.allow(null, null));
            check("allow bearer", auth.allow("Bearer abc", null));
            check("allow x-key", auth.allow(null, "abc"));
        });

        benchmark("ApiAuth enabled strict", () -> {
            ApiAuth auth = new ApiAuth("secret-key-123");
            check("enabled=true", auth.enabled());
            check("reject empty", !auth.allow(null, null));
            check("reject wrong", !auth.allow("Bearer wrong", null));
            check("reject wrong x-key", !auth.allow(null, "wrong"));
            check("accept correct bearer", auth.allow("Bearer secret-key-123", null));
            check("accept correct x-key", auth.allow(null, "secret-key-123"));
        });

        benchmark("ApiAuth challenge", () -> {
            ApiAuth auth = new ApiAuth("key");
            check("has challenge", auth.challenge().isPresent());
            check("challenge is Bearer", auth.challenge().get().equals("Bearer"));
        });
    }

    // ── D6 ───────────────────────────────────────────────────────────────────
    static void d6OpenAiTypesRoundtrip() {
        section("D6 OpenAiTypes roundtrip");
        benchmark("ChatCompletionRequest fromMap", () -> {
            java.util.Map<String, Object> body = java.util.Map.of(
                    "model", "tiny-gpt2",
                    "messages", java.util.List.of(
                            java.util.Map.of("role", "user", "content", "hello")),
                    "temperature", 0.7,
                    "max_tokens", 64,
                    "stream", false);
            OpenAiTypes.ChatCompletionRequest req = OpenAiTypes.ChatCompletionRequest.fromMap(body);
            check("model=tiny-gpt2", "tiny-gpt2".equals(req.model));
            check("messages non-empty", !req.messages.isEmpty());
            check("temperature=0.7", Math.abs(req.temperature - 0.7) < 1e-9);
            check("maxTokens=64", req.maxTokens == 64);
            check("stream=false", !req.stream);
            check("first msg role=user", "user".equals(req.messages.get(0).role));
            check("first msg content=hello", "hello".equals(req.messages.get(0).content));
        });

        benchmark("chatCompletionResponse", () -> {
            java.util.Map<String, Object> resp = OpenAiTypes.chatCompletionResponse(
                    "tiny-gpt2", "hello world", 5, 3);
            check("resp has id", resp.containsKey("id"));
            check("resp has choices", resp.containsKey("choices"));
            check("resp has usage", resp.containsKey("usage"));
            check("resp has created", resp.containsKey("created"));
            java.util.Map<String, Object> usage = (java.util.Map<String, Object>) resp.get("usage");
            check("usage total_tokens=8", ((Number) usage.get("total_tokens")).intValue() == 8);
        });

        benchmark("CompletionRequest fromMap", () -> {
            java.util.Map<String, Object> body = java.util.Map.of(
                    "prompt", "say hello", "temperature", 0.9, "max_tokens", 32);
            OpenAiTypes.CompletionRequest req = OpenAiTypes.CompletionRequest.fromMap(body);
            check("prompt non-empty", !req.prompt.isEmpty());
            check("temperature=0.9", Math.abs(req.temperature - 0.9) < 1e-9);
        });

        benchmark("modelsList", () -> {
            java.util.Map<String, Object> resp = OpenAiTypes.modelsList("tiny-gpt2");
            check("resp has data", resp.containsKey("data"));
            java.util.List<?> data = (java.util.List<?>) resp.get("data");
            check("data non-empty", !data.isEmpty());
            java.util.Map<String, Object> m = (java.util.Map<String, Object>) data.get(0);
            check("model id=tiny-gpt2", "tiny-gpt2".equals(m.get("id")));
        });
    }

    // ── D7 ───────────────────────────────────────────────────────────────────
    static void d7BoardStop() throws Exception {
        section("D7 BoardState cooperative cancel");
        benchmark("BoardState.requestStop", () -> {
            BoardState s = new BoardState();
            check("stopRequested initially false", !s.stopRequested());
            s.requestStop();
            check("stopRequested=true after requestStop", s.stopRequested());
        });

        benchmark("BoardState clearStop", () -> {
            BoardState s = new BoardState();
            s.requestStop();
            s.clearStop();
            check("stopRequested=false after clearStop", !s.stopRequested());
        });
    }

    // ── D8 ───────────────────────────────────────────────────────────────────
    static void d8BoardClose() throws Exception {
        section("D8 LlamaBoard close is clean");
        benchmark("LlamaBoard.close no-throw", () -> {
            LlamaBoard board = null;
            try {
                board = LlamaBoard.start(0);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            board.close(); // should not throw
            check("board closed ok", true);
        });
    }

    // ── helpers ───────────────────────────────────────────────────────────────
    static String get(String url) throws Exception {
        HttpRequest req = HttpRequest.newBuilder()
                .uri(java.net.URI.create(url))
                .timeout(Duration.ofSeconds(10))
                .GET()
                .build();
        HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
        return resp.body();
    }

    static void section(String n) { System.out.println("\n=== " + n + " ==="); }
    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  PASS  " + name); }
        else { failed++; failures.add(name); System.out.println("  FAIL  " + name); }
    }
    static void benchmark(String name, Runnable r) {
        try { r.run(); }
        catch (Throwable t) { failed++; failures.add(name);
            System.out.println("  EXC   " + name + " — " + t.getMessage()); }
    }
    static void done() {
        System.out.println("\n=== RESULT ===");
        System.out.println("PASSED : " + passed);
        System.out.println("FAILED : " + failed);
        if (!failures.isEmpty()) {
            System.out.println("FAILURES:");
            for (String f : failures) System.out.println("  " + f);
        }
        if (failed > 0) throw new RuntimeException(failed + " tests failed");
    }
}
