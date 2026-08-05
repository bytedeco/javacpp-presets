package org.bytedeco.pytorch.plot.vista;

import java.awt.Desktop;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import org.bytedeco.pytorch.utils.json.Json;

/**
 * Interactive HTML renderer for {@link TraceGraph}.
 *
 * <p>Features: LR / RL / TB / BT layout, zoom buttons, multi-theme (cute /
 * dark / office), flowing dashed edges, export SVG / PNG / JPEG / PDF via
 * browser print, pan-drag / wheel-zoom / collapse / inspect.
 *
 * <p><b>Critical:</b> never animate CSS {@code transform} on SVG node groups —
 * that overrides {@code transform="translate(x,y)"} and collapses all but the
 * first node. Pop-in uses opacity only.
 */
public final class VistaRender {
    private VistaRender() {}

    public static Path plotGraph(TraceGraph graph, VistaOptions options) throws IOException {
        if (graph == null) throw new IllegalArgumentException("graph is null");
        if (options == null) options = VistaOptions.defaults();

        ExportFormat fmt = options.exportFormat();
        if (fmt == null && options.exportPath() != null) fmt = ExportFormat.HTML;
        if (fmt == ExportFormat.PNG || fmt == ExportFormat.SVG) {
            // Still write interactive HTML; in-page Export menu covers static formats.
            System.err.println("[vista] writing interactive HTML (use in-page Export for SVG/PNG/JPEG/PDF).");
            fmt = ExportFormat.HTML;
        }
        if (fmt == null) fmt = ExportFormat.HTML;

        String uniqueId = UUID.randomUUID().toString().replace("-", "");
        String html = buildHtml(graph, options, uniqueId);
        Path out = resolveOutputPath(options.exportPath(), uniqueId);
        Files.createDirectories(out.getParent() != null ? out.getParent() : Paths.get("."));
        Files.write(out, html.getBytes(StandardCharsets.UTF_8));
        System.out.println("[vista] wrote interactive graph → " + out.toAbsolutePath());
        if (options.openBrowser()) openInBrowser(out);
        return out;
    }

    public static String buildHtml(TraceGraph graph, VistaOptions options, String uniqueId) {
        if (uniqueId == null || uniqueId.isEmpty()) {
            uniqueId = UUID.randomUUID().toString().replace("-", "");
        }
        if (options == null) options = VistaOptions.defaults();

        Map<String, Object> payload = graph.toJsonPayload();
        payload.put("collapse_modules_after_depth", options.collapseModulesAfterDepth());
        payload.put("show_module_attr_names", options.showModuleAttrNames());
        payload.put("show_modular_view", options.showCompressedView());
        payload.put("height", options.height());
        payload.put("width", options.width());
        if (graph.exception() != null) {
            // Short, user-facing message — never dump native stack into the HTML banner
            String msg = graph.exception().getMessage();
            if (msg == null || msg.isEmpty()) msg = graph.exception().getClass().getSimpleName();
            else {
                // first line only, trim "Exception raised from..." dumps
                int nl = msg.indexOf('\n');
                if (nl > 0) msg = msg.substring(0, nl);
                if (msg.length() > 220) msg = msg.substring(0, 217) + "…";
            }
            payload.put("error_message",
                    graph.exception().getClass().getSimpleName() + ": " + msg);
        }
        String json = Json.encode(payload).replace("</", "<\\/");
        String widthCss = options.width() == null ? "100%" : (options.width() + "px");
        int height = Math.max(560, options.height());

        StringBuilder sb = new StringBuilder(160 * 1024);
        sb.append("<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"/>");
        sb.append("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>");
        sb.append("<title>jnitorch vista · model graph</title>");
        sb.append("<style>");
        appendCss(sb, uniqueId, widthCss, height);
        sb.append("</style></head><body data-theme=\"cute\">");

        sb.append("<div id=\"app-").append(uniqueId).append("\" class=\"vx-app\">");
        // ── toolbar ────────────────────────────────────────────────────────
        sb.append("<header class=\"vx-top\">");
        sb.append("<div class=\"vx-brand\">");
        sb.append("<span class=\"vx-logo\" id=\"logo-").append(uniqueId).append("\" aria-hidden=\"true\">🍡</span>");
        sb.append("<div><div class=\"vx-title\">Vista <span class=\"vx-sparkle\">✦</span></div>");
        sb.append("<div class=\"vx-sub\">layout · zoom · style · export · themes</div></div></div>");
        sb.append("<div class=\"vx-stats\" id=\"stats-").append(uniqueId).append("\"></div>");
        sb.append("<div class=\"vx-actions\">");
        // direction
        sb.append("<div class=\"vx-group\" title=\"Layout direction\">");
        sb.append("<button type=\"button\" class=\"vx-btn vx-dir active\" data-dir=\"LR\" id=\"dir-LR-").append(uniqueId).append("\">→ LR</button>");
        sb.append("<button type=\"button\" class=\"vx-btn vx-dir\" data-dir=\"RL\" id=\"dir-RL-").append(uniqueId).append("\">← RL</button>");
        sb.append("<button type=\"button\" class=\"vx-btn vx-dir\" data-dir=\"TB\" id=\"dir-TB-").append(uniqueId).append("\">↓ TB</button>");
        sb.append("<button type=\"button\" class=\"vx-btn vx-dir\" data-dir=\"BT\" id=\"dir-BT-").append(uniqueId).append("\">↑ BT</button>");
        sb.append("</div>");
        // zoom
        sb.append("<div class=\"vx-group\">");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"zoomin-").append(uniqueId).append("\" title=\"Zoom in\">＋</button>");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"zoomout-").append(uniqueId).append("\" title=\"Zoom out\">－</button>");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"fit-").append(uniqueId).append("\" title=\"Fit\">⊞ Fit</button>");
        sb.append("</div>");
        // expand/collapse/reset
        sb.append("<div class=\"vx-group\">");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"exp-").append(uniqueId).append("\">⊕ Expand</button>");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"col-").append(uniqueId).append("\">⊖ Collapse</button>");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"rst-").append(uniqueId).append("\">↺ Reset</button>");
        sb.append("</div>");
        // theme
        sb.append("<div class=\"vx-group\">");
        sb.append("<button type=\"button\" class=\"vx-btn vx-theme active\" data-theme=\"cute\" id=\"th-cute-").append(uniqueId).append("\">🍡 Cute</button>");
        sb.append("<button type=\"button\" class=\"vx-btn vx-theme\" data-theme=\"dark\" id=\"th-dark-").append(uniqueId).append("\">🌙 Dark</button>");
        sb.append("<button type=\"button\" class=\"vx-btn vx-theme\" data-theme=\"office\" id=\"th-office-").append(uniqueId).append("\">💼 Office</button>");
        sb.append("</div>");
        // export
        sb.append("<div class=\"vx-group vx-export-wrap\">");
        sb.append("<button type=\"button\" class=\"vx-btn vx-btn-accent\" id=\"export-").append(uniqueId).append("\">⇩ Export ▾</button>");
        sb.append("<div class=\"vx-menu\" id=\"export-menu-").append(uniqueId).append("\" hidden>");
        sb.append("<button type=\"button\" data-fmt=\"svg\">SVG 矢量</button>");
        sb.append("<button type=\"button\" data-fmt=\"png\">PNG 图片</button>");
        sb.append("<button type=\"button\" data-fmt=\"jpeg\">JPEG 图片</button>");
        sb.append("<button type=\"button\" data-fmt=\"pdf\">PDF 文档</button>");
        sb.append("</div></div>");
        sb.append("</div></header>");

        sb.append("<div id=\"err-").append(uniqueId).append("\" class=\"vx-err\" hidden></div>");
        sb.append("<div class=\"vx-toast\" id=\"toast-").append(uniqueId).append("\" hidden></div>");

        sb.append("<div class=\"vx-body\" style=\"min-height:").append(height).append("px\">");
        sb.append("<div class=\"vx-stage\" id=\"stage-").append(uniqueId).append("\">");
        sb.append("<div class=\"vx-bg-blobs\" aria-hidden=\"true\"></div>");
        sb.append("<svg id=\"svg-").append(uniqueId).append("\" xmlns=\"http://www.w3.org/2000/svg\">");
        sb.append("<defs>");
        sb.append("<marker id=\"arr-").append(uniqueId).append("\" viewBox=\"0 0 12 12\" refX=\"10\" refY=\"6\" markerWidth=\"9\" markerHeight=\"9\" orient=\"auto\">");
        sb.append("<path d=\"M1,1.5 L10,6 L1,10.5 Z\" class=\"vx-arrow-path\"></path></marker>");
        sb.append("<filter id=\"soft-").append(uniqueId).append("\" x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\">");
        sb.append("<feDropShadow dx=\"0\" dy=\"4\" stdDeviation=\"5\" flood-color=\"#64748b\" flood-opacity=\"0.18\"/>");
        sb.append("</filter>");
        sb.append("<filter id=\"glow-").append(uniqueId).append("\" x=\"-80%\" y=\"-80%\" width=\"260%\" height=\"260%\">");
        sb.append("<feDropShadow dx=\"0\" dy=\"0\" stdDeviation=\"6\" flood-color=\"#38bdf8\" flood-opacity=\"0.45\"/>");
        sb.append("</filter>");
        // Concrete stop colors (CSS vars on stop-color are unreliable in SVG paint servers).
        // JS setTheme() rewrites these on theme switch.
        sb.append("<linearGradient id=\"flow-").append(uniqueId).append("\" gradientUnits=\"userSpaceOnUse\" x1=\"0\" y1=\"0\" x2=\"400\" y2=\"0\">");
        sb.append("<stop offset=\"0%\" stop-color=\"#c4b5fd\"/><stop offset=\"50%\" stop-color=\"#67e8f9\"/><stop offset=\"100%\" stop-color=\"#fbcfe8\"/>");
        sb.append("</linearGradient>");
        sb.append("</defs>");
        sb.append("<g id=\"world-").append(uniqueId).append("\"></g>");
        sb.append("</svg>");
        sb.append("<div class=\"vx-hint\">方向切换 · ＋/－缩放 · 滚轮缩放 · 拖拽平移 · 点击检查 · ±折叠 · 导出</div>");
        sb.append("<div class=\"vx-legend\" id=\"legend-").append(uniqueId).append("\"></div>");
        sb.append("<div class=\"vx-zoom-label\" id=\"zoomlab-").append(uniqueId).append("\">100%</div>");
        sb.append("</div>");

        sb.append("<aside class=\"vx-drawer\" id=\"drawer-").append(uniqueId).append("\">");
        sb.append("<div class=\"vx-drawer-head\"><span id=\"dtitle-").append(uniqueId).append("\">✨ Inspector</span>");
        sb.append("<button type=\"button\" class=\"vx-icon\" id=\"dclose-").append(uniqueId).append("\">✕</button></div>");
        sb.append("<div class=\"vx-chips\" id=\"dchips-").append(uniqueId).append("\"></div>");
        sb.append("<div class=\"vx-shape-row\" id=\"dshapes-").append(uniqueId).append("\"></div>");
        // Feature / label meta card
        sb.append("<div class=\"vx-feat\" id=\"dfeat-").append(uniqueId).append("\" hidden></div>");
        // Style editor — color + shape for selected component
        sb.append("<div class=\"vx-style\" id=\"dstyle-").append(uniqueId).append("\" hidden>");
        sb.append("<div class=\"vx-style-title\">🎨 组件样式</div>");
        sb.append("<div class=\"vx-style-label\">扁平柔美填充色（点选立即应用）</div>");
        sb.append("<div class=\"vx-swatches\" id=\"dswatch-").append(uniqueId).append("\"></div>");
        sb.append("<div class=\"vx-style-label\">深浅调节 <span id=\"ddepth-lab-").append(uniqueId).append("\">100%</span></div>");
        sb.append("<input type=\"range\" class=\"vx-depth\" id=\"ddepth-").append(uniqueId).append("\" min=\"40\" max=\"160\" value=\"100\" step=\"5\"/>");
        sb.append("<div class=\"vx-style-label\">形状（点选应用）</div>");
        sb.append("<div class=\"vx-shapes\" id=\"dshape-").append(uniqueId).append("\"></div>");
        sb.append("<button type=\"button\" class=\"vx-btn\" id=\"dstyle-rst-").append(uniqueId).append("\" style=\"margin-top:8px;width:100%\">↺ 恢复默认样式</button>");
        sb.append("</div>");
        sb.append("<pre class=\"vx-pre\" id=\"dbody-").append(uniqueId).append("\">点选节点查看特征 / 超参 / shape · 可改颜色与形状</pre>");
        sb.append("</aside></div>");

        sb.append("<footer class=\"vx-foot\">jnitorch <b>utils.vista</b> · non-invasive structure expand · ");
        sb.append("<a href=\"https://github.com/sachinhosmani/torchvista\" target=\"_blank\" rel=\"noopener\">torchvista schema</a></footer>");
        sb.append("</div>");

        sb.append("<script type=\"application/json\" id=\"data-").append(uniqueId).append("\">");
        sb.append(json);
        sb.append("</script><script>\n");
        appendJs(sb, uniqueId);
        sb.append("\n</script></body></html>\n");
        return sb.toString();
    }

    private static void appendCss(StringBuilder sb, String id, String widthCss, int height) {
        // ── themes ─────────────────────────────────────────────────────────
        // cute (default)
        sb.append(":root, [data-theme=cute]{");
        sb.append("--bg:#faf7ff;--bg2:#ffffff;--ink:#2d2640;--muted:#8b83a0;--line:#efe8f8;");
        sb.append("--card:#ffffff;--accent:#9b87f5;--accent2:#ff9ec8;--cyan:#7dd3fc;");
        sb.append("--good:#6ee7b7;--warn:#fcd34d;--bad:#fda4af;");
        sb.append("--shadow:0 10px 28px rgba(155,135,245,.14);");
        sb.append("--flow1:#c4b5fd;--flow2:#67e8f9;--flow3:#fbcfe8;");
        sb.append("--logo-bg:linear-gradient(145deg,#ffd6ea,#d4c4ff 55%,#c8f0ff);");
        sb.append("--btn-accent:linear-gradient(135deg,#ffb3d4,#c4b5fd);");
        sb.append("--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;");
        sb.append("--sans:\"Segoe UI\",ui-rounded,system-ui,-apple-system,Roboto,Helvetica,Arial,\"PingFang SC\",\"Noto Sans SC\",sans-serif}");
        // dark purple
        sb.append("[data-theme=dark]{--bg:#14121c;--bg2:#1c1828;--ink:#f3eefc;--muted:#a89bb8;--line:#2c2540;");
        sb.append("--card:#221c32;--accent:#a78bfa;--accent2:#f0abfc;--cyan:#22d3ee;");
        sb.append("--good:#34d399;--warn:#fbbf24;--bad:#fb7185;");
        sb.append("--shadow:0 12px 32px rgba(0,0,0,.45);");
        sb.append("--flow1:#a78bfa;--flow2:#22d3ee;--flow3:#f0abfc;");
        sb.append("--logo-bg:linear-gradient(145deg,#4c1d95,#7c3aed 55%,#db2777);");
        sb.append("--btn-accent:linear-gradient(135deg,#7c3aed,#db2777)}");
        // office / internet workplace — clean blue-gray, Inter-like
        sb.append("[data-theme=office]{--bg:#f5f7fa;--bg2:#ffffff;--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;");
        sb.append("--card:#ffffff;--accent:#2563eb;--accent2:#0ea5e9;--cyan:#06b6d4;");
        sb.append("--good:#10b981;--warn:#f59e0b;--bad:#ef4444;");
        sb.append("--shadow:0 4px 16px rgba(15,23,42,.08);");
        sb.append("--flow1:#60a5fa;--flow2:#38bdf8;--flow3:#818cf8;");
        sb.append("--logo-bg:linear-gradient(145deg,#2563eb,#0ea5e9);");
        sb.append("--btn-accent:linear-gradient(135deg,#2563eb,#0ea5e9)}");

        sb.append("*{box-sizing:border-box}");
        sb.append("body{margin:0;font-family:var(--sans);background:var(--bg);color:var(--ink);");
        sb.append("background-image:radial-gradient(900px 500px at 8% -8%,color-mix(in srgb,var(--accent2) 18%,transparent),transparent 55%),");
        sb.append("radial-gradient(800px 480px at 100% 0%,color-mix(in srgb,var(--accent) 14%,transparent),transparent 50%)}");
        sb.append(".vx-app{width:").append(widthCss).append(";margin:0 auto;min-height:100vh;display:flex;flex-direction:column}");
        sb.append(".vx-top{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;padding:10px 14px;");
        sb.append("position:sticky;top:0;z-index:40;backdrop-filter:blur(14px);");
        sb.append("background:color-mix(in srgb,var(--bg) 84%,transparent);border-bottom:1px solid var(--line)}");
        sb.append(".vx-brand{display:flex;gap:10px;align-items:center;min-width:140px}");
        sb.append(".vx-logo{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-size:20px;");
        sb.append("background:var(--logo-bg);box-shadow:var(--shadow);color:#fff}");
        sb.append(".vx-title{font-weight:800;letter-spacing:.02em;font-size:15px}");
        sb.append(".vx-sparkle{color:var(--accent2)}");
        sb.append(".vx-sub{font-size:10px;color:var(--muted);font-weight:600}");
        sb.append(".vx-stats{display:flex;gap:6px;flex-wrap:wrap;font-size:11px;font-weight:700;color:var(--muted)}");
        sb.append(".vx-stat{background:var(--bg2);border:1px solid var(--line);border-radius:999px;padding:4px 9px}");
        sb.append(".vx-stat b{color:var(--accent);margin-left:3px}");
        sb.append(".vx-actions{display:flex;flex-wrap:wrap;gap:6px;align-items:center}");
        sb.append(".vx-group{display:flex;gap:3px;padding:3px;border:1px solid var(--line);border-radius:12px;background:color-mix(in srgb,var(--bg2) 80%,transparent)}");
        sb.append(".vx-btn{border:1px solid transparent;background:transparent;color:var(--ink);border-radius:9px;");
        sb.append("padding:6px 10px;font-size:11px;font-weight:700;cursor:pointer;transition:.15s ease;white-space:nowrap}");
        sb.append(".vx-btn:hover{background:color-mix(in srgb,var(--accent) 12%,var(--bg2));border-color:color-mix(in srgb,var(--accent) 30%,var(--line))}");
        sb.append(".vx-btn:active{transform:scale(.97)}");
        sb.append(".vx-btn.active{background:color-mix(in srgb,var(--accent) 18%,var(--bg2));color:var(--accent);border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}");
        sb.append(".vx-btn-accent{background:var(--btn-accent);color:#fff;border:none}");
        sb.append(".vx-btn-accent:hover{filter:brightness(1.05);border:none}");
        sb.append(".vx-export-wrap{position:relative}");
        // CRITICAL: do NOT set display:flex on the base rule — it overrides the HTML
        // [hidden] attribute (hidden uses display:none but loses to later display:flex).
        sb.append(".vx-menu{position:absolute;right:0;top:calc(100% + 6px);min-width:148px;background:var(--card);");
        sb.append("border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:6px;z-index:50;");
        sb.append("flex-direction:column;gap:2px}");
        sb.append(".vx-menu:not([hidden]){display:flex}");
        sb.append(".vx-menu[hidden]{display:none !important}");
        sb.append(".vx-menu button{border:0;background:transparent;text-align:left;padding:8px 12px;border-radius:8px;");
        sb.append("font-size:12px;font-weight:700;color:var(--ink);cursor:pointer;font-family:var(--sans);width:100%}");
        sb.append(".vx-menu button:hover{background:color-mix(in srgb,var(--accent) 12%,var(--bg2))}");
        sb.append(".vx-err{margin:10px 18px;padding:10px 14px;border-radius:12px;background:#fff1f2;color:#9f1239;");
        sb.append("border:1px solid #fecdd3;font-size:13px;white-space:pre-wrap}");
        sb.append("[data-theme=dark] .vx-err{background:#3f1d24;color:#fecdd3;border-color:#7f1d1d}");
        sb.append(".vx-toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:100;");
        sb.append("background:var(--ink);color:var(--bg2);padding:10px 18px;border-radius:999px;font-size:12px;font-weight:700;");
        sb.append("box-shadow:var(--shadow)}");
        sb.append(".vx-body{display:grid;grid-template-columns:1fr minmax(280px,340px);gap:0;flex:1;min-height:").append(height).append("px;");
        sb.append("transition:grid-template-columns .22s ease}");
        sb.append(".vx-body.drawer-closed{grid-template-columns:1fr 0px}");
        sb.append("@media(max-width:980px){.vx-body{grid-template-columns:1fr}.vx-drawer{max-height:40vh}");
        sb.append(".vx-body.drawer-closed .vx-drawer{max-height:0;border-top:0}}");
        sb.append(".vx-stage{position:relative;overflow:hidden;background:var(--bg2);");
        sb.append("background-image:radial-gradient(circle at 1px 1px, color-mix(in srgb,var(--accent) 12%,transparent) 1px, transparent 0);");
        sb.append("background-size:20px 20px}");
        sb.append(".vx-bg-blobs{pointer-events:none;position:absolute;inset:0;opacity:.55;");
        sb.append("background:radial-gradient(520px 260px at 18% 28%,color-mix(in srgb,var(--accent2) 14%,transparent),transparent),");
        sb.append("radial-gradient(480px 240px at 82% 72%,color-mix(in srgb,var(--accent) 12%,transparent),transparent)}");
        sb.append(".vx-stage svg{width:100%;height:100%;display:block;cursor:grab;touch-action:none}");
        sb.append(".vx-stage svg:active{cursor:grabbing}");
        sb.append(".vx-hint{position:absolute;left:12px;bottom:12px;font-size:10px;color:var(--muted);font-weight:600;");
        sb.append("background:color-mix(in srgb,var(--card) 92%,transparent);backdrop-filter:blur(8px);");
        sb.append("border:1px solid var(--line);border-radius:999px;padding:5px 11px;pointer-events:none;z-index:5}");
        sb.append(".vx-legend{position:absolute;right:12px;bottom:12px;display:flex;flex-wrap:wrap;gap:5px;max-width:48%;");
        sb.append("justify-content:flex-end;pointer-events:none;z-index:5}");
        sb.append(".vx-leg{font-size:9px;font-weight:800;border-radius:999px;padding:3px 8px;border:1px solid var(--line);");
        sb.append("background:var(--card);color:var(--ink);opacity:.92}");
        sb.append(".vx-zoom-label{position:absolute;left:12px;top:12px;font-size:11px;font-weight:800;color:var(--muted);");
        sb.append("background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px 8px;z-index:5;pointer-events:none}");
        sb.append(".vx-drawer{border-left:1px solid var(--line);background:color-mix(in srgb,var(--card) 94%,transparent);");
        sb.append("backdrop-filter:blur(16px);display:flex;flex-direction:column;min-height:0;overflow:hidden;");
        sb.append("min-width:0;opacity:1;transition:opacity .18s ease,border-color .18s ease}");
        sb.append(".vx-drawer.closed{opacity:0;pointer-events:none;border-left-color:transparent}");
        sb.append(".vx-drawer-head{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;");
        sb.append("border-bottom:1px solid var(--line);font-weight:800;flex-shrink:0}");
        sb.append(".vx-icon{border:0;background:transparent;color:var(--muted);font-size:15px;cursor:pointer;border-radius:8px;padding:4px 8px}");
        sb.append(".vx-icon:hover{background:var(--line)}");
        sb.append(".vx-chips{display:flex;flex-wrap:wrap;gap:6px;padding:10px 14px 0}");
        sb.append(".vx-chip{font-size:10px;font-weight:800;border-radius:999px;padding:3px 9px;border:1px solid var(--line);");
        sb.append("background:color-mix(in srgb,var(--accent) 10%,var(--bg2))}");
        sb.append(".vx-shape-row{display:flex;flex-wrap:wrap;gap:6px;padding:8px 14px 0}");
        sb.append(".vx-shape-pill{font-size:10px;font-weight:700;font-family:var(--mono);border-radius:10px;");
        sb.append("padding:5px 9px;border:1px dashed color-mix(in srgb,var(--accent) 35%,var(--line));");
        sb.append("background:color-mix(in srgb,var(--cyan) 8%,var(--bg2));color:var(--ink)}");
        sb.append(".vx-pre{margin:0;padding:12px 14px 20px;font-family:var(--mono);font-size:11px;line-height:1.55;");
        sb.append("white-space:pre-wrap;word-break:break-word;color:var(--ink);overflow:auto;flex:1}");
        // Feature / label meta card
        sb.append(".vx-feat{margin:8px 14px;padding:10px 12px;border-radius:12px;border:1px solid var(--line);");
        sb.append("background:color-mix(in srgb,var(--accent) 6%,var(--bg2));font-size:11px;line-height:1.55}");
        sb.append(".vx-feat[hidden]{display:none !important}");
        sb.append(".vx-feat-head{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-weight:800}");
        sb.append(".vx-feat-badge{font-size:9px;font-weight:900;letter-spacing:.04em;border-radius:999px;");
        sb.append("padding:2px 8px;color:#fff}");
        sb.append(".vx-feat-badge.sparse{background:#6366f1}.vx-feat-badge.dense{background:#06b6d4}");
        sb.append(".vx-feat-badge.sequence{background:#ec4899}.vx-feat-badge.label{background:#3b82f6}");
        sb.append(".vx-feat-badge.feature{background:#8b5cf6}.vx-feat-badge.input{background:#10b981}");
        sb.append(".vx-feat-row{display:flex;justify-content:space-between;gap:8px;padding:2px 0;");
        sb.append("border-bottom:1px dashed color-mix(in srgb,var(--line) 80%,transparent)}");
        sb.append(".vx-feat-row:last-child{border-bottom:0}.vx-feat-k{color:var(--muted);font-weight:700}");
        sb.append(".vx-feat-v{font-family:var(--mono);font-weight:700;color:var(--ink);text-align:right}");
        // Style editor
        sb.append(".vx-style{margin:8px 14px;padding:10px 12px;border-radius:12px;border:1px solid var(--line);");
        sb.append("background:var(--bg2)}");
        sb.append(".vx-style[hidden]{display:none !important}");
        sb.append(".vx-style-title{font-weight:800;font-size:12px;margin-bottom:6px}");
        sb.append(".vx-style-label{font-size:10px;font-weight:700;color:var(--muted);margin:8px 0 4px}");
        sb.append(".vx-swatches{display:flex;flex-wrap:wrap;gap:6px}");
        sb.append(".vx-swatch{width:26px;height:26px;border-radius:9px;border:2px solid rgba(0,0,0,.08);cursor:pointer;");
        sb.append("box-shadow:0 2px 6px rgba(0,0,0,.12);position:relative}");
        sb.append(".vx-swatch:hover{transform:scale(1.1)}.vx-swatch.active{border-color:var(--ink);outline:2px solid var(--accent);outline-offset:1px}");
        sb.append(".vx-depth{width:100%;accent-color:var(--accent);cursor:pointer;margin:4px 0 2px}");
        sb.append(".vx-shapes{display:flex;flex-wrap:wrap;gap:5px}");
        sb.append(".vx-shape-btn{border:1px solid var(--line);background:var(--bg2);border-radius:8px;");
        sb.append("padding:4px 8px;font-size:10px;font-weight:700;cursor:pointer;color:var(--ink)}");
        sb.append(".vx-shape-btn:hover{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 10%,var(--bg2))}");
        sb.append(".vx-shape-btn.active{border-color:var(--accent);color:var(--accent);background:color-mix(in srgb,var(--accent) 14%,var(--bg2))}");
        sb.append(".vx-foot{position:sticky;bottom:0;z-index:30;padding:6px 14px;font-size:11px;color:var(--muted);");
        sb.append("background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(10px);");
        sb.append("border-top:1px solid var(--line);text-align:center;flex-shrink:0}");
        sb.append(".vx-foot a{color:var(--accent);text-decoration:none;font-weight:700}");

        // SVG nodes — NO css transform on .vn
        sb.append(".vn{cursor:pointer}");
        sb.append(".vn .card{filter:url(#soft-").append(id).append(");transition:stroke .15s,fill .15s,filter .2s}");
        sb.append(".vn:hover .card{filter:url(#glow-").append(id).append(")}");
        sb.append(".vn.selected .card{stroke-width:3 !important;filter:url(#glow-").append(id).append(")}");
        // Failed: red stroke only — do NOT force fill (user color edits must stick)
        sb.append(".vn.failed .card{stroke:#fb7185 !important;stroke-width:2.6 !important}");
        sb.append(".vn text{pointer-events:none;font-family:var(--sans)}");
        sb.append(".vn .t1{font-size:12.5px;font-weight:800;fill:var(--ink)}");
        sb.append(".vn .t2{font-size:9.5px;font-weight:700;fill:var(--muted)}");
        sb.append(".vn .t3{font-size:9px;font-weight:700;fill:var(--accent);font-family:var(--mono)}");
        sb.append(".vn .badge{font-size:8.5px;font-weight:900}");
        sb.append(".vn .pm{font-size:14px;font-weight:900;fill:var(--accent)}");
        sb.append(".vn.pop{animation:vx-fade .4s ease both}");
        sb.append("@keyframes vx-fade{from{opacity:0}to{opacity:1}}");
        sb.append(".vn.pulse .card{animation:vx-pulse-stroke .85s ease}");
        sb.append("@keyframes vx-pulse-stroke{0%{stroke-width:1.6}40%{stroke-width:3.2}100%{stroke-width:2.2}}");

        // edges — dual layer: solid base + flowing dashed overlay (visible inside expanded frames too)
        sb.append(".ve-base{fill:none;stroke:var(--accent);stroke-width:1.35;stroke-linecap:round;");
        sb.append("opacity:.28;pointer-events:none}");
        sb.append(".ve-base.implied{opacity:.12;stroke-dasharray:2 5}");
        sb.append(".ve{fill:none;stroke:url(#flow-").append(id).append(");stroke-width:2.35;stroke-linecap:round;");
        sb.append("marker-end:url(#arr-").append(id).append(");opacity:.98;");
        sb.append("stroke-dasharray:8 7;animation:vx-dash .85s linear infinite}");
        sb.append(".ve.implied{opacity:.5;stroke-dasharray:2 5;animation-duration:1.4s}");
        sb.append(".ve.hi{stroke-width:3.4;opacity:1;filter:drop-shadow(0 0 4px var(--accent))}");
        sb.append(".ve-base.hi{opacity:.55;stroke-width:2.2}");
        sb.append("@keyframes vx-dash{to{stroke-dashoffset:-30}}");
        sb.append(".ve-label{fill:var(--accent);font-size:9.5px;font-weight:800;font-family:var(--mono);pointer-events:none}");
        sb.append(".ve-label-bg{fill:var(--bg2);stroke:var(--line);stroke-width:1;opacity:.94}");
        sb.append(".vx-arrow-path{fill:var(--accent)}");
        // container frames — pinned state solid border + stronger fill
        sb.append(".vc-frame{fill:color-mix(in srgb,var(--accent) 5%,transparent);stroke:color-mix(in srgb,var(--accent) 28%,var(--line));");
        sb.append("stroke-width:1.35;stroke-dasharray:5 4;cursor:grab}");
        sb.append(".vc-frame.pinned{stroke:var(--accent);stroke-width:2;stroke-dasharray:none;");
        sb.append("fill:color-mix(in srgb,var(--accent) 9%,transparent)}");
        sb.append(".vc-label{font-size:10.5px;font-weight:800;fill:var(--accent);pointer-events:none}");
        sb.append(".vc-pin{font-size:11px;pointer-events:none}");
        sb.append(".vc-group{cursor:grab}");
        sb.append(".vc-group:active{cursor:grabbing}");
        // PDF / print — hide chrome, expand stage to full page
        sb.append("@media print{");
        sb.append("@page{size:A4 landscape;margin:8mm}");
        sb.append("html,body{width:100% !important;height:auto !important;overflow:visible !important;");
        sb.append("background:#fff !important;background-image:none !important}");
        sb.append(".vx-top,.vx-drawer,.vx-hint,.vx-legend,.vx-zoom-label,.vx-foot,.vx-err,.vx-toast,");
        sb.append(".vx-style,.vx-feat{display:none !important}");
        sb.append(".vx-app{width:100% !important;min-height:0 !important}");
        sb.append(".vx-body{display:block !important;height:auto !important;min-height:0 !important;");
        sb.append("border:0 !important;grid-template-columns:1fr !important}");
        sb.append(".vx-stage{overflow:visible !important;height:auto !important;min-height:0 !important;");
        sb.append("background:#fff !important;background-image:none !important;position:relative !important}");
        sb.append(".vx-stage svg{width:100% !important;height:auto !important;max-height:none !important;");
        sb.append("display:block !important}");
        sb.append(".vx-bg-blobs{display:none !important}");
        sb.append("/* animation off for clean PDF */");
        sb.append(".ve{animation:none !important;stroke-dashoffset:0 !important}");
        sb.append("}");
    }

    private static void appendJs(StringBuilder s, String id) {
        s.append("(function(){\n'use strict';\n");
        s.append("const ID='").append(id).append("';\n");
        s.append("const raw=JSON.parse(document.getElementById('data-'+ID).textContent);\n");
        s.append("const adj=raw.adj_list||{}, moduleInfo=raw.module_info||{}, funcInfo=raw.func_info||{};\n");
        s.append("const displayNames=raw.graph_node_display_names||{}, bareNames=raw.graph_node_name_to_without_suffix||{};\n");
        s.append("const attrNames=raw.node_to_attr_name||{}, ancestorMap=raw.ancestor_map||{};\n");
        s.append("const parentToNodes=raw.parent_module_to_nodes||{}, parentDepth=raw.parent_module_to_depth||{};\n");
        s.append("const pathMap=raw.node_to_module_path||{};\n");
        s.append("const nodeMeta=raw.node_meta||{};\n");
        s.append("const collapseDepth=(raw.collapse_modules_after_depth|0);\n");
        s.append("const showAttr=!!raw.show_module_attr_names;\n");
        s.append("const stage=document.getElementById('stage-'+ID);\n");
        s.append("const svg=document.getElementById('svg-'+ID);\n");
        s.append("const world=document.getElementById('world-'+ID);\n");
        s.append("const dtitle=document.getElementById('dtitle-'+ID);\n");
        s.append("const dbody=document.getElementById('dbody-'+ID);\n");
        s.append("const dchips=document.getElementById('dchips-'+ID);\n");
        s.append("const dshapes=document.getElementById('dshapes-'+ID);\n");
        s.append("const dfeat=document.getElementById('dfeat-'+ID);\n");
        s.append("const dstyle=document.getElementById('dstyle-'+ID);\n");
        s.append("const dswatch=document.getElementById('dswatch-'+ID);\n");
        s.append("const dshape=document.getElementById('dshape-'+ID);\n");
        s.append("const ddepth=document.getElementById('ddepth-'+ID);\n");
        s.append("const ddepthLab=document.getElementById('ddepth-lab-'+ID);\n");
        s.append("const drawer=document.getElementById('drawer-'+ID);\n");
        s.append("const bodyEl=drawer?drawer.parentElement:null;\n");
        s.append("const errBox=document.getElementById('err-'+ID);\n");
        s.append("const statsEl=document.getElementById('stats-'+ID);\n");
        s.append("const legendEl=document.getElementById('legend-'+ID);\n");
        s.append("const zoomLab=document.getElementById('zoomlab-'+ID);\n");
        s.append("const toastEl=document.getElementById('toast-'+ID);\n");
        s.append("const logoEl=document.getElementById('logo-'+ID);\n");
        s.append("const exportMenu=document.getElementById('export-menu-'+ID);\n");
        s.append("if(raw.error_message){errBox.hidden=false;errBox.textContent='⚠️ '+raw.error_message;}\n");

        s.append("let transform={x:28,y:36,k:1}, selected=null, selectedName=null;\n");
        s.append("let orient='LR'; // LR | RL | TB | BT\n");
        s.append("const collapsed=new Set(), pos={};\n");
        s.append("const framePos={};\n");
        // Per-node style overrides {fill,stroke,badge,shape} — soft flat defaults below
        s.append("const styleOverride={};\n");
        s.append("const NW=200, NH=78, HG=90, VG=70;\n");
        s.append("const THEMES={cute:'🍡',dark:'🌙',office:'💼'};\n");
        // Flat soft pastel palette (12)
        s.append("const PALETTE=[\n");
        s.append("  {fill:'#eef2ff',stroke:'#a5b4fc',badge:'#6366f1',name:'Indigo'},\n");
        s.append("  {fill:'#fdf2f8',stroke:'#f9a8d4',badge:'#ec4899',name:'Pink'},\n");
        s.append("  {fill:'#ecfeff',stroke:'#67e8f9',badge:'#06b6d4',name:'Cyan'},\n");
        s.append("  {fill:'#f0fdf4',stroke:'#86efac',badge:'#22c55e',name:'Mint'},\n");
        s.append("  {fill:'#fff7ed',stroke:'#fdba74',badge:'#f97316',name:'Peach'},\n");
        s.append("  {fill:'#faf5ff',stroke:'#d8b4fe',badge:'#a855f7',name:'Lilac'},\n");
        s.append("  {fill:'#fff1f2',stroke:'#fda4af',badge:'#f43f5e',name:'Rose'},\n");
        s.append("  {fill:'#eff6ff',stroke:'#93c5fd',badge:'#3b82f6',name:'Sky'},\n");
        s.append("  {fill:'#fefce8',stroke:'#fde047',badge:'#eab308',name:'Butter'},\n");
        s.append("  {fill:'#f8fafc',stroke:'#cbd5e1',badge:'#64748b',name:'Slate'},\n");
        s.append("  {fill:'#ecfdf5',stroke:'#6ee7b7',badge:'#10b981',name:'Emerald'},\n");
        s.append("  {fill:'#f5f3ff',stroke:'#c4b5fd',badge:'#8b5cf6',name:'Violet'}\n");
        s.append("];\n");
        // 14 shapes (original 4 + 10 new)
        s.append("const SHAPES=['round','pill','stadium','diamond','hex','octagon','parallelogram','trapezoid','chevron','cylinder','cloud','shield','arrow','notch'];\n");
        s.append("const SHAPE_LABEL={round:'圆角',pill:'胶囊',stadium:'跑道',diamond:'菱形',hex:'六边',octagon:'八边',parallelogram:'平行四边',trapezoid:'梯形',chevron:'箭头块',cylinder:'圆柱',cloud:'云朵',shield:'盾牌',arrow:'指向',notch:'缺口'};\n");

        s.append("const childrenOf={};\n");
        s.append("Object.keys(ancestorMap).forEach(c=>{const p=ancestorMap[c]; if(p)(childrenOf[p]=childrenOf[p]||[]).push(c);});\n");
        s.append("const containers=new Set([...Object.keys(parentToNodes),...Object.values(ancestorMap).filter(Boolean),...Object.keys(childrenOf)]);\n");
        s.append("function containerDepth(n){if(parentDepth[n]!=null)return parentDepth[n]|0;let d=0,c=n,g=0;while(ancestorMap[c]&&g++<64){d++;c=ancestorMap[c];}return d;}\n");
        s.append("containers.forEach(c=>{const m=parentToNodes[c]||childrenOf[c]||[]; if(m.length&&containerDepth(c)>collapseDepth) collapsed.add(c);});\n");

        s.append("function isHidden(n){let c=ancestorMap[n],g=0;while(c&&g++<64){if(collapsed.has(c))return true;c=ancestorMap[c];}return false;}\n");
        s.append("function visibleNodes(){const vis=new Set();Object.keys(adj).forEach(n=>{if(!isHidden(n))vis.add(n);});\n");
        s.append("  collapsed.forEach(c=>{if(isHidden(c))return; const m=parentToNodes[c]||[]; if(m.some(x=>adj[x]||displayNames[x]))vis.add(c);}); return vis;}\n");
        s.append("function rep(n){let c=n,g=0;while(g++<64){const p=ancestorMap[c]; if(p&&collapsed.has(p)){c=p;continue;} break;}\n");
        s.append("  while(isHidden(c)&&ancestorMap[c]&&g++<64)c=ancestorMap[c]; return c;}\n");
        s.append("function labelOf(n){if(showAttr&&attrNames[n])return attrNames[n]; if(displayNames[n])return displayNames[n]; if(bareNames[n])return bareNames[n]; return n;}\n");
        s.append("function typeOf(n){if(adj[n])return adj[n].node_type||'Module'; return 'Module';}\n");
        s.append("function bareOf(n){return bareNames[n]||labelOf(n)||n;}\n");

        s.append("function familyOf(n){\n");
        s.append("  const t=typeOf(n);\n");
        s.append("  const meta=nodeMeta[n]||{};\n");
        s.append("  const ft=(meta.feature_type||'').toLowerCase();\n");
        s.append("  if(t==='Input'){\n");
        s.append("    if(ft==='sparse') return 'feat_sparse';\n");
        s.append("    if(ft==='dense') return 'feat_dense';\n");
        s.append("    if(ft==='sequence'||ft==='seq') return 'feat_seq';\n");
        s.append("    if(ft==='label') return 'feat_label';\n");
        s.append("    return 'input';\n");
        s.append("  }\n");
        s.append("  if(t==='Output') return ft==='label'?'feat_label':'output';\n");
        s.append("  if(t==='Operation') return 'op'; if(t==='Constant') return 'const'; if(t==='Parameter') return 'param';\n");
        s.append("  const b=(bareOf(n)+' '+(moduleInfo[n]&&moduleInfo[n].type||'')+' '+(pathMap[n]||'')).toLowerCase();\n");
        s.append("  if(/linear|meta.?linear|fc|dense|lm_head|proj/.test(b)) return 'linear';\n");
        s.append("  if(/conv|embedding|embed|token_emb|pos_emb/.test(b)) return 'embed';\n");
        s.append("  if(/relu|gelu|silu|sigmoid|tanh|softmax|activation|act/.test(b)) return 'act';\n");
        s.append("  if(/norm|batchnorm|layernorm|groupnorm|final_norm/.test(b)) return 'norm';\n");
        s.append("  if(/dropout|drop/.test(b)) return 'drop';\n");
        s.append("  if(/attention|attn|mha|multihead|q_layer|k_layer|v_layer/.test(b)) return 'attn';\n");
        s.append("  if(/lstm|gru|rnn|transformer|hstu|decoder|encoder/.test(b)) return 'rnn';\n");
        s.append("  if(/sequential|modulelist|moduledict|container|mlp|weightbag|residual|sharedbottom|mmoe|esmm|ple|cgc|omoe|aitm|onerec|tiger|rqvae|hllm/.test(b)) return 'container';\n");
        s.append("  if(collapsed.has(n)||containers.has(n)) return 'container';\n");
        s.append("  return 'module';\n");
        s.append("}\n");

        s.append("const FAMILY={\n");
        s.append("  input:       {fill:'#ecfdf5',stroke:'#6ee7b7',badge:'#10b981',badgeText:'IN',  shape:'stadium', emoji:'📥'},\n");
        s.append("  feat_sparse: {fill:'#eef2ff',stroke:'#a5b4fc',badge:'#6366f1',badgeText:'SPR', shape:'hex',     emoji:'🔢'},\n");
        s.append("  feat_dense:  {fill:'#ecfeff',stroke:'#67e8f9',badge:'#06b6d4',badgeText:'DNS', shape:'round',   emoji:'📊'},\n");
        s.append("  feat_seq:    {fill:'#fdf2f8',stroke:'#f9a8d4',badge:'#ec4899',badgeText:'SEQ', shape:'chevron', emoji:'📜'},\n");
        s.append("  feat_label:  {fill:'#eff6ff',stroke:'#93c5fd',badge:'#3b82f6',badgeText:'LBL', shape:'shield',  emoji:'🏷'},\n");
        s.append("  output:      {fill:'#eff6ff',stroke:'#93c5fd',badge:'#3b82f6',badgeText:'OUT', shape:'stadium', emoji:'📤'},\n");
        s.append("  op:          {fill:'#faf5ff',stroke:'#d8b4fe',badge:'#a855f7',badgeText:'OP',  shape:'diamond', emoji:'⚡'},\n");
        s.append("  const:       {fill:'#fffbeb',stroke:'#fcd34d',badge:'#f59e0b',badgeText:'C',   shape:'hex',     emoji:'📌'},\n");
        s.append("  param:       {fill:'#fdf4ff',stroke:'#f0abfc',badge:'#d946ef',badgeText:'P',   shape:'hex',     emoji:'🎛'},\n");
        s.append("  linear:      {fill:'#eef2ff',stroke:'#a5b4fc',badge:'#6366f1',badgeText:'LIN', shape:'round',   emoji:'📐'},\n");
        s.append("  embed:       {fill:'#ecfeff',stroke:'#67e8f9',badge:'#06b6d4',badgeText:'EMB', shape:'round',   emoji:'🧩'},\n");
        s.append("  act:         {fill:'#fff1f2',stroke:'#fda4af',badge:'#f43f5e',badgeText:'ACT', shape:'pill',    emoji:'✨'},\n");
        s.append("  norm:        {fill:'#f0fdf4',stroke:'#86efac',badge:'#22c55e',badgeText:'NRM', shape:'round',   emoji:'📏'},\n");
        s.append("  drop:        {fill:'#fff7ed',stroke:'#fdba74',badge:'#f97316',badgeText:'DRP', shape:'pill',    emoji:'🌧'},\n");
        s.append("  attn:        {fill:'#f5f3ff',stroke:'#c4b5fd',badge:'#8b5cf6',badgeText:'ATT', shape:'hex',     emoji:'👁'},\n");
        s.append("  rnn:         {fill:'#fdf2f8',stroke:'#f9a8d4',badge:'#ec4899',badgeText:'RNN', shape:'round',   emoji:'🔁'},\n");
        s.append("  container:   {fill:'#f8fafc',stroke:'#cbd5e1',badge:'#64748b',badgeText:'GRP', shape:'round',   emoji:'📦'},\n");
        s.append("  module:      {fill:'#f5f3ff',stroke:'#c4b5fd',badge:'#7c3aed',badgeText:'MOD', shape:'round',   emoji:'🧱'}\n");
        s.append("};\n");
        s.append("function styleOf(n, failed){\n");
        s.append("  const base=FAMILY[familyOf(n)]||FAMILY.module;\n");
        s.append("  const ov=styleOverride[n]||{};\n");
        s.append("  // Start from family defaults, then apply user override (fill/stroke/badge/shape/depth)\n");
        s.append("  let st=Object.assign({}, base);\n");
        s.append("  if(ov.fill) st.fill=ov.fill;\n");
        s.append("  if(ov.stroke) st.stroke=ov.stroke;\n");
        s.append("  if(ov.badge) st.badge=ov.badge;\n");
        s.append("  if(ov.shape) st.shape=ov.shape;\n");
        s.append("  // Depth: 40–160% brightness on fill/stroke/badge (100 = original)\n");
        s.append("  const depth = (ov.depth!=null) ? (+ov.depth) : 100;\n");
        s.append("  if(depth!==100){\n");
        s.append("    st.fill=shadeHex(st.fill, depth);\n");
        s.append("    st.stroke=shadeHex(st.stroke, Math.min(160, depth+10));\n");
        s.append("    st.badge=shadeHex(st.badge, Math.max(40, depth-5));\n");
        s.append("  }\n");
        s.append("  st._depth=depth;\n");
        s.append("  // Failed keeps red stroke but PRESERVES user fill if set (so color edits still show)\n");
        s.append("  if(failed){\n");
        s.append("    st=Object.assign({}, st, {stroke:'#fb7185', badgeText: st.badgeText||'FAIL'});\n");
        s.append("    if(!ov.fill) st.fill='#fff1f2';\n");
        s.append("    if(!ov.badge) st.badge='#e11d48';\n");
        s.append("  }\n");
        s.append("  return st;\n");
        s.append("}\n");
        // shadeHex: depth 100 = identity; <100 lighten toward white; >100 darken toward black\n
        s.append("function shadeHex(hex, depth){\n");
        s.append("  if(!hex || typeof hex!=='string') return hex;\n");
        s.append("  let h=hex.trim(); if(h[0]==='#') h=h.slice(1);\n");
        s.append("  if(h.length===3) h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];\n");
        s.append("  if(h.length!==6 || /[^0-9a-fA-F]/.test(h)) return hex;\n");
        s.append("  let r=parseInt(h.slice(0,2),16), g=parseInt(h.slice(2,4),16), b=parseInt(h.slice(4,6),16);\n");
        s.append("  const d=Math.max(40, Math.min(160, +depth||100))/100;\n");
        s.append("  if(d<1){ // lighten\n");
        s.append("    r=Math.round(r+(255-r)*(1-d)); g=Math.round(g+(255-g)*(1-d)); b=Math.round(b+(255-b)*(1-d));\n");
        s.append("  } else if(d>1){ // darken\n");
        s.append("    r=Math.round(r*(2-d)); g=Math.round(g*(2-d)); b=Math.round(b*(2-d));\n");
        s.append("  }\n");
        s.append("  const clamp=v=>Math.max(0,Math.min(255,v));\n");
        s.append("  const to=v=>clamp(v).toString(16).padStart(2,'0');\n");
        s.append("  return '#'+to(r)+to(g)+to(b);\n");
        s.append("}\n");
        s.append("function metaLine(n){\n");
        s.append("  const m=nodeMeta[n]; if(!m) return '';\n");
        s.append("  const bits=[];\n");
        s.append("  if(m.feature_type) bits.push(String(m.feature_type).toUpperCase());\n");
        s.append("  if(m.shape) bits.push(m.shape);\n");
        s.append("  else if(m.vocab_size!=null) bits.push('V='+m.vocab_size);\n");
        s.append("  if(m.embed_dim!=null) bits.push('E='+m.embed_dim);\n");
        s.append("  if(m.pooling) bits.push(m.pooling);\n");
        s.append("  if(m.label&&m.kind==='output') bits.push('y='+m.label);\n");
        s.append("  return bits.join(' · ');\n");
        s.append("}\n");

        s.append("function inDims(n){const a=adj[n]; return (a&&a.original_incoming_dims)||[];}\n");
        s.append("function outDims(n){const a=adj[n]; return (a&&a.original_outgoing_dims)||[];}\n");
        s.append("function primaryDim(arr){if(!arr||!arr.length)return ''; const u=[...new Set(arr.filter(Boolean))]; return u[0]||'';}\n");
        s.append("function hyperLine(n){\n");
        s.append("  const mi=moduleInfo[n]; if(!mi) return '';\n");
        s.append("  const a=mi.attributes||{}; const keys=Object.keys(a).filter(k=>k!=='kind'&&k!=='class');\n");
        s.append("  if(!keys.length){const ps=mi.parameters||{}; const pk=Object.keys(ps); if(!pk.length)return '';\n");
        s.append("    return pk.slice(0,2).map(k=>{const sh=(ps[k].shape||[]).join('×'); return k+(sh?(' '+sh):'');}).join(' · ');}\n");
        s.append("  return keys.slice(0,3).map(k=>k.replace(/_features|_dim/,'')+'='+a[k]).join(' · ');\n");
        s.append("}\n");

        // ── layout with orientation — every rank centered on cross-axis ────
        s.append("function layout(vis){\n");
        s.append("  const nodes=[...vis], edges=[], seen=new Set();\n");
        s.append("  Object.keys(adj).forEach(src=>{const sRep=rep(src); if(!vis.has(sRep))return;\n");
        s.append("    (adj[src].edges||[]).forEach(e=>{const tRep=rep(e.target); if(!vis.has(tRep)||sRep===tRep)return;\n");
        s.append("      const k=sRep+'->'+tRep+'|'+(e.dims||'')+'|'+(e.edge_data_id||''); if(seen.has(k))return; seen.add(k);\n");
        s.append("      edges.push({s:sRep,t:tRep,dims:e.dims||'',implied:!!e.is_implied_edge});});});\n");
        s.append("  const indeg={}, outs={}; nodes.forEach(n=>{indeg[n]=0;outs[n]=[];});\n");
        s.append("  edges.forEach(e=>{if(indeg[e.t]!=null&&indeg[e.s]!=null){indeg[e.t]++; outs[e.s].push(e.t);}});\n");
        s.append("  const level={}, q=[]; nodes.forEach(n=>{if(!indeg[n]){level[n]=0;q.push(n);}});\n");
        s.append("  while(q.length){const n=q.shift(); (outs[n]||[]).forEach(t=>{const nl=(level[n]||0)+1; if(level[t]==null||nl>level[t])level[t]=nl; indeg[t]--; if(indeg[t]===0)q.push(t);});}\n");
        s.append("  nodes.forEach(n=>{if(level[n]==null)level[n]=0;});\n");
        s.append("  let maxL=0; nodes.forEach(n=>{if(level[n]>maxL)maxL=level[n];});\n");
        s.append("  const by={}; nodes.forEach(n=>{(by[level[n]]=by[level[n]]||[]).push(n);});\n");
        s.append("  const isVert=(orient==='TB'||orient==='BT');\n");
        s.append("  const rev=(orient==='RL'||orient==='BT');\n");
        s.append("  const stepCross = isVert ? (NW+HG*0.55) : (NH+VG);\n");
        s.append("  let maxCross=0;\n");
        s.append("  const placed={};\n");
        s.append("  Object.keys(by).map(Number).sort((a,b)=>a-b).forEach(l=>{\n");
        s.append("    const row=by[l].slice().sort((a,b)=>{\n");
        s.append("      const ra=typeOf(a)==='Input'?-1:typeOf(a)==='Output'?1:0;\n");
        s.append("      const rb=typeOf(b)==='Input'?-1:typeOf(b)==='Output'?1:0;\n");
        s.append("      if(ra!==rb) return ra-rb;\n");
        // Natural sort: handles numeric suffixes like "0","1","2","10" correctly
        s.append("      const la=String(labelOf(a)), lb=String(labelOf(b));\n");
        s.append("      return la.localeCompare(lb, undefined, {numeric:true, sensitivity:'base'});\n");
        s.append("    });\n");
        s.append("    const depth = rev ? (maxL - l) : l;\n");
        s.append("    const crossSpan = Math.max(0, row.length*stepCross - (isVert?HG*0.55:VG));\n");
        s.append("    if(crossSpan>maxCross) maxCross=crossSpan;\n");
        s.append("    placed[l]={row, depth, crossSpan};\n");
        s.append("  });\n");
        // Ensure a minimum canvas cross-span so single-node ranks look centered in view
        s.append("  const minCanvas = isVert ? Math.max(maxCross, NW*2.2) : Math.max(maxCross, NH*3);\n");
        s.append("  if(minCanvas>maxCross) maxCross=minCanvas;\n");
        s.append("  Object.keys(placed).map(Number).sort((a,b)=>a-b).forEach(l=>{\n");
        s.append("    const {row, depth, crossSpan}=placed[l];\n");
        s.append("    // DEFAULT CENTER: every rank centered on the cross-axis\n");
        s.append("    const offset = Math.max(0, (maxCross - crossSpan) / 2);\n");
        s.append("    row.forEach((n,i)=>{\n");
        s.append("      let x,y;\n");
        s.append("      if(isVert){ x=40+offset+i*stepCross; y=40+depth*(NH+VG); }\n");
        s.append("      else { x=40+depth*(NW+HG); y=48+offset+i*stepCross; }\n");
        // Only auto-place if not user-pinned (_auto!==false means free to relayout)
        s.append("      if(!pos[n]) pos[n]={x,y,_auto:true};\n");
        s.append("      else if(pos[n]._auto!==false){ pos[n].x=x; pos[n].y=y; }\n");
        s.append("    });\n");
        s.append("  });\n");
        s.append("  return edges;}\n");

        s.append("function applyT(){world.setAttribute('transform','translate('+transform.x+','+transform.y+') scale('+transform.k+')');\n");
        s.append("  zoomLab.textContent=Math.round(transform.k*100)+'%';}\n");

        s.append("function drawShape(g, st, failed){\n");
        s.append("  const shape=st.shape||'round'; let el;\n");
        s.append("  const ns='http://www.w3.org/2000/svg';\n");
        s.append("  const W=NW, H=NH, cx=W/2, cy=H/2;\n");
        s.append("  function poly(pts){ const p=document.createElementNS(ns,'polygon'); p.setAttribute('points', pts); return p; }\n");
        s.append("  function rect(x,y,w,h,rx,ry){ const r=document.createElementNS(ns,'rect');\n");
        s.append("    r.setAttribute('x',x); r.setAttribute('y',y); r.setAttribute('width',w); r.setAttribute('height',h);\n");
        s.append("    if(rx!=null) r.setAttribute('rx',rx); if(ry!=null) r.setAttribute('ry',ry); return r; }\n");
        s.append("  if(shape==='diamond'){ el=poly(cx+',3 '+(W-3)+','+cy+' '+cx+','+(H-3)+' 3,'+cy); }\n");
        s.append("  else if(shape==='hex'){ const x=7,y=5,w=W-14,h=H-10,cut=13;\n");
        s.append("    el=poly((x+cut)+','+y+' '+(x+w-cut)+','+y+' '+(x+w)+','+(y+h/2)+' '+(x+w-cut)+','+(y+h)+' '+(x+cut)+','+(y+h)+' '+x+','+(y+h/2)); }\n");
        s.append("  else if(shape==='octagon'){ const x=6,y=4,w=W-12,h=H-8,c=12;\n");
        s.append("    el=poly((x+c)+','+y+' '+(x+w-c)+','+y+' '+(x+w)+','+(y+c)+' '+(x+w)+','+(y+h-c)+' '+(x+w-c)+','+(y+h)+' '+(x+c)+','+(y+h)+' '+x+','+(y+h-c)+' '+x+','+(y+c)); }\n");
        s.append("  else if(shape==='parallelogram'){ el=poly('22,4 '+(W-4)+',4 '+(W-22)+','+(H-4)+' 4,'+(H-4)); }\n");
        s.append("  else if(shape==='trapezoid'){ el=poly('28,5 '+(W-28)+',5 '+(W-6)+','+(H-5)+' 6,'+(H-5)); }\n");
        s.append("  else if(shape==='chevron'){ el=poly('4,4 '+(W-22)+',4 '+W+','+cy+' '+(W-22)+','+(H-4)+' 4,'+(H-4)+' 22,'+cy); }\n");
        s.append("  else if(shape==='arrow'){ el=poly('4,14 '+(W-28)+',14 '+(W-28)+',4 '+W+','+cy+' '+(W-28)+','+(H-4)+' '+(W-28)+','+(H-14)+' 4,'+(H-14)); }\n");
        s.append("  else if(shape==='notch'){ el=poly('4,4 '+(W-4)+',4 '+(W-4)+','+(H-4)+' 4,'+(H-4)+' 18,'+cy); }\n");
        s.append("  else if(shape==='shield'){ el=poly(cx+',4 '+(W-8)+',14 '+(W-12)+','+(H*0.55)+' '+cx+','+(H-4)+' 12,'+(H*0.55)+' 8,14'); }\n");
        s.append("  else if(shape==='cylinder'){\n");
        s.append("    const g2=document.createElementNS(ns,'g');\n");
        s.append("    const body=rect(8,14,W-16,H-28,0,0);\n");
        s.append("    body.setAttribute('class','card'); body.setAttribute('fill',st.fill); body.setAttribute('stroke',st.stroke);\n");
        s.append("    body.style.fill=st.fill; body.style.stroke=st.stroke;\n");
        s.append("    body.setAttribute('stroke-width', failed?2.6:1.65); g2.appendChild(body);\n");
        s.append("    const top=document.createElementNS(ns,'ellipse'); top.setAttribute('cx',cx); top.setAttribute('cy',14);\n");
        s.append("    top.setAttribute('rx',(W-16)/2); top.setAttribute('ry',10); top.setAttribute('class','card');\n");
        s.append("    top.setAttribute('fill',st.fill); top.setAttribute('stroke',st.stroke); top.style.fill=st.fill; top.style.stroke=st.stroke;\n");
        s.append("    top.setAttribute('stroke-width', failed?2.6:1.65); g2.appendChild(top);\n");
        s.append("    const bot=document.createElementNS(ns,'path');\n");
        s.append("    bot.setAttribute('d','M8,'+(H-14)+' A'+(W-16)/2+' 10 0 0 0 '+(W-8)+' '+(H-14));\n");
        s.append("    bot.setAttribute('fill','none'); bot.setAttribute('stroke',st.stroke); bot.setAttribute('stroke-width', failed?2.6:1.65); g2.appendChild(bot);\n");
        s.append("    g.appendChild(g2); return body;\n");
        s.append("  }\n");
        s.append("  else if(shape==='cloud'){\n");
        s.append("    el=document.createElementNS(ns,'path');\n");
        s.append("    el.setAttribute('d','M46,52 C28,52 18,40 26,28 C22,14 40,8 54,16 C62,6 86,8 90,22 C108,20 118,36 106,48 C100,58 60,60 46,52 Z');\n");
        s.append("  }\n");
        s.append("  else if(shape==='stadium'||shape==='pill'){ el=rect(2,2,W-4,H-4, shape==='stadium'?(H/2-2):20, shape==='stadium'?(H/2-2):20); }\n");
        s.append("  else { el=rect(1,1,W-2,H-2,14,14); }\n");
        s.append("  el.setAttribute('class','card');\n");
        s.append("  el.setAttribute('fill', st.fill);\n");
        s.append("  el.setAttribute('stroke', st.stroke);\n");
        s.append("  // Also set presentation style so user overrides always win over CSS\n");
        s.append("  el.style.fill=st.fill; el.style.stroke=st.stroke;\n");
        s.append("  el.setAttribute('stroke-width', failed?2.6:1.65); g.appendChild(el); return el;\n");
        s.append("}\n");

        s.append("function edgePath(a,b){\n");
        s.append("  // connect from center-exit of a toward center-entry of b depending on orient\n");
        s.append("  let x1,y1,x2,y2,c1x,c1y,c2x,c2y;\n");
        s.append("  if(orient==='LR'){ x1=a.x+NW; y1=a.y+NH/2; x2=b.x; y2=b.y+NH/2; c1x=(x1+x2)/2; c1y=y1; c2x=(x1+x2)/2; c2y=y2; }\n");
        s.append("  else if(orient==='RL'){ x1=a.x; y1=a.y+NH/2; x2=b.x+NW; y2=b.y+NH/2; c1x=(x1+x2)/2; c1y=y1; c2x=(x1+x2)/2; c2y=y2; }\n");
        s.append("  else if(orient==='TB'){ x1=a.x+NW/2; y1=a.y+NH; x2=b.x+NW/2; y2=b.y; c1x=x1; c1y=(y1+y2)/2; c2x=x2; c2y=(y1+y2)/2; }\n");
        s.append("  else { x1=a.x+NW/2; y1=a.y; x2=b.x+NW/2; y2=b.y+NH; c1x=x1; c1y=(y1+y2)/2; c2x=x2; c2y=(y1+y2)/2; }\n");
        s.append("  return {d:'M'+x1+','+y1+' C'+c1x+','+c1y+' '+c2x+','+c2y+' '+x2+','+y2, mx:(x1+x2)/2, my:(y1+y2)/2};\n");
        s.append("}\n");

        s.append("function render(){\n");
        s.append("  const vis=visibleNodes(); const edges=layout(vis);\n");
        s.append("  while(world.firstChild) world.removeChild(world.firstChild);\n");
        s.append("  const frameLayer=document.createElementNS('http://www.w3.org/2000/svg','g');\n");
        s.append("  const edgeLayer=document.createElementNS('http://www.w3.org/2000/svg','g');\n");
        s.append("  const nodeLayer=document.createElementNS('http://www.w3.org/2000/svg','g');\n");
        s.append("  world.appendChild(frameLayer); world.appendChild(edgeLayer); world.appendChild(nodeLayer);\n");

        // container frames — draggable & pin-fixed after user drag
        s.append("  Object.keys(parentToNodes).forEach(c=>{\n");
        s.append("    if(collapsed.has(c)||isHidden(c)) return;\n");
        s.append("    const members=(parentToNodes[c]||[]).map(rep).filter(m=>vis.has(m)&&pos[m]);\n");
        s.append("    if(members.length<1) return;\n");
        s.append("    let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;\n");
        s.append("    members.forEach(m=>{const p=pos[m]; minX=Math.min(minX,p.x); minY=Math.min(minY,p.y); maxX=Math.max(maxX,p.x+NW); maxY=Math.max(maxY,p.y+NH);});\n");
        s.append("    const pad=20;\n");
        s.append("    // Auto-compute frame bbox from members; if user pinned the frame, keep its origin\n");
        s.append("    let fx=minX-pad, fy=minY-pad-10;\n");
        s.append("    const fw=maxX-minX+pad*2, fh=maxY-minY+pad*2+10;\n");
        s.append("    if(!framePos[c]) framePos[c]={x:fx,y:fy,_auto:true,w:fw,h:fh};\n");
        s.append("    else if(framePos[c]._auto!==false){ framePos[c].x=fx; framePos[c].y=fy; framePos[c].w=fw; framePos[c].h=fh; }\n");
        s.append("    else { // pinned: keep origin, refresh size to wrap members relative offset\n");
        s.append("      framePos[c].w=fw; framePos[c].h=fh;\n");
        s.append("      fx=framePos[c].x; fy=framePos[c].y;\n");
        // When frame is pinned, shift member nodes that are still auto so they stay inside
        s.append("      const dx=fx-(minX-pad), dy=fy-(minY-pad-10);\n");
        s.append("      if(dx||dy){ members.forEach(m=>{ if(pos[m]&&pos[m]._auto!==false){ pos[m].x+=dx; pos[m].y+=dy; } }); }\n");
        s.append("    }\n");
        s.append("    const fg=document.createElementNS('http://www.w3.org/2000/svg','g');\n");
        s.append("    fg.setAttribute('class','vc-group');\n");
        s.append("    fg.setAttribute('data-frame', c);\n");
        s.append("    fg.style.cursor='grab';\n");
        s.append("    const fr=document.createElementNS('http://www.w3.org/2000/svg','rect');\n");
        s.append("    fr.setAttribute('class','vc-frame'+(framePos[c]._auto===false?' pinned':''));\n");
        s.append("    fr.setAttribute('x', framePos[c].x); fr.setAttribute('y', framePos[c].y);\n");
        s.append("    fr.setAttribute('width', framePos[c].w); fr.setAttribute('height', framePos[c].h);\n");
        s.append("    fr.setAttribute('rx',16); fr.setAttribute('ry',16); fg.appendChild(fr);\n");
        s.append("    const tl=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("    tl.setAttribute('class','vc-label'); tl.setAttribute('x', framePos[c].x+8); tl.setAttribute('y', framePos[c].y+14);\n");
        s.append("    tl.textContent=(framePos[c]._auto===false?'📌 ':'📦 ')+(labelOf(c)||c); fg.appendChild(tl);\n");
        // pin badge
        s.append("    const pin=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("    pin.setAttribute('class','vc-pin'); pin.setAttribute('x', framePos[c].x+framePos[c].w-10); pin.setAttribute('y', framePos[c].y+14);\n");
        s.append("    pin.setAttribute('text-anchor','end');\n");
        s.append("    pin.textContent=framePos[c]._auto===false?'🔒':'🔓'; fg.appendChild(pin);\n");
        s.append("    enableFrameDrag(fg, c, members); frameLayer.appendChild(fg);\n");
        s.append("  });\n");

        // edges — dual layer: solid base (always visible) + flowing dashed overlay
        s.append("  edges.forEach(e=>{\n");
        s.append("    const a=pos[e.s], b=pos[e.t]; if(!a||!b) return;\n");
        s.append("    const ep=edgePath(a,b);\n");
        s.append("    const base=document.createElementNS('http://www.w3.org/2000/svg','path');\n");
        s.append("    base.setAttribute('class','ve-base'+(e.implied?' implied':''));\n");
        s.append("    base.setAttribute('d', ep.d);\n");
        s.append("    base.setAttribute('data-s', e.s); base.setAttribute('data-t', e.t);\n");
        s.append("    edgeLayer.appendChild(base);\n");
        s.append("    const p=document.createElementNS('http://www.w3.org/2000/svg','path');\n");
        s.append("    p.setAttribute('class','ve'+(e.implied?' implied':''));\n");
        s.append("    p.setAttribute('d', ep.d);\n");
        s.append("    p.setAttribute('data-s', e.s); p.setAttribute('data-t', e.t);\n");
        s.append("    edgeLayer.appendChild(p);\n");
        s.append("    if(e.dims){\n");
        s.append("      const tw=Math.min(120, 8+e.dims.length*6);\n");
        s.append("      const bg=document.createElementNS('http://www.w3.org/2000/svg','rect');\n");
        s.append("      bg.setAttribute('class','ve-label-bg'); bg.setAttribute('x', ep.mx-tw/2); bg.setAttribute('y', ep.my-10);\n");
        s.append("      bg.setAttribute('width', tw); bg.setAttribute('height', 15); bg.setAttribute('rx',6); edgeLayer.appendChild(bg);\n");
        s.append("      const t=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("      t.setAttribute('class','ve-label'); t.setAttribute('x', ep.mx); t.setAttribute('y', ep.my+1);\n");
        s.append("      t.setAttribute('text-anchor','middle'); t.textContent=e.dims; edgeLayer.appendChild(t);\n");
        s.append("    }\n");
        s.append("  });\n");

        // nodes
        s.append("  let idx=0;\n");
        s.append("  vis.forEach(n=>{\n");
        s.append("    const p=pos[n]; if(!p) return;\n");
        s.append("    const g=document.createElementNS('http://www.w3.org/2000/svg','g');\n");
        s.append("    g.setAttribute('class','vn pop'+(selectedName===n?' selected':''));\n");
        s.append("    g.style.animationDelay=(Math.min(idx,16)*0.02)+'s';\n");
        s.append("    g.setAttribute('transform','translate('+p.x+','+p.y+')');\n");
        s.append("    g.setAttribute('data-node', n);\n");
        s.append("    const failed=!!(adj[n]&&adj[n].failed); if(failed) g.classList.add('failed');\n");
        s.append("    const st=styleOf(n, failed);\n");
        s.append("    drawShape(g, st, failed);\n");
        s.append("    if(st.shape==='round'||st.shape==='pill'||st.shape==='stadium'||st.shape==='cylinder'){\n");
        s.append("      const tick=document.createElementNS('http://www.w3.org/2000/svg','rect');\n");
        s.append("      tick.setAttribute('x',6); tick.setAttribute('y',12); tick.setAttribute('width',3.5); tick.setAttribute('height',NH-24);\n");
        s.append("      tick.setAttribute('rx',2); tick.setAttribute('fill', st.badge); g.appendChild(tick);}\n");
        s.append("    const br=document.createElementNS('http://www.w3.org/2000/svg','rect');\n");
        s.append("    br.setAttribute('x',12); br.setAttribute('y',7); br.setAttribute('width',40); br.setAttribute('height',13);\n");
        s.append("    br.setAttribute('rx',7); br.setAttribute('fill', st.badge); g.appendChild(br);\n");
        s.append("    const bt=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("    bt.setAttribute('class','badge'); bt.setAttribute('x',32); bt.setAttribute('y',17);\n");
        s.append("    bt.setAttribute('text-anchor','middle'); bt.setAttribute('fill','#fff'); bt.textContent=st.badgeText; g.appendChild(bt);\n");
        s.append("    const t1=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("    t1.setAttribute('class','t1'); t1.setAttribute('x',12); t1.setAttribute('y',36);\n");
        s.append("    const metaN=nodeMeta[n]||{};\n");
        s.append("    const titleName=(metaN.name||metaN.label||labelOf(n));\n");
        s.append("    const lab=(st.emoji?st.emoji+' ':'')+titleName;\n");
        s.append("    t1.textContent=lab.length>20?lab.slice(0,19)+'…':lab; g.appendChild(t1);\n");
        s.append("    const od=primaryDim(outDims(n))||primaryDim(inDims(n));\n");
        s.append("    const idm=primaryDim(inDims(n));\n");
        s.append("    let shapeTxt='';\n");
        s.append("    if(metaN.shape) shapeTxt=String(metaN.shape);\n");
        s.append("    else if(idm&&od&&idm!==od) shapeTxt=idm+' → '+od;\n");
        s.append("    else if(od) shapeTxt='out '+od; else if(idm) shapeTxt='in '+idm;\n");
        s.append("    const t3=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("    t3.setAttribute('class','t3'); t3.setAttribute('x',12); t3.setAttribute('y',50);\n");
        s.append("    t3.textContent=shapeTxt||typeOf(n); g.appendChild(t3);\n");
        s.append("    const hl=hyperLine(n); const ml=metaLine(n);\n");
        s.append("    const t2=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("    t2.setAttribute('class','t2'); t2.setAttribute('x',12); t2.setAttribute('y',64);\n");
        s.append("    let sub=ml||hl||(collapsed.has(n)?'collapsed · +':(containers.has(n)&&(parentToNodes[n]||[]).length?'group · −':familyOf(n)));\n");
        s.append("    if(failed) sub='FAILED · '+sub;\n");
        s.append("    t2.textContent=sub.length>28?sub.slice(0,27)+'…':sub; g.appendChild(t2);\n");
        s.append("    const members=parentToNodes[n]||childrenOf[n]||[];\n");
        s.append("    if(members.length||collapsed.has(n)){\n");
        s.append("      const pm=document.createElementNS('http://www.w3.org/2000/svg','text');\n");
        s.append("      pm.setAttribute('class','pm'); pm.setAttribute('x',NW-10); pm.setAttribute('y',20);\n");
        s.append("      pm.setAttribute('text-anchor','end'); pm.textContent=collapsed.has(n)?'＋':'－'; g.appendChild(pm);}\n");
        s.append("    enableDrag(g,n);\n");
        s.append("    g.addEventListener('click',ev=>{ev.stopPropagation(); onClick(n,g);});\n");
        s.append("    nodeLayer.appendChild(g); idx++;\n");
        s.append("  });\n");
        s.append("  applyT(); updateStats(vis, edges); updateLegend(vis); refreshFlowGradient();\n");
        s.append("}\n");

        s.append("function updateStats(vis, edges){\n");
        s.append("  const nN=vis.size, nE=edges.length;\n");
        s.append("  const ops=[...vis].filter(n=>typeOf(n)==='Operation').length;\n");
        s.append("  const mods=[...vis].filter(n=>typeOf(n)==='Module').length;\n");
        s.append("  statsEl.innerHTML='<span class=\"vx-stat\">nodes<b>'+nN+'</b></span><span class=\"vx-stat\">edges<b>'+nE+'</b></span>'+\n");
        s.append("    '<span class=\"vx-stat\">mod<b>'+mods+'</b></span><span class=\"vx-stat\">op<b>'+ops+'</b></span><span class=\"vx-stat\">'+orient+'</span>';\n");
        s.append("}\n");
        s.append("function updateLegend(vis){\n");
        s.append("  const seen=new Set(), bits=[];\n");
        s.append("  const LAB={feat_sparse:'稀疏特征',feat_dense:'稠密特征',feat_seq:'序列特征',feat_label:'Label',input:'输入',output:'输出'};\n");
        s.append("  vis.forEach(n=>{const f=familyOf(n); if(seen.has(f))return; seen.add(f);\n");
        s.append("    const st=FAMILY[f]||FAMILY.module;\n");
        s.append("    const name=LAB[f]||f;\n");
        s.append("    bits.push('<span class=\"vx-leg\" style=\"border-color:'+st.stroke+';background:'+st.fill+'\">'+st.emoji+' '+name+'</span>');});\n");
        s.append("  legendEl.innerHTML=bits.join('');\n");
        s.append("}\n");

        s.append("function onClick(n,g){\n");
        s.append("  const members=parentToNodes[n]||childrenOf[n]||[];\n");
        s.append("  if(members.length||collapsed.has(n)){\n");
        s.append("    if(collapsed.has(n)){\n");
        s.append("      collapsed.delete(n);\n");
        // Reset positions of newly visible children so they get properly re-laid out
        s.append("      const resetKids=parentToNodes[n]||childrenOf[n]||[];\n");
        s.append("      resetKids.forEach(m=>{ if(pos[m]) pos[m]._auto=true; });\n");
        s.append("      if(framePos[n]) framePos[n]._auto=true;\n");
        s.append("    } else { collapsed.add(n); }\n");
        s.append("    selectedName=n; render(); fit();\n");
        s.append("    const ng=world.querySelector('g[data-node=\"'+CSS.escape(n)+'\"]');\n");
        s.append("    if(ng){ ng.classList.add('selected','pulse'); selected=ng; }\n");
        s.append("  } else {\n");
        s.append("    if(selected) selected.classList.remove('selected','pulse');\n");
        s.append("    selected=g; selectedName=n; g.classList.add('selected','pulse');\n");
        s.append("  }\n");
        s.append("  highlightEdges(n); showInfo(n);\n");
        s.append("}\n");
        s.append("function highlightEdges(n){\n");
        s.append("  world.querySelectorAll('path.ve, path.ve-base').forEach(p=>{\n");
        s.append("    const s=p.getAttribute('data-s'), t=p.getAttribute('data-t');\n");
        s.append("    if(s===n||t===n) p.classList.add('hi'); else p.classList.remove('hi');\n");
        s.append("  });\n");
        s.append("}\n");
        s.append("function openDrawer(){\n");
        s.append("  if(drawer) drawer.classList.remove('closed');\n");
        s.append("  if(bodyEl) bodyEl.classList.remove('drawer-closed');\n");
        s.append("}\n");
        s.append("function closeDrawer(){\n");
        s.append("  dtitle.textContent='✨ Inspector'; dchips.innerHTML=''; dshapes.innerHTML='';\n");
        s.append("  dfeat.hidden=true; dfeat.innerHTML=''; dstyle.hidden=true;\n");
        s.append("  dbody.textContent='点选节点查看特征 / 超参 / shape · 可改颜色与形状';\n");
        s.append("  if(selected){selected.classList.remove('selected','pulse'); selected=null; selectedName=null;}\n");
        s.append("  world.querySelectorAll('path.ve.hi, path.ve-base.hi').forEach(p=>p.classList.remove('hi'));\n");
        s.append("  if(drawer) drawer.classList.add('closed');\n");
        s.append("  if(bodyEl) bodyEl.classList.add('drawer-closed');\n");
        s.append("}\n");
        s.append("function showInfo(n){\n");
        s.append("  openDrawer();\n");
        s.append("  const st=styleOf(n, !!(adj[n]&&adj[n].failed));\n");
        s.append("  const meta=nodeMeta[n]||{};\n");
        s.append("  const titleName=meta.name||meta.label||labelOf(n);\n");
        s.append("  dtitle.textContent=(st.emoji||'✨')+' '+titleName;\n");
        s.append("  dchips.innerHTML='';\n");
        s.append("  const chipBits=[typeOf(n), familyOf(n)];\n");
        s.append("  if(meta.feature_type) chipBits.push(String(meta.feature_type).toUpperCase());\n");
        s.append("  if(meta.task) chipBits.push('task:'+meta.task);\n");
        s.append("  if(attrNames[n]) chipBits.push('#'+attrNames[n]);\n");
        s.append("  chipBits.filter(Boolean).forEach(c=>{ const s=document.createElement('span'); s.className='vx-chip'; s.textContent=c; dchips.appendChild(s);});\n");
        s.append("  dshapes.innerHTML='';\n");
        s.append("  const id=inDims(n), od=outDims(n);\n");
        s.append("  if(meta.shape){const p=document.createElement('div'); p.className='vx-shape-pill'; p.textContent='shape '+meta.shape+(meta.dtype?(' · '+meta.dtype):''); dshapes.appendChild(p);}\n");
        s.append("  if(id.length){const p=document.createElement('div'); p.className='vx-shape-pill'; p.textContent='in  '+[...new Set(id)].join(' | '); dshapes.appendChild(p);}\n");
        s.append("  if(od.length){const p=document.createElement('div'); p.className='vx-shape-pill'; p.textContent='out '+[...new Set(od)].join(' | '); dshapes.appendChild(p);}\n");
        // Feature / label card
        s.append("  if(meta && (meta.feature_type||meta.kind==='input'||meta.kind==='output'||meta.label)){\n");
        s.append("    dfeat.hidden=false;\n");
        s.append("    const ft=String(meta.feature_type||meta.kind||'feature').toLowerCase();\n");
        s.append("    let rows='<div class=\"vx-feat-head\"><span class=\"vx-feat-badge '+ft+'\">'+(ft.toUpperCase())+'</span><span>'+esc(titleName)+'</span></div>';\n");
        s.append("    function row(k,v){ if(v==null||v==='') return ''; return '<div class=\"vx-feat-row\"><span class=\"vx-feat-k\">'+k+'</span><span class=\"vx-feat-v\">'+esc(String(v))+'</span></div>'; }\n");
        s.append("    rows+=row('名称', meta.name||titleName);\n");
        s.append("    rows+=row('类型', meta.feature_type||meta.kind);\n");
        s.append("    rows+=row('形状 shape', meta.shape);\n");
        s.append("    rows+=row('dtype', meta.dtype);\n");
        s.append("    rows+=row('vocab_size', meta.vocab_size);\n");
        s.append("    rows+=row('embed_dim', meta.embed_dim);\n");
        s.append("    rows+=row('pooling', meta.pooling);\n");
        s.append("    rows+=row('max_len', meta.max_len);\n");
        s.append("    rows+=row('padding_idx', meta.padding_idx);\n");
        s.append("    rows+=row('shared_with', meta.shared_with);\n");
        s.append("    rows+=row('label / target', meta.label);\n");
        s.append("    rows+=row('task', meta.task);\n");
        s.append("    dfeat.innerHTML=rows;\n");
        s.append("  } else { dfeat.hidden=true; dfeat.innerHTML=''; }\n");
        // Style editor
        s.append("  dstyle.hidden=false; buildStyleEditor(n, st);\n");
        s.append("  const lines=['id: '+n, 'type: '+typeOf(n)+' · family: '+familyOf(n)];\n");
        s.append("  if(meta && Object.keys(meta).length){ lines.push('','—— feature / label ——');\n");
        s.append("    Object.keys(meta).forEach(k=>lines.push('  '+k+': '+JSON.stringify(meta[k]))); }\n");
        s.append("  if(pathMap[n]) lines.push('path: '+pathMap[n]);\n");
        s.append("  if(ancestorMap[n]) lines.push('parent: '+ancestorMap[n]+' ('+labelOf(ancestorMap[n])+')');\n");
        s.append("  if(moduleInfo[n]){ lines.push('','—— module ——');\n");
        s.append("    const mi=moduleInfo[n]; if(mi.type) lines.push('class: '+mi.type);\n");
        s.append("    if(mi.attributes&&Object.keys(mi.attributes).length){ lines.push('attributes:');\n");
        s.append("      Object.keys(mi.attributes).forEach(k=>lines.push('  '+k+': '+JSON.stringify(mi.attributes[k])));}\n");
        s.append("    if(mi.parameters&&Object.keys(mi.parameters).length){ lines.push('parameters:');\n");
        s.append("      Object.keys(mi.parameters).forEach(k=>{const p=mi.parameters[k];\n");
        s.append("        lines.push('  '+k+': shape='+JSON.stringify(p.shape||[])+' requires_grad='+!!p.requires_grad);});}\n");
        s.append("  }\n");
        s.append("  if(funcInfo[n]){ lines.push('','—— call args ——', JSON.stringify(funcInfo[n],null,2)); }\n");
        s.append("  dbody.textContent=lines.join('\\n');\n");
        s.append("}\n");
        s.append("function esc(s){ return String(s).replace(/[&<>\"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c])); }\n");
        s.append("function reselect(n){\n");
        s.append("  render();\n");
        s.append("  const ng=world.querySelector('g[data-node=\"'+CSS.escape(n)+'\"]');\n");
        s.append("  if(ng){ selected=ng; selectedName=n; ng.classList.add('selected'); showInfo(n); }\n");
        s.append("}\n");
        s.append("function buildStyleEditor(n, st){\n");
        s.append("  if(!dswatch||!dshape) return;\n");
        s.append("  dswatch.innerHTML=''; dshape.innerHTML='';\n");
        s.append("  const cur=styleOverride[n]||{};\n");
        s.append("  const baseFill=cur.fill||(FAMILY[familyOf(n)]||FAMILY.module).fill;\n");
        s.append("  // Color swatches — apply fill to ANY node including middle modules & failed ones\n");
        s.append("  PALETTE.forEach((p)=>{\n");
        s.append("    const b=document.createElement('button'); b.type='button'; b.className='vx-swatch';\n");
        s.append("    b.title=p.name+' 填充'; b.style.background=p.fill; b.style.boxShadow='inset 0 0 0 1px '+p.stroke;\n");
        s.append("    if(cur.fill===p.fill||(!cur.fill&&baseFill===p.fill)) b.classList.add('active');\n");
        s.append("    b.onclick=(ev)=>{ ev.stopPropagation();\n");
        s.append("      const prev=styleOverride[n]||{};\n");
        s.append("      styleOverride[n]=Object.assign({}, prev, {fill:p.fill, stroke:p.stroke, badge:p.badge,\n");
        s.append("        depth: prev.depth!=null?prev.depth:100});\n");
        s.append("      reselect(n); toast('填充色 → '+p.name);\n");
        s.append("    };\n");
        s.append("    dswatch.appendChild(b);\n");
        s.append("  });\n");
        s.append("  // Depth slider 40–160%\n");
        s.append("  if(ddepth){\n");
        s.append("    const dval=cur.depth!=null?cur.depth:100;\n");
        s.append("    ddepth.value=String(dval);\n");
        s.append("    if(ddepthLab) ddepthLab.textContent=dval+'%';\n");
        s.append("    ddepth.oninput=(ev)=>{\n");
        s.append("      ev.stopPropagation();\n");
        s.append("      const v=+ddepth.value;\n");
        s.append("      if(ddepthLab) ddepthLab.textContent=v+'%';\n");
        s.append("      const prev=styleOverride[n]||{};\n");
        // Ensure a base fill exists so depth has something to shade
        s.append("      const fam=(FAMILY[familyOf(n)]||FAMILY.module);\n");
        s.append("      styleOverride[n]=Object.assign({}, prev, {\n");
        s.append("        fill: prev.fill||fam.fill,\n");
        s.append("        stroke: prev.stroke||fam.stroke,\n");
        s.append("        badge: prev.badge||fam.badge,\n");
        s.append("        depth: v\n");
        s.append("      });\n");
        s.append("      // Live preview without full inspector rebuild flicker: just re-render cards\n");
        s.append("      reselect(n);\n");
        s.append("    };\n");
        s.append("  }\n");
        s.append("  SHAPES.forEach(sh=>{\n");
        s.append("    const b=document.createElement('button'); b.type='button'; b.className='vx-shape-btn';\n");
        s.append("    b.textContent=SHAPE_LABEL[sh]||sh;\n");
        s.append("    const curSh=cur.shape||st.shape||'round';\n");
        s.append("    if(curSh===sh) b.classList.add('active');\n");
        s.append("    b.onclick=(ev)=>{ ev.stopPropagation();\n");
        s.append("      styleOverride[n]=Object.assign({}, styleOverride[n]||{}, {shape:sh});\n");
        s.append("      reselect(n); toast('形状 → '+(SHAPE_LABEL[sh]||sh));\n");
        s.append("    };\n");
        s.append("    dshape.appendChild(b);\n");
        s.append("  });\n");
        s.append("}\n");
        s.append("document.getElementById('dstyle-rst-'+ID).onclick=(ev)=>{\n");
        s.append("  ev.stopPropagation();\n");
        s.append("  if(!selectedName) return;\n");
        s.append("  delete styleOverride[selectedName];\n");
        s.append("  if(ddepth) ddepth.value='100'; if(ddepthLab) ddepthLab.textContent='100%';\n");
        s.append("  reselect(selectedName);\n");
        s.append("  toast('已恢复默认样式');\n");
        s.append("};\n");
        s.append("document.getElementById('dclose-'+ID).onclick=(ev)=>{\n");
        s.append("  if(ev){ev.preventDefault(); ev.stopPropagation();}\n");
        s.append("  closeDrawer();\n");
        s.append("};\n");

        s.append("function enableDrag(g,n){\n");
        s.append("  let dragging=false, ox=0, oy=0, moved=false;\n");
        s.append("  g.addEventListener('pointerdown',ev=>{\n");
        s.append("    if(ev.button!==0)return; dragging=true; moved=false;\n");
        s.append("    g.setPointerCapture(ev.pointerId); ox=ev.clientX; oy=ev.clientY; ev.stopPropagation();\n");
        s.append("  });\n");
        s.append("  g.addEventListener('pointermove',ev=>{\n");
        s.append("    if(!dragging)return;\n");
        s.append("    const dx=(ev.clientX-ox)/transform.k, dy=(ev.clientY-oy)/transform.k;\n");
        s.append("    if(Math.abs(dx)+Math.abs(dy)<0.5 && !moved) return;\n");
        s.append("    moved=true; ox=ev.clientX; oy=ev.clientY;\n");
        // Pin on first real move — stays fixed across re-layout / orient / collapse
        s.append("    pos[n].x+=dx; pos[n].y+=dy; pos[n]._auto=false;\n");
        s.append("    g.setAttribute('transform','translate('+pos[n].x+','+pos[n].y+')');\n");
        // live-update edges without full re-render for snappier drag
        s.append("    liveUpdateEdges();\n");
        s.append("  });\n");
        s.append("  g.addEventListener('pointerup',()=>{\n");
        s.append("    if(!dragging)return; dragging=false;\n");
        s.append("    if(moved){ render(); const ng=world.querySelector('g[data-node=\"'+CSS.escape(n)+'\"]');\n");
        s.append("      if(ng){ selected=ng; selectedName=n; ng.classList.add('selected'); } }\n");
        s.append("  });\n");
        s.append("}\n");

        // Drag whole container frame + its member nodes together; pin frame
        s.append("function enableFrameDrag(fg, c, members){\n");
        s.append("  let dragging=false, ox=0, oy=0, moved=false;\n");
        s.append("  fg.addEventListener('pointerdown',ev=>{\n");
        s.append("    if(ev.button!==0)return; dragging=true; moved=false;\n");
        s.append("    fg.setPointerCapture(ev.pointerId); ox=ev.clientX; oy=ev.clientY; ev.stopPropagation();\n");
        s.append("  });\n");
        s.append("  fg.addEventListener('pointermove',ev=>{\n");
        s.append("    if(!dragging)return;\n");
        s.append("    const dx=(ev.clientX-ox)/transform.k, dy=(ev.clientY-oy)/transform.k;\n");
        s.append("    if(Math.abs(dx)+Math.abs(dy)<0.5 && !moved) return;\n");
        s.append("    moved=true; ox=ev.clientX; oy=ev.clientY;\n");
        s.append("    if(!framePos[c]) return;\n");
        s.append("    framePos[c].x+=dx; framePos[c].y+=dy; framePos[c]._auto=false;\n");
        // Move all member nodes with the frame and pin them too
        s.append("    members.forEach(m=>{ if(pos[m]){ pos[m].x+=dx; pos[m].y+=dy; pos[m]._auto=false; } });\n");
        s.append("    render();\n");
        s.append("  });\n");
        s.append("  fg.addEventListener('pointerup',()=>{ dragging=false; if(moved) render(); });\n");
        // double-click pin toggle
        s.append("  fg.addEventListener('dblclick',ev=>{\n");
        s.append("    ev.stopPropagation();\n");
        s.append("    if(!framePos[c]) return;\n");
        s.append("    framePos[c]._auto = framePos[c]._auto===false ? true : false;\n");
        s.append("    if(framePos[c]._auto){\n");
        // unpin members that were only pinned via frame drag — leave individually pinned alone
        s.append("      /* keep member pins; only free the frame box */\n");
        s.append("    }\n");
        s.append("    render();\n");
        s.append("  });\n");
        s.append("}\n");

        // Live-update edge paths while dragging a node — keeps flowing dashes attached
        s.append("function liveUpdateEdges(){\n");
        s.append("  const edgeLayer=world.children[1]; if(!edgeLayer) return;\n");
        s.append("  const paths=edgeLayer.querySelectorAll('path.ve, path.ve-base');\n");
        s.append("  // group by s->t so base+flow share the same geometry\n");
        s.append("  const seen=new Set();\n");
        s.append("  paths.forEach(p=>{\n");
        s.append("    const sN=p.getAttribute('data-s'), tN=p.getAttribute('data-t');\n");
        s.append("    if(!sN||!tN) return;\n");
        s.append("    const a=pos[sN], b=pos[tN]; if(!a||!b) return;\n");
        s.append("    const ep=edgePath(a,b);\n");
        s.append("    p.setAttribute('d', ep.d);\n");
        s.append("    // move dim label once per edge pair\n");
        s.append("    const key=sN+'->'+tN; if(seen.has(key)) return; seen.add(key);\n");
        s.append("    // labels are siblings after the flow path — find nearby text with matching mid\n");
        s.append("  });\n");
        s.append("  // reposition dim labels: walk children, pair consecutive label-bg + label after paths\n");
        s.append("  const kids=Array.from(edgeLayer.children);\n");
        s.append("  for(let i=0;i<kids.length;i++){\n");
        s.append("    const el=kids[i];\n");
        s.append("    if(el.classList&&el.classList.contains('ve')&&el.getAttribute('data-s')){\n");
        s.append("      const a=pos[el.getAttribute('data-s')], b=pos[el.getAttribute('data-t')];\n");
        s.append("      if(!a||!b) continue;\n");
        s.append("      const ep=edgePath(a,b);\n");
        s.append("      // next two siblings may be label-bg + label\n");
        s.append("      const bg=kids[i+1], tx=kids[i+2];\n");
        s.append("      if(bg&&bg.classList&&bg.classList.contains('ve-label-bg')&&tx&&tx.classList&&tx.classList.contains('ve-label')){\n");
        s.append("        const tw=parseFloat(bg.getAttribute('width')||'40');\n");
        s.append("        bg.setAttribute('x', ep.mx-tw/2); bg.setAttribute('y', ep.my-10);\n");
        s.append("        tx.setAttribute('x', ep.mx); tx.setAttribute('y', ep.my+1);\n");
        s.append("      }\n");
        s.append("    }\n");
        s.append("  }\n");
        s.append("}\n");

        s.append("let panning=false, px=0, py=0;\n");
        s.append("svg.addEventListener('pointerdown',ev=>{if(ev.button!==0)return; panning=true; px=ev.clientX; py=ev.clientY; svg.setPointerCapture(ev.pointerId);});\n");
        s.append("svg.addEventListener('pointermove',ev=>{if(!panning)return; transform.x+=ev.clientX-px; transform.y+=ev.clientY-py; px=ev.clientX; py=ev.clientY; applyT();});\n");
        s.append("svg.addEventListener('pointerup',()=>{panning=false;});\n");
        s.append("svg.addEventListener('wheel',ev=>{\n");
        s.append("  ev.preventDefault();\n");
        s.append("  const rect=svg.getBoundingClientRect(); const mx=ev.clientX-rect.left, my=ev.clientY-rect.top;\n");
        s.append("  const scale=ev.deltaY<0?1.12:1/1.12;\n");
        s.append("  const nk=Math.min(3.5, Math.max(0.12, transform.k*scale));\n");
        s.append("  transform.x=mx-(mx-transform.x)*(nk/transform.k);\n");
        s.append("  transform.y=my-(my-transform.y)*(nk/transform.k);\n");
        s.append("  transform.k=nk; applyT();\n");
        s.append("},{passive:false});\n");

        s.append("function fit(){\n");
        s.append("  const vis=visibleNodes(); let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9, any=false;\n");
        s.append("  vis.forEach(n=>{const p=pos[n]; if(!p)return; any=true; minX=Math.min(minX,p.x); minY=Math.min(minY,p.y); maxX=Math.max(maxX,p.x+NW); maxY=Math.max(maxY,p.y+NH);});\n");
        s.append("  if(!any){transform={x:28,y:36,k:1}; applyT(); return;}\n");
        s.append("  const pad=52; const w=Math.max(100, stage.clientWidth-pad*2), h=Math.max(100, stage.clientHeight-pad*2);\n");
        s.append("  const bw=Math.max(1,maxX-minX), bh=Math.max(1,maxY-minY);\n");
        s.append("  const k=Math.min(w/bw, h/bh, 1.5);\n");
        s.append("  transform.k=k; transform.x=pad-minX*k+(w-bw*k)/2; transform.y=pad-minY*k+(h-bh*k)/2; applyT();\n");
        s.append("}\n");
        s.append("function zoomBy(factor){\n");
        s.append("  const rect=svg.getBoundingClientRect(); const mx=rect.width/2, my=rect.height/2;\n");
        s.append("  const nk=Math.min(3.5, Math.max(0.12, transform.k*factor));\n");
        s.append("  transform.x=mx-(mx-transform.x)*(nk/transform.k);\n");
        s.append("  transform.y=my-(my-transform.y)*(nk/transform.k);\n");
        s.append("  transform.k=nk; applyT();\n");
        s.append("}\n");
        s.append("function setOrient(o){\n");
        s.append("  orient=o;\n");
        s.append("  document.querySelectorAll('.vx-dir').forEach(b=>b.classList.toggle('active', b.getAttribute('data-dir')===o));\n");
        s.append("  Object.keys(pos).forEach(k=>{if(pos[k])pos[k]._auto=true;});\n");
        s.append("  render(); fit();\n");
        s.append("}\n");
        s.append("function refreshFlowGradient(){\n");
        s.append("  const cs=getComputedStyle(document.body);\n");
        s.append("  const g=document.getElementById('flow-'+ID); if(!g) return;\n");
        s.append("  const stops=g.querySelectorAll('stop');\n");
        s.append("  const c1=cs.getPropertyValue('--flow1').trim()||'#c4b5fd';\n");
        s.append("  const c2=cs.getPropertyValue('--flow2').trim()||'#67e8f9';\n");
        s.append("  const c3=cs.getPropertyValue('--flow3').trim()||'#fbcfe8';\n");
        s.append("  if(stops[0]) stops[0].setAttribute('stop-color', c1);\n");
        s.append("  if(stops[1]) stops[1].setAttribute('stop-color', c2);\n");
        s.append("  if(stops[2]) stops[2].setAttribute('stop-color', c3);\n");
        s.append("  // stretch gradient along current layout span for userSpaceOnUse\n");
        s.append("  let maxX=400, maxY=0;\n");
        s.append("  Object.keys(pos).forEach(k=>{const p=pos[k]; if(!p)return; maxX=Math.max(maxX,p.x+NW); maxY=Math.max(maxY,p.y+NH);});\n");
        s.append("  if(orient==='TB'||orient==='BT'){ g.setAttribute('x1','0'); g.setAttribute('y1','0'); g.setAttribute('x2','0'); g.setAttribute('y2', String(Math.max(200,maxY))); }\n");
        s.append("  else { g.setAttribute('x1','0'); g.setAttribute('y1','0'); g.setAttribute('x2', String(Math.max(200,maxX))); g.setAttribute('y2','0'); }\n");
        s.append("}\n");
        s.append("function setTheme(t){\n");
        s.append("  document.body.setAttribute('data-theme', t);\n");
        s.append("  document.querySelectorAll('.vx-theme').forEach(b=>b.classList.toggle('active', b.getAttribute('data-theme')===t));\n");
        s.append("  if(logoEl) logoEl.textContent=THEMES[t]||'🍡';\n");
        s.append("  refreshFlowGradient();\n");
        s.append("}\n");
        s.append("function toast(msg){ toastEl.hidden=false; toastEl.textContent=msg; clearTimeout(toastEl._t); toastEl._t=setTimeout(()=>{toastEl.hidden=true;}, 2200); }\n");

        // ── export ─────────────────────────────────────────────────────────
        // Export must INLINE presentation attributes: CSS variables like
        // fill:var(--ink) do not resolve in standalone SVG / canvas Image,
        // which is why users only saw arrows (paths) and blank cards/text.
        s.append("function exportSvgString(){\n");
        s.append("  const vis=visibleNodes();\n");
        s.append("  let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;\n");
        s.append("  vis.forEach(n=>{const p=pos[n]; if(!p)return; minX=Math.min(minX,p.x); minY=Math.min(minY,p.y); maxX=Math.max(maxX,p.x+NW); maxY=Math.max(maxY,p.y+NH);});\n");
        s.append("  if(minX>maxX){minX=0;minY=0;maxX=800;maxY=600;}\n");
        s.append("  const pad=48;\n");
        s.append("  const vbX=minX-pad, vbY=minY-pad, vbW=maxX-minX+pad*2, vbH=maxY-minY+pad*2;\n");
        s.append("  // Build a fresh SVG from the LIVE world, inlining computed styles\n");
        s.append("  const ns='http://www.w3.org/2000/svg';\n");
        s.append("  const out=document.createElementNS(ns,'svg');\n");
        s.append("  out.setAttribute('xmlns', ns);\n");
        s.append("  out.setAttribute('viewBox', vbX+' '+vbY+' '+vbW+' '+vbH);\n");
        s.append("  out.setAttribute('width', String(Math.max(400, Math.round(vbW))));\n");
        s.append("  out.setAttribute('height', String(Math.max(300, Math.round(vbH))));\n");
        s.append("  const cs=getComputedStyle(document.body);\n");
        s.append("  const bg=cs.getPropertyValue('--bg2').trim()||'#ffffff';\n");
        s.append("  const bgRect=document.createElementNS(ns,'rect');\n");
        s.append("  bgRect.setAttribute('x', String(vbX)); bgRect.setAttribute('y', String(vbY));\n");
        s.append("  bgRect.setAttribute('width', String(vbW)); bgRect.setAttribute('height', String(vbH));\n");
        s.append("  bgRect.setAttribute('fill', bg); out.appendChild(bgRect);\n");
        // copy defs (markers) with concrete fill
        s.append("  const liveDefs=svg.querySelector('defs');\n");
        s.append("  if(liveDefs){\n");
        s.append("    const defs=liveDefs.cloneNode(true);\n");
        s.append("    // bake arrow fill\n");
        s.append("    defs.querySelectorAll('path').forEach(p=>{\n");
        s.append("      const acc=cs.getPropertyValue('--accent').trim()||'#6366f1';\n");
        s.append("      p.setAttribute('fill', acc); p.removeAttribute('class');\n");
        s.append("    });\n");
        s.append("    out.appendChild(defs);\n");
        s.append("  }\n");
        s.append("  const gRoot=document.createElementNS(ns,'g');\n");
        s.append("  function bake(el){\n");
        s.append("    // deep-clone one live element with computed presentation attrs\n");
        s.append("    const tag=el.tagName;\n");
        s.append("    if(tag==='g'){\n");
        s.append("      const g=document.createElementNS(ns,'g');\n");
        s.append("      const tr=el.getAttribute('transform'); if(tr) g.setAttribute('transform', tr);\n");
        s.append("      Array.from(el.children).forEach(ch=>g.appendChild(bake(ch)));\n");
        s.append("      return g;\n");
        s.append("    }\n");
        s.append("    const neo=document.createElementNS(ns, tag);\n");
        s.append("    // copy geometry attributes\n");
        s.append("    ['d','x','y','x1','y1','x2','y2','cx','cy','r','rx','ry','width','height','points','text-anchor','marker-end','stroke-dasharray','stroke-width','font-size','font-weight','font-family'].forEach(a=>{\n");
        s.append("      const v=el.getAttribute(a); if(v!=null) neo.setAttribute(a,v);\n");
        s.append("    });\n");
        s.append("    // inline computed paint — critical for export fidelity\n");
        s.append("    try{\n");
        s.append("      const st=getComputedStyle(el);\n");
        s.append("      const fill=st.fill; if(fill && fill!=='none' && fill!=='rgba(0, 0, 0, 0)') neo.setAttribute('fill', fill);\n");
        s.append("      else if(el.getAttribute('fill')) neo.setAttribute('fill', el.getAttribute('fill'));\n");
        s.append("      const stroke=st.stroke; if(stroke && stroke!=='none') neo.setAttribute('stroke', stroke);\n");
        s.append("      else if(el.getAttribute('stroke')) neo.setAttribute('stroke', el.getAttribute('stroke'));\n");
        s.append("      const sw=st.strokeWidth; if(sw) neo.setAttribute('stroke-width', sw);\n");
        s.append("      const op=st.opacity; if(op && op!=='1') neo.setAttribute('opacity', op);\n");
        s.append("      if(tag==='text'){\n");
        s.append("        neo.setAttribute('font-size', st.fontSize||'12px');\n");
        s.append("        neo.setAttribute('font-weight', st.fontWeight||'700');\n");
        s.append("        neo.setAttribute('font-family', st.fontFamily||'sans-serif');\n");
        s.append("        if(fill && fill!=='none') neo.setAttribute('fill', fill);\n");
        s.append("        else neo.setAttribute('fill', cs.getPropertyValue('--ink').trim()||'#1f2937');\n");
        s.append("      }\n");
        s.append("      if(tag==='path'){\n");
        s.append("        // freeze animation for static export; dual-layer: base solid + flow dashed\n");
        s.append("        neo.setAttribute('fill','none');\n");
        s.append("        const acc=cs.getPropertyValue('--accent').trim()||'#6366f1';\n");
        s.append("        const isBase=el.classList.contains('ve-base');\n");
        s.append("        const isFlow=el.classList.contains('ve');\n");
        s.append("        if(!neo.getAttribute('stroke') || (stroke&&stroke.indexOf('url(')===0))\n");
        s.append("          neo.setAttribute('stroke', acc);\n");
        s.append("        if(isBase){\n");
        s.append("          neo.setAttribute('stroke-width', '1.4');\n");
        s.append("          neo.setAttribute('opacity', el.classList.contains('implied')?'0.12':'0.35');\n");
        s.append("          if(el.classList.contains('implied')) neo.setAttribute('stroke-dasharray','2 5');\n");
        s.append("          else neo.removeAttribute('stroke-dasharray');\n");
        s.append("        } else if(isFlow){\n");
        s.append("          neo.setAttribute('stroke-width', sw||'2.4');\n");
        s.append("          neo.setAttribute('stroke-linecap','round');\n");
        s.append("          const me=el.getAttribute('marker-end'); if(me) neo.setAttribute('marker-end', me);\n");
        s.append("          neo.setAttribute('stroke-dasharray', el.classList.contains('implied')?'2 5':'8 7');\n");
        s.append("          neo.setAttribute('opacity', el.classList.contains('implied')?'0.5':'0.95');\n");
        s.append("        } else {\n");
        s.append("          neo.setAttribute('stroke-width', sw||'2.2');\n");
        s.append("          neo.setAttribute('stroke-linecap','round');\n");
        s.append("          const me=el.getAttribute('marker-end'); if(me) neo.setAttribute('marker-end', me);\n");
        s.append("        }\n");
        s.append("      }\n");
        s.append("    }catch(err){}\n");
        s.append("    if(tag==='text') neo.textContent=el.textContent||'';\n");
        s.append("    Array.from(el.children).forEach(ch=>neo.appendChild(bake(ch)));\n");
        s.append("    return neo;\n");
        s.append("  }\n");
        s.append("  // bake each top-level layer under live world (no pan/zoom transform)\n");
        s.append("  Array.from(world.children).forEach(ch=>gRoot.appendChild(bake(ch)));\n");
        s.append("  out.appendChild(gRoot);\n");
        s.append("  return '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\\n'+new XMLSerializer().serializeToString(out);\n");
        s.append("}\n");
        s.append("function downloadBlob(blob, filename){\n");
        s.append("  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=filename;\n");
        s.append("  document.body.appendChild(a); a.click(); setTimeout(()=>{URL.revokeObjectURL(a.href); a.remove();}, 1500);\n");
        s.append("}\n");
        s.append("function exportSvg(){\n");
        s.append("  const str=exportSvgString();\n");
        s.append("  downloadBlob(new Blob([str],{type:'image/svg+xml;charset=utf-8'}), 'vista_graph.svg');\n");
        s.append("  toast('已导出 SVG');\n");
        s.append("}\n");
        s.append("function exportRaster(mime, filename, quality){\n");
        s.append("  const str=exportSvgString();\n");
        s.append("  const img=new Image();\n");
        s.append("  const url=URL.createObjectURL(new Blob([str],{type:'image/svg+xml;charset=utf-8'}));\n");
        s.append("  img.onload=function(){\n");
        s.append("    const scale=2; const canvas=document.createElement('canvas');\n");
        s.append("    canvas.width=Math.max(1, img.width*scale); canvas.height=Math.max(1, img.height*scale);\n");
        s.append("    const ctx=canvas.getContext('2d');\n");
        s.append("    const cs=getComputedStyle(document.body);\n");
        s.append("    ctx.fillStyle=cs.getPropertyValue('--bg2').trim()||'#ffffff';\n");
        s.append("    ctx.fillRect(0,0,canvas.width,canvas.height);\n");
        s.append("    ctx.drawImage(img,0,0,canvas.width,canvas.height);\n");
        s.append("    canvas.toBlob(function(blob){ if(blob){ downloadBlob(blob, filename); toast('已导出 '+filename); } URL.revokeObjectURL(url); }, mime, quality||0.92);\n");
        s.append("  };\n");
        s.append("  img.onerror=function(){ toast('导出失败'); URL.revokeObjectURL(url); };\n");
        s.append("  img.src=url;\n");
        s.append("}\n");
        s.append("function exportPdf(){\n");
        s.append("  // 1) Expand all modules so the full structure is visible\n");
        s.append("  collapsed.clear();\n");
        s.append("  Object.keys(pos).forEach(k=>{ if(pos[k] && pos[k]._auto!==false){ /* keep pins */ } else if(pos[k]) pos[k]._auto=true; });\n");
        s.append("  // Only auto-relayout unpinned nodes\n");
        s.append("  Object.keys(pos).forEach(k=>{ if(pos[k] && pos[k]._auto!==false) pos[k]._auto=true; });\n");
        s.append("  render();\n");
        s.append("  // 2) Fit + center the full graph in the stage (and for print layout)\n");
        s.append("  fit();\n");
        s.append("  // 3) Also size the SVG viewBox to the content bbox so print/PDF captures everything\n");
        s.append("  try {\n");
        s.append("    const vis=visibleNodes();\n");
        s.append("    let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9, any=false;\n");
        s.append("    vis.forEach(n=>{const p=pos[n]; if(!p)return; any=true;\n");
        s.append("      minX=Math.min(minX,p.x); minY=Math.min(minY,p.y);\n");
        s.append("      maxX=Math.max(maxX,p.x+NW); maxY=Math.max(maxY,p.y+NH);});\n");
        s.append("    if(any){\n");
        s.append("      const pad=48;\n");
        s.append("      const vbX=minX-pad, vbY=minY-pad, vbW=maxX-minX+pad*2, vbH=maxY-minY+pad*2;\n");
        s.append("      // Temporarily set viewBox so print captures the full graph centered\n");
        s.append("      svg.setAttribute('data-prev-viewbox', svg.getAttribute('viewBox')||'');\n");
        s.append("      svg.setAttribute('viewBox', vbX+' '+vbY+' '+vbW+' '+vbH);\n");
        s.append("      svg.setAttribute('width', String(Math.max(600, Math.round(vbW))));\n");
        s.append("      svg.setAttribute('height', String(Math.max(400, Math.round(vbH))));\n");
        s.append("      // Reset pan/zoom so world is identity under the viewBox\n");
        s.append("      transform={x:0,y:0,k:1}; applyT();\n");
        s.append("    }\n");
        s.append("  } catch(err) { console.warn(err); }\n");
        s.append("  toast('PDF：已展开并居中全图，请在打印框选择「存储为 PDF」');\n");
        s.append("  const restore=()=>{\n");
        s.append("    const prev=svg.getAttribute('data-prev-viewbox');\n");
        s.append("    if(prev) svg.setAttribute('viewBox', prev); else svg.removeAttribute('viewBox');\n");
        s.append("    svg.removeAttribute('data-prev-viewbox');\n");
        s.append("    svg.style.width=''; svg.style.height='';\n");
        s.append("    svg.setAttribute('width','100%'); svg.removeAttribute('height');\n");
        s.append("    fit();\n");
        s.append("    window.removeEventListener('afterprint', restore);\n");
        s.append("  };\n");
        s.append("  window.addEventListener('afterprint', restore);\n");
        s.append("  setTimeout(()=>window.print(), 500);\n");
        s.append("}\n");

        // wire buttons
        s.append("['LR','RL','TB','BT'].forEach(d=>{ const b=document.getElementById('dir-'+d+'-'+ID); if(b) b.onclick=()=>setOrient(d); });\n");
        s.append("document.getElementById('zoomin-'+ID).onclick=()=>zoomBy(1.2);\n");
        s.append("document.getElementById('zoomout-'+ID).onclick=()=>zoomBy(1/1.2);\n");
        s.append("document.getElementById('fit-'+ID).onclick=()=>fit();\n");
        s.append("document.getElementById('exp-'+ID).onclick=()=>{collapsed.clear(); Object.keys(pos).forEach(k=>{if(pos[k])pos[k]._auto=true;}); render(); fit();};\n");
        s.append("document.getElementById('col-'+ID).onclick=()=>{collapsed.clear(); containers.forEach(c=>{const m=parentToNodes[c]||[]; if(m.length&&containerDepth(c)>0)collapsed.add(c);}); Object.keys(pos).forEach(k=>{if(pos[k])pos[k]._auto=true;}); render(); fit();};\n");
        s.append("document.getElementById('rst-'+ID).onclick=()=>{Object.keys(pos).forEach(k=>delete pos[k]); Object.keys(framePos).forEach(k=>delete framePos[k]); selected=null; selectedName=null; render(); fit();};\n");
        s.append("['cute','dark','office'].forEach(t=>{ const b=document.getElementById('th-'+t+'-'+ID); if(b) b.onclick=()=>setTheme(t); });\n");
        s.append("const expBtn=document.getElementById('export-'+ID);\n");
        s.append("expBtn.onclick=(ev)=>{ ev.stopPropagation(); exportMenu.hidden=!exportMenu.hidden; };\n");
        s.append("document.addEventListener('click',()=>{ exportMenu.hidden=true; });\n");
        s.append("exportMenu.querySelectorAll('button').forEach(b=>{\n");
        s.append("  b.onclick=(ev)=>{ ev.stopPropagation(); exportMenu.hidden=true;\n");
        s.append("    const f=b.getAttribute('data-fmt');\n");
        s.append("    if(f==='svg') exportSvg();\n");
        s.append("    else if(f==='png') exportRaster('image/png','vista_graph.png');\n");
        s.append("    else if(f==='jpeg') exportRaster('image/jpeg','vista_graph.jpg',0.92);\n");
        s.append("    else if(f==='pdf') exportPdf();\n");
        s.append("  };\n");
        s.append("});\n");

        s.append("render(); fit();\n");
        s.append("window.addEventListener('resize', ()=>fit());\n");
        s.append("})();\n");
    }

    private static Path resolveOutputPath(String exportPath, String uniqueId) {
        String fileName = "jnitorch_vista_" + uniqueId + ".html";
        if (exportPath == null || exportPath.isEmpty()) {
            return Paths.get(System.getProperty("user.dir", ".")).resolve(fileName);
        }
        Path base = Paths.get(exportPath).toAbsolutePath().normalize();
        String lower = base.getFileName() != null
                ? base.getFileName().toString().toLowerCase(Locale.ROOT) : "";
        if (lower.endsWith(".png") || lower.endsWith(".svg") || lower.endsWith(".pdf") || lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            base = base.resolveSibling(
                    base.getFileName().toString().replaceAll("\\.(?i)(png|svg|pdf|jpe?g)$", ".html"));
            System.err.println("[vista] export_path non-HTML; saving interactive HTML as " + base
                    + " (use in-page Export for static formats)");
        }
        if (lower.endsWith(".html") || lower.endsWith(".htm")
                || (base.getFileName() != null && base.getFileName().toString().contains("."))) {
            return base;
        }
        return base.resolve(fileName);
    }

    private static void openInBrowser(Path html) {
        try {
            if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                Desktop.getDesktop().browse(html.toUri());
                return;
            }
        } catch (Throwable ignored) {}
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        try {
            if (os.contains("mac")) {
                new ProcessBuilder("open", html.toAbsolutePath().toString()).start();
            } else if (os.contains("win")) {
                new ProcessBuilder("cmd", "/c", "start", html.toAbsolutePath().toString()).start();
            } else {
                new ProcessBuilder("xdg-open", html.toAbsolutePath().toString()).start();
            }
        } catch (Throwable e) {
            System.err.println("[vista] could not open browser: " + e.getMessage());
            System.err.println("[vista] open manually: " + html.toAbsolutePath());
        }
    }
}