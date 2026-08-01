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
package org.bytedeco.pytorch.llm.unsloth.studio.webui;

/** Embedded HTML for the visual training board (no npm). */
public final class BoardStaticAssets {
    private BoardStaticAssets() {}

    public static String indexHtml(int port) {
        StringBuilder sb = new StringBuilder(4096);
        sb.append("<!DOCTYPE html>\n");
        sb.append("<html lang=\"en\">\n<head>\n");
        sb.append("<meta charset=\"utf-8\"/>\n");
        sb.append("<title>Unsloth Studio Board (Java)</title>\n");
        sb.append("<style>\n");
        sb.append(" body{font-family:ui-sans-serif,system-ui,sans-serif;background:#0b1220;color:#e7eef7;margin:0;padding:24px}\n");
        sb.append(" h1{font-size:20px;margin:0 0 8px}\n");
        sb.append(" .sub{color:#9fb3c8;margin-bottom:20px}\n");
        sb.append(" .card{background:#121a2b;border:1px solid #243047;border-radius:12px;padding:16px;margin-bottom:16px}\n");
        sb.append(" .row{display:flex;gap:16px;flex-wrap:wrap}\n");
        sb.append(" img{max-width:100%;border-radius:8px;background:#0b1220}\n");
        sb.append(" button{background:#5b9cff;color:#041018;border:0;border-radius:8px;padding:8px 12px;font-weight:600;cursor:pointer}\n");
        sb.append(" pre{white-space:pre-wrap;font-size:12px;color:#c7d2e0}\n");
        sb.append(" .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#1b2a44;margin-right:6px;font-size:12px}\n");
        sb.append("</style>\n</head>\n<body>\n");
        sb.append("<h1>Unsloth Studio · Visual Training Board</h1>\n");
        sb.append("<div class=\"sub\">Pure-Java board on port ").append(port);
        sb.append(" — live loss / GPU / run control via SSE</div>\n");
        sb.append("<div class=\"row\">\n");
        sb.append(" <div class=\"card\" style=\"flex:2\">\n");
        sb.append("  <div>Loss curve</div>\n");
        sb.append("  <img id=\"chart\" src=\"/api/svg?key=loss\" alt=\"loss\"/>\n");
        sb.append(" </div>\n");
        sb.append(" <div class=\"card\" style=\"flex:1\">\n");
        sb.append("  <div>Status</div>\n");
        sb.append("  <div id=\"status\"><span class=\"pill\">connecting...</span></div>\n");
        sb.append("  <button onclick=\"refresh()\">Refresh</button>\n");
        sb.append("  <pre id=\"snap\">{}</pre>\n");
        sb.append(" </div>\n</div>\n");
        sb.append("<script>\n");
        sb.append("async function refresh(){\n");
        sb.append("  try{\n");
        sb.append("    const r = await fetch('/api/snapshot');\n");
        sb.append("    const j = await r.json();\n");
        sb.append("    document.getElementById('snap').textContent = JSON.stringify(j,null,2);\n");
        sb.append("    const runs = j.runs||[];\n");
        sb.append("    if(runs.length){\n");
        sb.append("      const id = runs[runs.length-1].run_id;\n");
        sb.append("      document.getElementById('chart').src = '/api/svg?key='+encodeURIComponent(id+'/loss')+'&t='+Date.now();\n");
        sb.append("      document.getElementById('status').innerHTML = '<span class=\"pill\">'+runs.length+' runs</span><span class=\"pill\">'+id+'</span>';\n");
        sb.append("    } else {\n");
        sb.append("      document.getElementById('status').innerHTML = '<span class=\"pill\">idle</span>';\n");
        sb.append("    }\n");
        sb.append("  }catch(e){ document.getElementById('status').textContent = String(e); }\n");
        sb.append("}\n");
        sb.append("try{\n");
        sb.append("  const es = new EventSource('/board/events');\n");
        sb.append("  es.addEventListener('progress', ev => { refresh(); });\n");
        sb.append("  es.onopen = () => refresh();\n");
        sb.append("}catch(e){ refresh(); }\n");
        sb.append("setInterval(refresh, 5000);\n");
        sb.append("</script>\n</body>\n</html>\n");
        return sb.toString();
    }
}
