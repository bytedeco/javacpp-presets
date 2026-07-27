/*
 * Copyright (C) 2026 Bytedeco
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
package org.bytedeco.pytorch;

import java.lang.instrument.Instrumentation;
import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.Set;

/**
 * JVM module helpers for Apache Arrow (and other NIO address reflection).
 *
 * <p><b>Why this exists:</b> Arrow MemoryUtil needs reflective access to
 * {@code java.nio.Buffer.address}. On Java 16+ that requires
 * {@code --add-opens=java.base/java.nio=ALL-UNNAMED}. A dependency JAR
 * <em>cannot</em> inject command-line VM options for the host process, so we
 * provide the next-best packaged options:
 *
 * <ol>
 *   <li><b>Manifest</b> ({@code Add-Opens} / {@code Enable-Native-Access}) —
 *       applied automatically only when this jar is the {@code java -jar} main
 *       artifact (JDK launcher reads the main jar manifest).</li>
 *   <li><b>Java agent</b> — open modules via {@link Instrumentation} when the
 *       process is started with {@code -javaagent:path/to/pytorch.jar}
 *       (single flag; works even when pytorch is only on the classpath).</li>
 *   <li>{@link #ensureNioBufferAccess()} — fail fast with the exact flags to
 *       copy/paste if neither of the above applied.</li>
 * </ol>
 *
 * <p>Recommended for library users (classpath / IDE / Maven):
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *        --enable-native-access=ALL-UNNAMED \
 *        -cp ... your.Main
 * </pre>
 * or equivalently:
 * <pre>
 *   java -javaagent:/path/to/pytorch.jar -cp ... your.Main
 * </pre>
 */
public final class JvmModuleSupport {

    /** Exact CLI flags (classpath / unnamed module). */
    public static final String ADD_OPENS_NIO =
            "--add-opens=java.base/java.nio=ALL-UNNAMED";
    public static final String ENABLE_NATIVE_ACCESS =
            "--enable-native-access=ALL-UNNAMED";

    /**
     * Manifest form (main jar only, {@code java -jar}). Format is
     * {@code module/package} space-separated — not the CLI {@code =ALL-UNNAMED} form.
     */
    public static final String MANIFEST_ADD_OPENS = "java.base/java.nio";

    private static final Object LOCK = new Object();
    private static volatile Boolean nioAccessible;
    private static volatile boolean agentApplied;

    private JvmModuleSupport() {}

    /** Java agent entry (static attach via {@code -javaagent:pytorch.jar}). */
    public static void premain(String args, Instrumentation inst) {
        applyInstrumentation(inst);
    }

    /** Dynamic agent entry (if attach is allowed). */
    public static void agentmain(String args, Instrumentation inst) {
        applyInstrumentation(inst);
    }

    /**
     * Open {@code java.base/java.nio} to all unnamed modules via Instrumentation.
     * Safe to call multiple times.
     */
    public static void applyInstrumentation(Instrumentation inst) {
        if (inst == null) {
            return;
        }
        try {
            Module base = Object.class.getModule(); // java.base
            Module unnamed = ClassLoader.getSystemClassLoader().getUnnamedModule();
            // extraOpens: package -> modules that should receive open access
            inst.redefineModule(
                    base,
                    Set.of(),
                    Map.of(),
                    Map.of("java.nio", Set.of(unnamed)),
                    Set.of(),
                    Map.of());
            agentApplied = true;
            nioAccessible = null; // re-probe
        } catch (Throwable t) {
            System.err.println("[bytedeco-pytorch] JvmModuleSupport agent failed to open java.base/java.nio: " + t);
        }
    }

    /** @return true if {@code Buffer.address} is reflectively accessible right now. */
    public static boolean isNioBufferAccessible() {
        Boolean cached = nioAccessible;
        if (cached != null) {
            return cached;
        }
        synchronized (LOCK) {
            if (nioAccessible != null) {
                return nioAccessible;
            }
            nioAccessible = probeNioBufferAccess();
            return nioAccessible;
        }
    }

    /**
     * Ensure Arrow / NIO address reflection can work. No-op when already open.
     *
     * @throws IllegalStateException with the exact JVM flags if not open
     */
    public static void ensureNioBufferAccess() {
        if (isNioBufferAccessible()) {
            return;
        }
        throw new IllegalStateException(missingOpensMessage());
    }

    /** Human-readable guidance for missing module opens. */
    public static String missingOpensMessage() {
        return "Apache Arrow (used by DataFrame Arrow IPC / Feather / Lance bridge) "
                + "requires reflective access to java.nio.Buffer.address.\n"
                + "Start the JVM with either:\n"
                + "  " + ADD_OPENS_NIO + " " + ENABLE_NATIVE_ACCESS + "\n"
                + "or (single flag, agent bundled in this jar):\n"
                + "  -javaagent:/path/to/pytorch.jar\n"
                + "When using java -jar on a main jar that depends on pytorch, put "
                + "Add-Opens: " + MANIFEST_ADD_OPENS + " and Enable-Native-Access: ALL-UNNAMED "
                + "in that main jar's META-INF/MANIFEST.MF (dependency manifests are ignored).\n"
                + "Maven Surefire/Failsafe example:\n"
                + "  <argLine>" + ADD_OPENS_NIO + " " + ENABLE_NATIVE_ACCESS + "</argLine>\n"
                + "agentApplied=" + agentApplied + " java.version=" + System.getProperty("java.version");
    }

    private static boolean probeNioBufferAccess() {
        try {
            Field f = Buffer.class.getDeclaredField("address");
            f.setAccessible(true);
            ByteBuffer bb = ByteBuffer.allocateDirect(8);
            f.getLong(bb);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }
}
