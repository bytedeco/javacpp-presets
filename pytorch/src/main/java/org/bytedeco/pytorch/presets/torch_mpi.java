/*
 * Copyright (C) 2025-2026 Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.presets;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * JavaCPP preset for {@code c10d::ProcessGroupMPI}.
 *
 * <p>Java peers are generated on every OS (headers only need {@code USE_C10D_MPI}
 * + {@code mpi.h}). Native {@code jnitorch_mpi} is produced when:
 * <ul>
 *   <li>platform is linux or macosx,</li>
 *   <li>{@code mpi.h} / libmpi are on the path (Homebrew OpenMPI, system MPI, or
 *       {@code MPI_HOME}),</li>
 *   <li>and either the env opt-in is set ({@code JAVACPP_ENABLE_MPI_NATIVE=1} /
 *       {@code USE_MPI=1}) <em>or</em> {@code libtorch_cpu} already exports
 *       {@code ProcessGroupMPI} / {@code createProcessGroupMPI} (libtorch built
 *       with {@code USE_MPI=1}).</li>
 * </ul>
 *
 * <p>Skip native generation with {@code JAVACPP_SKIP_MPI_NATIVE=1} (same dummy
 * executable pattern as {@link torch_nccl} / {@link torch_ucc}).
 *
 * <p>Rebuild libtorch with {@code USE_MPI=1} after {@code brew install open-mpi}
 * (see {@code cppbuild.sh} / {@code scripts/rebuild_distributed_backends.sh}).
 */
@Properties(
    inherit = torch.class,
    value = {
        // Parse + generate Java peers on all platforms.
        @Platform(
            value = {"linux", "macosx", "windows"},
            define = {"USE_C10D_MPI", "USE_DISTRIBUTED"},
            include = {
                "torch/csrc/distributed/c10d/ProcessGroupMPI.hpp",
            }
        ),
        // Native jnitorch_mpi on POSIX when MPI + MPI-enabled libtorch are present.
        // preload "mpi" so Loader extracts/loads libmpi alongside jnitorch_mpi when
        // copyLibs pulled it into the platform classifier jar.
        @Platform(
            value = {"linux", "macosx"},
            link = { "c10", "torch", "torch_cpu", "mpi" },
            preload = { "mpi" }
        )
    },
    target = "org.bytedeco.pytorch.distributed",
    global = "org.bytedeco.pytorch.global.torch_mpi"
)
public class torch_mpi implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        torch.initIncludes(getClass(), properties);

        MpiPaths paths = discoverMpi();
        if (paths.includeDir != null) {
            properties.addAll("platform.includepath", paths.includeDir);
        }
        if (paths.libDir != null) {
            properties.addAll("platform.linkpath", paths.libDir);
            // Also put MPI lib dir first so the linker finds libmpi before any stubs.
            List<String> linkPaths = properties.get("platform.linkpath");
            if (linkPaths != null && !linkPaths.isEmpty()) {
                // already added; ensure uniqueness is fine
            }
        }

        // Build-time vs runtime:
        // - Builder calls init() to decide whether to *compile* jnitorch_mpi.
        // - Loader.load() also calls init() at runtime. Wiping platform.library
        //   here makes createProcessGroupMPI UnsatisfiedLinkError even when
        //   libjnitorch_mpi.dylib is already inside the platform classifier jar.
        // So: enable whenever we should build OR a packaged jnitorch_mpi is visible.
        boolean nativeMpi = shouldBuildNative(properties, paths);
        boolean packagedMpi = packagedJnitorchMpiPresent();
        if (!nativeMpi && !packagedMpi) {
            // No MPI build inputs and no prebuilt JNI — Java peers only.
            properties.setProperty("platform.library", "");
            properties.put("platform.link", new ArrayList<String>());
            properties.put("platform.preload", new ArrayList<String>());
            ArrayList<String> skipNative = new ArrayList<String>();
            skipNative.add("__skip_native_library__");
            properties.put("platform.executable", skipNative);
        } else {
            // Keep / restore library name so Loader extracts jnitorch_mpi.
            // (Default from global target is jnitorch_mpi; re-set if a prior
            //  skip path cleared it in the same ClassProperties reuse.)
            if (properties.getProperty("platform.library") == null
                    || properties.getProperty("platform.library").isEmpty()) {
                properties.setProperty("platform.library", "jnitorch_mpi");
            }
            ArrayList<String> links = new ArrayList<String>();
            links.add("c10");
            links.add("torch");
            links.add("torch_cpu");
            links.add("mpi");
            properties.put("platform.link", links);
            ArrayList<String> preloads = new ArrayList<String>();
            preloads.add("mpi");
            properties.put("platform.preload", preloads);
            System.err.println("torch_mpi: enabling native jnitorch_mpi"
                    + " (include=" + paths.includeDir
                    + ", lib=" + paths.libDir
                    + ", libtorchMpi=" + paths.libtorchHasMpi
                    + ", packaged=" + packagedMpi + ")");
        }
    }

    /**
     * True when platform jar / cache already ships {@code libjnitorch_mpi}.
     * Used so runtime {@link LoadEnabled#init} does not clear {@code platform.library}.
     */
    static boolean packagedJnitorchMpiPresent() {
        String[] resources = {
            "org/bytedeco/pytorch/macosx-arm64/libjnitorch_mpi.dylib",
            "org/bytedeco/pytorch/macosx-x86_64/libjnitorch_mpi.dylib",
            "org/bytedeco/pytorch/linux-x86_64/libjnitorch_mpi.so",
            "org/bytedeco/pytorch/linux-arm64/libjnitorch_mpi.so",
            "org/bytedeco/pytorch/windows-x86_64/jnitorch_mpi.dll",
        };
        ClassLoader cl = torch_mpi.class.getClassLoader();
        if (cl == null) {
            cl = ClassLoader.getSystemClassLoader();
        }
        for (String r : resources) {
            try {
                if (cl.getResource(r) != null) {
                    return true;
                }
            } catch (Throwable ignored) {
                // continue
            }
        }
        return false;
    }

    /**
     * Build native jnitorch_mpi when MPI headers/libs exist and either the user
     * opted in or libtorch_cpu already contains ProcessGroupMPI symbols.
     */
    static boolean shouldBuildNative(ClassProperties properties, MpiPaths paths) {
        if ("1".equals(System.getenv("JAVACPP_SKIP_MPI_NATIVE"))) {
            return false;
        }
        String platform = properties.getProperty("platform");
        if (platform == null
                || !(platform.startsWith("linux") || platform.startsWith("macosx"))) {
            return false;
        }
        if (paths.includeDir == null || paths.libDir == null) {
            return false;
        }
        // Explicit opt-in always wins when MPI is present.
        if ("1".equals(System.getenv("JAVACPP_ENABLE_MPI_NATIVE"))
                || "1".equals(System.getenv("USE_MPI"))) {
            return true;
        }
        // Auto-enable when libtorch was built with USE_MPI=1 (symbols present).
        return paths.libtorchHasMpi;
    }

    /** Resolved MPI include/lib dirs and whether libtorch exports MPI PG. */
    static final class MpiPaths {
        String includeDir;
        String libDir;
        boolean libtorchHasMpi;
    }

    static MpiPaths discoverMpi() {
        MpiPaths out = new MpiPaths();
        String mpiHome = System.getenv("MPI_HOME");
        if (mpiHome == null || mpiHome.isEmpty()) {
            mpiHome = System.getenv("OPENMPI_HOME");
        }

        List<String> incCandidates = new ArrayList<String>();
        List<String> libCandidates = new ArrayList<String>();
        if (mpiHome != null && !mpiHome.isEmpty()) {
            incCandidates.add(mpiHome + "/include");
            libCandidates.add(mpiHome + "/lib");
            // Homebrew Cellar layout sometimes nests version dirs; also try lib64.
            libCandidates.add(mpiHome + "/lib64");
        }
        // mpicc --showme:compile / link (OpenMPI)
        tryShowMe(incCandidates, libCandidates);

        incCandidates.add("/opt/homebrew/opt/open-mpi/include");
        incCandidates.add("/usr/local/opt/open-mpi/include");
        incCandidates.add("/opt/homebrew/opt/mpich/include");
        incCandidates.add("/usr/local/opt/mpich/include");
        incCandidates.add("/usr/include/openmpi");
        incCandidates.add("/usr/include/openmpi-x86_64");
        incCandidates.add("/usr/include/mpich");
        incCandidates.add("/usr/lib/x86_64-linux-gnu/openmpi/include");
        incCandidates.add("/usr/lib/aarch64-linux-gnu/openmpi/include");
        incCandidates.add("/usr/include");

        libCandidates.add("/opt/homebrew/opt/open-mpi/lib");
        libCandidates.add("/usr/local/opt/open-mpi/lib");
        libCandidates.add("/opt/homebrew/opt/mpich/lib");
        libCandidates.add("/usr/local/opt/mpich/lib");
        libCandidates.add("/usr/lib/x86_64-linux-gnu/openmpi/lib");
        libCandidates.add("/usr/lib/aarch64-linux-gnu/openmpi/lib");
        libCandidates.add("/usr/lib64/openmpi/lib");
        libCandidates.add("/usr/lib/openmpi/lib");
        libCandidates.add("/usr/lib");
        libCandidates.add("/usr/lib64");
        libCandidates.add("/usr/local/lib");

        for (String p : incCandidates) {
            if (p != null && new File(p, "mpi.h").isFile()) {
                out.includeDir = p;
                break;
            }
        }
        for (String p : libCandidates) {
            if (p == null) {
                continue;
            }
            File dir = new File(p);
            if (!dir.isDirectory()) {
                continue;
            }
            if (new File(dir, "libmpi.dylib").isFile()
                    || new File(dir, "libmpi.so").isFile()
                    || new File(dir, "libmpi.so.12").isFile()
                    || new File(dir, "libmpi.so.40").isFile()
                    || new File(dir, "libmpi.a").isFile()
                    || new File(dir, "libmpich.dylib").isFile()
                    || new File(dir, "libmpich.so").isFile()) {
                out.libDir = p;
                break;
            }
        }

        out.libtorchHasMpi = detectLibtorchMpiSymbols();
        return out;
    }

    /** Parse {@code mpicc --showme:compile} / {@code --showme:link} when available. */
    static void tryShowMe(List<String> inc, List<String> lib) {
        String mpicc = findMpicc();
        if (mpicc == null) {
            return;
        }
        try {
            String compile = runCmd(mpicc, "--showme:compile");
            if (compile != null) {
                for (String tok : compile.split("\\s+")) {
                    if (tok.startsWith("-I") && tok.length() > 2) {
                        inc.add(0, tok.substring(2));
                    }
                }
            }
            String link = runCmd(mpicc, "--showme:link");
            if (link != null) {
                for (String tok : link.split("\\s+")) {
                    if (tok.startsWith("-L") && tok.length() > 2) {
                        lib.add(0, tok.substring(2));
                    }
                }
            }
        } catch (Throwable ignored) {
            // best-effort
        }
    }

    static String findMpicc() {
        String mpiHome = System.getenv("MPI_HOME");
        if (mpiHome != null) {
            File f = new File(mpiHome, "bin/mpicc");
            if (f.canExecute()) {
                return f.getAbsolutePath();
            }
        }
        for (String c : new String[]{
                "/opt/homebrew/bin/mpicc",
                "/usr/local/bin/mpicc",
                "/usr/bin/mpicc",
                "mpicc"
        }) {
            if ("mpicc".equals(c)) {
                return c;
            }
            if (new File(c).canExecute()) {
                return c;
            }
        }
        return null;
    }

    static String runCmd(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd)
                    .redirectErrorStream(true)
                    .start();
            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (sb.length() > 0) {
                        sb.append(' ');
                    }
                    sb.append(line.trim());
                }
            }
            int code = p.waitFor();
            if (code != 0) {
                return null;
            }
            String s = sb.toString().trim();
            return s.isEmpty() ? null : s;
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * True when libtorch_cpu exports ProcessGroupMPI / createProcessGroupMPI
     * (i.e. libtorch was built with USE_MPI=1 / USE_C10D_MPI).
     */
    static boolean detectLibtorchMpiSymbols() {
        // Honor force flags without scanning.
        if ("1".equals(System.getenv("JAVACPP_ENABLE_MPI_NATIVE"))
                || "1".equals(System.getenv("USE_MPI"))) {
            // Still report via caller; scanning is optional then.
        }
        List<File> candidates = new ArrayList<File>();
        String userDir = System.getProperty("user.dir", ".");
        // cppbuild layout used by this preset module
        String[] relatives = {
            "cppbuild/macosx-arm64/lib/libtorch_cpu.dylib",
            "cppbuild/macosx-arm64/pytorch/torch/lib/libtorch_cpu.dylib",
            "cppbuild/macosx-arm64/pytorch/build/lib/libtorch_cpu.dylib",
            "cppbuild/macosx-x86_64/lib/libtorch_cpu.dylib",
            "cppbuild/macosx-x86_64/pytorch/torch/lib/libtorch_cpu.dylib",
            "cppbuild/linux-x86_64/lib/libtorch_cpu.so",
            "cppbuild/linux-x86_64/pytorch/torch/lib/libtorch_cpu.so",
            "cppbuild/linux-arm64/lib/libtorch_cpu.so",
            "cppbuild/linux-arm64/pytorch/torch/lib/libtorch_cpu.so",
            // when cwd is javacpp-presets root
            "pytorch/cppbuild/macosx-arm64/lib/libtorch_cpu.dylib",
            "pytorch/cppbuild/macosx-arm64/pytorch/torch/lib/libtorch_cpu.dylib",
            "pytorch/cppbuild/linux-x86_64/lib/libtorch_cpu.so",
            "pytorch/cppbuild/linux-x86_64/pytorch/torch/lib/libtorch_cpu.so",
        };
        for (String r : relatives) {
            candidates.add(new File(userDir, r));
        }
        // Also scan platform.linkpath-style env
        String torchLib = System.getenv("LIBTORCH_LIB_DIR");
        if (torchLib != null) {
            candidates.add(new File(torchLib, "libtorch_cpu.dylib"));
            candidates.add(new File(torchLib, "libtorch_cpu.so"));
        }

        for (File lib : candidates) {
            if (lib == null || !lib.isFile()) {
                continue;
            }
            if (nmContainsMpi(lib)) {
                return true;
            }
        }
        return false;
    }

    static boolean nmContainsMpi(File lib) {
        // Prefer nm -gU on macOS (defined globals); plain nm elsewhere.
        String[][] cmds = {
            {"nm", "-gU", lib.getAbsolutePath()},
            {"nm", "-D", lib.getAbsolutePath()},
            {"nm", lib.getAbsolutePath()},
        };
        for (String[] cmd : cmds) {
            try {
                Process p = new ProcessBuilder(cmd)
                        .redirectErrorStream(true)
                        .start();
                boolean found = false;
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (line.contains("ProcessGroupMPI")
                                || line.contains("createProcessGroupMPI")) {
                            found = true;
                            // drain a bit but can break early
                            break;
                        }
                    }
                }
                p.waitFor();
                if (found) {
                    return true;
                }
            } catch (Throwable ignored) {
                // try next nm flavor
            }
        }
        return false;
    }

    @Override
    public void map(InfoMap infoMap) {
        for (torch.PointerInfo pi : new torch.PointerInfo[]{
            new torch.PointerInfo("c10d::ProcessGroupMPI"),
            new torch.PointerInfo("c10d::ProcessGroupMPI::WorkMPI"),
            new torch.PointerInfo("c10d::ProcessGroupMPI::AsyncWork"),
        }) {
            pi.makeIntrusive(infoMap);
        }

        infoMap
            .put(new Info().javaText("import org.bytedeco.pytorch.distributed.Backend;"))
            .put(new Info().javaText("import org.bytedeco.pytorch.distributed.Work;"))
            // MPI opaque C types — keep as Pointer; MPI_Comm is often an int/pointer
            // typedef depending on the MPI implementation (OpenMPI vs MPICH).
            .put(new Info("MPI_Comm", "MPI_Request", "MPI_Status", "MPI_Datatype",
                          "MPI_Op", "MPI_Group", "MPI_Fint", "MPI_Aint",
                          "MPI_Count", "MPI_Offset", "MPI_File", "MPI_Info",
                          "MPI_Win", "MPI_Message", "MPI_Errhandler")
                    .cast().pointerTypes("Pointer"))
            // Internal worker / entry types not part of public API surface
            .put(new Info("c10d::WorkEntry",
                          "c10d::ProcessGroupMPI::WorkType",
                          "c10d::ProcessGroupMPI::enqueue",
                          "c10d::ProcessGroupMPI::runLoop",
                          "c10d::ProcessGroupMPI::destroy",
                          "c10d::ProcessGroupMPI::initMPIOnce",
                          "c10d::ProcessGroupMPI::mpiExit",
                          "c10d::ProcessGroupMPI::pgGlobalMutex_",
                          "c10d::ProcessGroupMPI::mpiThreadSupport_",
                          "std::unique_ptr<c10d::WorkEntry>",
                          "std::deque<c10d::ProcessGroupMPI::WorkType>",
                          "std::function<void(std::unique_ptr<c10d::WorkEntry>&)>").skip())
            // Backend collective returns are c10::intrusive_ptr<c10d::Work>, not ProcessGroupMPI.
            // Without this, JavaCPP attributes the intrusive_ptr to ProcessGroupMPI (the
            // enclosing class) and jnitorch_mpi.cpp fails to compile (IntrusivePtrAdapter
            // type mismatch). Match ProcessGroupGloo / ProcessGroupNCCL annotations.
            .put(new Info("c10::intrusive_ptr<c10d::Work>")
                    .annotations("@IntrusivePtr(\"c10d::Work\")")
                    .valueTypes("Work").pointerTypes("Work"))
            .put(new Info("c10d::kUnsetTimeout")
                    .javaText("@MemberGetter public static native @ByVal Milliseconds kUnsetTimeout();"))
            .put(new Info("std::chrono::milliseconds(kUnsetTimeout)")
                    .javaText("std::chrono::milliseconds(c10d::kUnsetTimeout)"))
            // Override all collective methods to use @IntrusivePtr("c10d::Work") instead of
            // @IntrusivePtr("c10d::ProcessGroupMPI"). All c10d backends return intrusive_ptr<Work>.
            .put(new Info("c10d::ProcessGroupMPI::broadcast")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work broadcast(@ByRef TensorVector data, @Const @ByRef(nullValue = \"c10d::BroadcastOptions()\") BroadcastOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work broadcast(@ByRef TensorVector data);"))
            .put(new Info("c10d::ProcessGroupMPI::allreduce")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work allreduce(@ByRef TensorVector tensors, @Const @ByRef(nullValue = \"c10d::AllreduceOptions()\") AllreduceOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work allreduce(@ByRef TensorVector tensors);"))
            .put(new Info("c10d::ProcessGroupMPI::allreduce_coalesced")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work allreduce_coalesced(@ByRef TensorVector tensors, @Const @ByRef(nullValue = \"c10d::AllreduceCoalescedOptions()\") AllreduceCoalescedOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work allreduce_coalesced(@ByRef TensorVector tensors);"))
            .put(new Info("c10d::ProcessGroupMPI::reduce")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work reduce(@ByRef TensorVector tensors, @Const @ByRef(nullValue = \"c10d::ReduceOptions()\") ReduceOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work reduce(@ByRef TensorVector tensors);"))
            .put(new Info("c10d::ProcessGroupMPI::allgather")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work allgather(@StdVector TensorVector outputTensors, @ByRef TensorVector inputTensors, @Const @ByRef(nullValue = \"c10d::AllgatherOptions()\") AllgatherOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work allgather(@StdVector TensorVector outputTensors, @ByRef TensorVector inputTensors);"))
            .put(new Info("c10d::ProcessGroupMPI::_allgather_base")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work _allgather_base(@ByRef Tensor outputbuffer, @ByRef Tensor inputbuffer, @Const @ByRef(nullValue = \"c10d::AllgatherOptions()\") AllgatherOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work _allgather_base(@ByRef Tensor outputbuffer, @ByRef Tensor inputbuffer);"))
            .put(new Info("c10d::ProcessGroupMPI::allgather_coalesced")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work allgather_coalesced(@StdVector TensorVector outputTensorLists, @ByRef TensorVector inputTensors, @Const @ByRef(nullValue = \"c10d::AllgatherOptions()\") AllgatherOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work allgather_coalesced(@StdVector TensorVector outputTensorLists, @ByRef TensorVector inputTensors);"))
            .put(new Info("c10d::ProcessGroupMPI::gather")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work gather(@StdVector TensorVector outputTensors, @ByRef TensorVector inputTensors, @Const @ByRef(nullValue = \"c10d::GatherOptions()\") GatherOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work gather(@StdVector TensorVector outputTensors, @ByRef TensorVector inputTensors);"))
            .put(new Info("c10d::ProcessGroupMPI::scatter")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work scatter(@ByRef TensorVector outputTensors, @StdVector TensorVector inputTensors, @Const @ByRef(nullValue = \"c10d::ScatterOptions()\") ScatterOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work scatter(@ByRef TensorVector outputTensors, @StdVector TensorVector inputTensors);"))
            .put(new Info("c10d::ProcessGroupMPI::reduce_scatter")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work reduce_scatter(@ByRef TensorVector outputTensors, @StdVector TensorVector inputTensors, @Const @ByRef(nullValue = \"c10d::ReduceScatterOptions()\") ReduceScatterOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work reduce_scatter(@ByRef TensorVector outputTensors, @StdVector TensorVector inputTensors);"))
            .put(new Info("c10d::ProcessGroupMPI::_reduce_scatter_base")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work _reduce_scatter_base(@ByRef Tensor outputTensor, @ByRef Tensor inputTensor, @Const @ByRef(nullValue = \"c10d::ReduceScatterOptions()\") ReduceScatterOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work _reduce_scatter_base(@ByRef Tensor outputTensor, @ByRef Tensor inputTensor);"))
            .put(new Info("c10d::ProcessGroupMPI::alltoall_base")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work alltoall_base(@ByRef Tensor outputTensor, @ByRef Tensor inputTensor, @Cast(\"std::vector<int64_t>*\") @ByRef LongVector outputSplitSizes, @Cast(\"std::vector<int64_t>*\") @ByRef LongVector inputSplitSizes, @Const @ByRef(nullValue = \"c10d::AllToAllOptions()\") AllToAllOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work alltoall_base(@ByRef Tensor outputTensor, @ByRef Tensor inputTensor, @Cast(\"std::vector<int64_t>*\") @ByRef LongVector outputSplitSizes, @Cast(\"std::vector<int64_t>*\") @ByRef LongVector inputSplitSizes);"))
            .put(new Info("c10d::ProcessGroupMPI::alltoall")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work alltoall(@ByRef TensorVector outputTensors, @ByRef TensorVector inputTensors, @Const @ByRef(nullValue = \"c10d::AllToAllOptions()\") AllToAllOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work alltoall(@ByRef TensorVector outputTensors, @ByRef TensorVector inputTensors);"))
            .put(new Info("c10d::ProcessGroupMPI::send")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work send(@ByRef TensorVector tensors, int dstRank, int tag);"))
            .put(new Info("c10d::ProcessGroupMPI::recv")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work recv(@ByRef TensorVector tensors, int srcRank, int tag);"))
            .put(new Info("c10d::ProcessGroupMPI::recvAnysource")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work recvAnysource(@ByRef TensorVector tensor, int tag);"))
            .put(new Info("c10d::ProcessGroupMPI::barrier")
                    .javaText("public native @IntrusivePtr(\"c10d::Work\") Work barrier(@Const @ByRef(nullValue = \"c10d::BarrierOptions()\") BarrierOptions opts);\n" +
                              "  public native @IntrusivePtr(\"c10d::Work\") Work barrier();"))
        ;
    }
}
