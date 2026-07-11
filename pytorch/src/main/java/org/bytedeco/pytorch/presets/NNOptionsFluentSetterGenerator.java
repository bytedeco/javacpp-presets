/*
 * Copyright (C) 2020-2026 Samuel Audet
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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Generates JavaCPP native bindings for LibTorch's {@code TORCH_ARG} setters. */
public final class NNOptionsFluentSetterGenerator {
    private static final Pattern CLASS_PATTERN = Pattern.compile("public class (\\w+) extends \\w+");
    // JavaCPP emits each getter on a single line as `public native <sign> name();`
    // so the trailing `();` is part of the line; the regex matches `name` followed
    // by `();`.
    private static final Pattern GETTER_PATTERN = Pattern.compile("^public native (.+) ([A-Za-z_][A-Za-z0-9_]*)\\(\\);\\s*$");
    // Match an existing fluent setter declaration. The setter's parameter list may
    // contain nested parens (e.g. `@Cast("bool") boolean`), so we scan ahead for
    // a balanced close instead of relying on `[^)]*`. Tolerates any depth up to 2,
    // which is enough for the patterns JavaCPP emits.
    private static final Pattern EXISTING_SETTER_PATTERN =
            Pattern.compile("\\b([A-Za-z_][A-Za-z0-9_]*)\\((?:[^()]|\\([^()]*\\))*\\bsetter\\b(?:[^()]|\\([^()]*\\))*\\)");
    private static final Pattern EXPANDING_ARRAY_PATTERN = Pattern.compile("@Cast\\(\"torch::ExpandingArray<([^>]+)>\\*\"\\)");

    private NNOptionsFluentSetterGenerator() { }

    public static void main(String[] args) throws IOException {
        Path base = args.length > 0
                ? Paths.get(args[0])
                : Paths.get("src/gen/java/org/bytedeco/pytorch");
        if (!Files.isDirectory(base)) {
            throw new IOException("Generated source directory not found: " + base.toAbsolutePath());
        }

        List<Path> optionFiles = new ArrayList<Path>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(base, "*Options.java")) {
            for (Path path : stream) {
                optionFiles.add(path);
            }
        }
        Collections.sort(optionFiles);

        int updatedFiles = 0;
        int addedMethods = 0;
        for (Path path : optionFiles) {
            UpdateResult result = updateFile(path);
            if (result.addedMethods > 0) {
                updatedFiles++;
                addedMethods += result.addedMethods;
            }
        }

        System.out.println("NNOptionsFluentSetterGenerator: updated " + updatedFiles
                + " files with " + addedMethods + " fluent setters.");
    }

    private static UpdateResult updateFile(Path path) throws IOException {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        // Match either @Namespace("torch::nn"...) (non-template classes such as
        // LinearOptions, GRUOptions, etc.) or @Name("torch::nn::Foo<N>") (template
        // instantiations such as AvgPool2dOptions, Conv3dOptions, etc.). Without
        // this, the templated options get skipped and miss their fluent setters.
        boolean isTorchOptions = text.contains("@Namespace(\"torch::nn")
                || text.contains("@Namespace(\"torch::data")
                || text.contains("@Namespace(\"torch::optim")
                || text.contains("@Name(\"torch::nn::");
        if (!isTorchOptions) {
            return UpdateResult.EMPTY;
        }

        Matcher classMatcher = CLASS_PATTERN.matcher(text);
        if (!classMatcher.find()) {
            return UpdateResult.EMPTY;
        }
        String className = classMatcher.group(1);

        // Strip Java comments BEFORE collapsing whitespace, so the line-comment
        // regex `//[^\n]*` still has a newline to terminate on. Javadocs often
        // quote C++ usage like `LinearOptions(5, 2).bias(false);` whose stray
        // `;` would otherwise corrupt any naive split of the file body.
        String codeNoComments = text
                .replaceAll("/\\*[\\s\\S]*?\\*/", " ")
                .replaceAll("//[^\\n]*", " ");
        String normalized = codeNoComments.replaceAll("\\s+", " ").trim();

        Set<String> existingSetters = new HashSet<String>();
        Matcher setterMatcher = EXISTING_SETTER_PATTERN.matcher(normalized);
        while (setterMatcher.find()) {
            existingSetters.add(setterMatcher.group(1));
        }

        List<String> generatedMethods = new ArrayList<String>();
        // Walk the file line by line. Each declaration we care about sits on a
        // single line in the JavaCPP-generated `*Options.java` output.
        for (String rawLine : codeNoComments.split("\\r?\\n")) {
            String line = rawLine.trim();
            if (!line.startsWith("public native ")) {
                continue;
            }
            Matcher getterMatcher = GETTER_PATTERN.matcher(line);
            if (!getterMatcher.matches()) {
                continue;
            }
            String returnSignature = getterMatcher.group(1);
            if (!returnSignature.contains("@NoException")) {
                continue;
            }
            String methodName = getterMatcher.group(2);
            if (existingSetters.contains(methodName)) {
                continue;
            }

            String setter = buildSetter(className, methodName, returnSignature);
            if (setter != null && !setter.isEmpty()) {
                generatedMethods.add(setter);
                existingSetters.add(methodName);
            }
        }

        if (generatedMethods.isEmpty()) {
            return UpdateResult.EMPTY;
        }

        int insertionPoint = text.lastIndexOf("\n}");
        if (insertionPoint < 0) {
            insertionPoint = text.lastIndexOf('}');
        }
        if (insertionPoint < 0) {
            throw new IOException("Could not locate class terminator in " + path);
        }

        StringBuilder builder = new StringBuilder(text.length() + generatedMethods.size() * 160);
        builder.append(text, 0, insertionPoint);
        builder.append('\n');
        for (String method : generatedMethods) {
            builder.append('\n').append(method);
        }
        builder.append(text.substring(insertionPoint));

        String updated = builder.toString();
        if (!updated.equals(text)) {
            Files.write(path, updated.getBytes(StandardCharsets.UTF_8));
        }
        return new UpdateResult(generatedMethods.size());
    }

    private static String buildSetter(String className, String methodName, String returnSignature) {
        String javaType = returnSignature.substring(returnSignature.lastIndexOf(' ') + 1);

        if ("BoolPointer".equals(javaType)) {
            return scalarSetter(className, methodName, "@Cast(\"bool\") boolean");
        } else if ("LongPointer".equals(javaType)) {
            if (parseExpandingArraySize(returnSignature) != null) {
                return objectSetter(className, methodName, returnSignature, javaType);
            }
            return scalarSetter(className, methodName, "@Cast(\"int64_t\") long");
        } else if ("SizeTPointer".equals(javaType)) {
            return scalarSetter(className, methodName, "@Cast(\"size_t\") long");
        } else if ("DoublePointer".equals(javaType)) {
            if (returnSignature.contains("std::tuple")) {
                return objectSetter(className, methodName, returnSignature, javaType);
            }
            return scalarSetter(className, methodName, "double");
        } else if ("FloatPointer".equals(javaType)) {
            return scalarSetter(className, methodName, "float");
        } else if ("IntPointer".equals(javaType)) {
            return scalarSetter(className, methodName, "int");
        } else if ("ShortPointer".equals(javaType)) {
            return scalarSetter(className, methodName, "short");
        } else if ("BytePointer".equals(javaType) && returnSignature.contains("@StdString")) {
            // e.g. GELUOptions.approximate(): emit a @StdString BytePointer setter and
            // a String setter, since the underlying C++ accepts an std::string.
            String stdSetter = objectSetter(className, methodName, returnSignature, javaType);
            String stringSetter = "  public native @ByRef @NoException(true) " + className + " " + methodName
                    + "(@StdString String setter);";
            return stdSetter + "\n" + stringSetter;
        } else if ("BytePointer".equals(javaType)) {
            return scalarSetter(className, methodName, "byte");
        }

        return objectSetter(className, methodName, returnSignature, javaType);
    }

    private static Integer parseExpandingArraySize(String returnSignature) {
        Matcher matcher = EXPANDING_ARRAY_PATTERN.matcher(returnSignature);
        if (!matcher.find()) {
            return null;
        }
        String[] factors = matcher.group(1).split("\\*");
        int size = 1;
        for (String factor : factors) {
            size *= Integer.parseInt(factor.trim());
        }
        return Integer.valueOf(size);
    }

    private static String scalarSetter(String className, String methodName, String valueType) {
        return "  public native @ByRef @NoException(true) " + className + " " + methodName
                + "(" + valueType + " setter);";
    }

    private static String objectSetter(String className, String methodName, String returnSignature, String javaType) {
        String annotations = returnSignature.substring(0, returnSignature.length() - javaType.length())
                .replace("@ByRef", "@ByVal")
                .replaceAll("@NoException(?:\\([^)]*\\))?", "")
                .trim();
        return "  public native @ByRef @NoException(true) " + className + " " + methodName
                + "(" + annotations + " " + javaType + " setter);";
    }

    private static final class UpdateResult {
        static final UpdateResult EMPTY = new UpdateResult(0);

        final int addedMethods;

        UpdateResult(int addedMethods) {
            this.addedMethods = addedMethods;
        }
    }
}
