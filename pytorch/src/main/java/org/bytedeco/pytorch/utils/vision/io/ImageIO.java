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
package org.bytedeco.pytorch.utils.vision.io;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.vision.utils.ImageTensors;

import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.Objects;

/**
 * torchvision.io-style image I/O using JDK {@link javax.imageio.ImageIO}.
 * {@link #read_image} returns a uint8-like float CHW tensor in {@code [0,1]} (same as ToTensor).
 */
public final class ImageIO {
    private ImageIO() {}

    public static Tensor read_image(String path) throws IOException {
        return read_image(Path.of(path));
    }

    public static Tensor read_image(Path path) throws IOException {
        BufferedImage img = javax.imageio.ImageIO.read(path.toFile());
        if (img == null) {
            throw new IOException("cannot decode image: " + path);
        }
        return ImageTensors.toTensor(img);
    }

    public static Tensor read_image(File file) throws IOException {
        return read_image(file.toPath());
    }

    public static Tensor decode_image(byte[] data) throws IOException {
        Objects.requireNonNull(data, "data");
        try (InputStream in = new ByteArrayInputStream(data)) {
            BufferedImage img = javax.imageio.ImageIO.read(in);
            if (img == null) {
                throw new IOException("cannot decode image bytes");
            }
            return ImageTensors.toTensor(img);
        }
    }

    public static void write_image(Tensor tensor, String path) throws IOException {
        write_image(tensor, Path.of(path));
    }

    public static void write_image(Tensor tensor, Path path) throws IOException {
        Objects.requireNonNull(tensor, "tensor");
        BufferedImage img = ImageTensors.toBufferedImage(tensor);
        String name = path.getFileName().toString();
        String format = formatFromName(name);
        Files.createDirectories(path.getParent() == null ? Path.of(".") : path.getParent());
        if (!javax.imageio.ImageIO.write(img, format, path.toFile())) {
            throw new IOException("no writer for format " + format);
        }
    }

    public static byte[] encode_png(Tensor tensor) throws IOException {
        return encode(tensor, "png", -1f);
    }

    public static byte[] encode_jpeg(Tensor tensor, float quality) throws IOException {
        return encode(tensor, "jpg", quality);
    }

    public static byte[] encode_jpeg(Tensor tensor) throws IOException {
        return encode_jpeg(tensor, 0.9f);
    }

    private static byte[] encode(Tensor tensor, String format, float quality) throws IOException {
        BufferedImage img = ImageTensors.toBufferedImage(tensor);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        if ("jpg".equals(format) || "jpeg".equals(format)) {
            writeJpeg(img, bos, quality <= 0 ? 0.9f : quality);
        } else if (!javax.imageio.ImageIO.write(img, format, bos)) {
            throw new IOException("no writer for format " + format);
        }
        return bos.toByteArray();
    }

    private static void writeJpeg(BufferedImage img, OutputStream out, float quality) throws IOException {
        Iterator<ImageWriter> writers = javax.imageio.ImageIO.getImageWritersByFormatName("jpeg");
        if (!writers.hasNext()) {
            throw new IOException("no jpeg writer");
        }
        ImageWriter writer = writers.next();
        try (ImageOutputStream ios = javax.imageio.ImageIO.createImageOutputStream(out)) {
            writer.setOutput(ios);
            ImageWriteParam param = writer.getDefaultWriteParam();
            if (param.canWriteCompressed()) {
                param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionQuality(Math.max(0.01f, Math.min(1f, quality)));
            }
            writer.write(null, new javax.imageio.IIOImage(img, null, null), param);
        } finally {
            writer.dispose();
        }
    }

    private static String formatFromName(String name) {
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot == name.length() - 1) {
            return "png";
        }
        String ext = name.substring(dot + 1).toLowerCase();
        if ("jpeg".equals(ext)) {
            return "jpg";
        }
        return ext;
    }

    // camelCase aliases
    public static Tensor readImage(String path) throws IOException { return read_image(path); }
    public static void writeImage(Tensor tensor, String path) throws IOException { write_image(tensor, path); }
    public static Tensor decodeImage(byte[] data) throws IOException { return decode_image(data); }
    public static byte[] encodePng(Tensor tensor) throws IOException { return encode_png(tensor); }
    public static byte[] encodeJpeg(Tensor tensor) throws IOException { return encode_jpeg(tensor); }
}
