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
package org.bytedeco.pytorch.vision.utils;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.vision.io.ImageIO;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * torchvision.utils: {@code make_grid}, {@code save_image}.
 */
public final class VisionUtils {
    private VisionUtils() {}

    /**
     * Make an image grid from NCHW tensor (float [0,1]).
     *
     * @return CHW tensor of the grid
     */
    public static Tensor make_grid(Tensor tensor, int nrow, int padding) {
        Objects.requireNonNull(tensor, "tensor");
        Tensor t = tensor.contiguous().cpu().to(torch.ScalarType.Float);
        long[] sizes = ImageTensors.sizes(t);
        if (sizes.length == 3) {
            t = t.unsqueeze(0);
            sizes = ImageTensors.sizes(t);
        }
        if (sizes.length != 4) {
            throw new IllegalArgumentException("make_grid expects NCHW or CHW");
        }
        int n = (int) sizes[0];
        int c = (int) sizes[1];
        int h = (int) sizes[2];
        int w = (int) sizes[3];
        int cols = Math.max(1, nrow);
        int rows = (n + cols - 1) / cols;
        int gh = rows * h + padding * (rows + 1);
        int gw = cols * w + padding * (cols + 1);
        float[] grid = new float[c * gh * gw]; // zeros = black padding
        float[] data = ImageTensors.toFloatArray(t);
        int sample = c * h * w;
        for (int i = 0; i < n; i++) {
            int row = i / cols;
            int col = i % cols;
            int top = padding + row * (h + padding);
            int left = padding + col * (w + padding);
            for (int ch = 0; ch < c; ch++) {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int src = i * sample + ch * h * w + y * w + x;
                        int dst = ch * gh * gw + (top + y) * gw + (left + x);
                        grid[dst] = data[src];
                    }
                }
            }
        }
        return torch.tensor(grid).reshape(c, gh, gw);
    }

    public static Tensor make_grid(Tensor tensor) {
        return make_grid(tensor, 8, 2);
    }

    public static Tensor makeGrid(Tensor tensor, int nrow, int padding) {
        return make_grid(tensor, nrow, padding);
    }

    public static void save_image(Tensor tensor, String path) throws IOException {
        Tensor grid = tensor.dim() == 4 ? make_grid(tensor) : tensor;
        ImageIO.write_image(grid, path);
    }

    public static void save_image(Tensor tensor, Path path) throws IOException {
        save_image(tensor, path.toString());
    }

    public static void saveImage(Tensor tensor, String path) throws IOException {
        save_image(tensor, path);
    }

    /** Tile BufferedImages into a single grid image (no Tensor required). */
    public static BufferedImage makeGridImages(BufferedImage[] images, int nrow, int padding) {
        Objects.requireNonNull(images, "images");
        if (images.length == 0) {
            throw new IllegalArgumentException("empty");
        }
        int h = images[0].getHeight();
        int w = images[0].getWidth();
        int cols = Math.max(1, nrow);
        int rows = (images.length + cols - 1) / cols;
        int gh = rows * h + padding * (rows + 1);
        int gw = cols * w + padding * (cols + 1);
        BufferedImage out = new BufferedImage(gw, gh, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setColor(java.awt.Color.BLACK);
        g.fillRect(0, 0, gw, gh);
        for (int i = 0; i < images.length; i++) {
            int row = i / cols;
            int col = i % cols;
            int top = padding + row * (h + padding);
            int left = padding + col * (w + padding);
            g.drawImage(images[i], left, top, w, h, null);
        }
        g.dispose();
        return out;
    }
}
