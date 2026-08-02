/**
 * Pure-Java Pillow (PIL) API mapping under {@code org.bytedeco.pytorch.vision.pillow}.
 *
 * <p>Maps the upstream
 * <a href="https://github.com/python-pillow/Pillow">python-pillow/Pillow</a>
 * user-facing API onto a pure-Java pixel core ({@link org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer})
 * plus JDK {@link javax.imageio.ImageIO} codecs. No CPython and no new C/C++ image codecs.
 *
 * <p>Design:
 * <ul>
 *   <li>{@code core.*} — algorithm-equivalent replacement for {@code _imaging}</li>
 *   <li>{@link org.bytedeco.pytorch.vision.pillow.Image} — PIL.Image.Image + module factories</li>
 *   <li>{@code codec.*} — plugin registry (PNG/JPEG/BMP/GIF via ImageIO, PPM pure Java)</li>
 *   <li>{@code tensor.*} / {@code dataframe.*} — Tensor / ImageData / DataFrame batch bridges</li>
 * </ul>
 *
 * <p>Capability honesty is exposed via {@link org.bytedeco.pytorch.vision.pillow.features.Features}.
 * Unsupported codecs return {@code false} from {@code check_codec} and refuse silent empty images.
 *
 * @see org.bytedeco.pytorch.vision.pillow.Pillow
 * @see org.bytedeco.pytorch.vision.pillow.Image
 */
package org.bytedeco.pytorch.vision.pillow;
