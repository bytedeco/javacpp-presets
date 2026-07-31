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
package org.bytedeco.pytorch.vision.ffmpeg;

import org.bytedeco.ffmpeg.avutil.AVRational;

/**
 * Immutable rational number — PyAV {@code fractions.Fraction} / FFmpeg {@code AVRational}.
 */
public final class Rational {

    public final int num;
    public final int den;

    public Rational(int num, int den) {
        if (den == 0) {
            this.num = 0;
            this.den = 1;
        } else {
            this.num = num;
            this.den = den;
        }
    }

    public static Rational of(AVRational r) {
        if (r == null || r.isNull()) return new Rational(0, 1);
        return new Rational(r.num(), r.den());
    }

    public static Rational of(int num, int den) {
        return new Rational(num, den);
    }

    public static Rational fromDouble(double value, int maxDen) {
        if (Double.isNaN(value) || Double.isInfinite(value)) return new Rational(0, 1);
        // simple continued-fraction approximation
        int max = Math.max(1, maxDen);
        long a = (long) Math.floor(value);
        double x = value - a;
        long n0 = 1, d0 = 0, n1 = a, d1 = 1;
        for (int i = 0; i < 20 && x > 1e-12; i++) {
            x = 1.0 / x;
            a = (long) Math.floor(x);
            long n2 = a * n1 + n0;
            long d2 = a * d1 + d0;
            if (d2 > max) break;
            n0 = n1; d0 = d1; n1 = n2; d1 = d2;
            x = x - a;
        }
        return new Rational((int) n1, (int) Math.max(1, d1));
    }

    /** Convert to double (num/den). */
    public double toDouble() {
        return den == 0 ? 0.0 : (double) num / (double) den;
    }

    /** Multiply timestamp by this rational → seconds (or other unit). */
    public double mul(long ts) {
        return den == 0 ? 0.0 : (double) ts * num / den;
    }

    public AVRational toAV() {
        AVRational r = new AVRational();
        r.num(num);
        r.den(den);
        return r;
    }

    @Override
    public String toString() {
        return num + "/" + den;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Rational)) return false;
        Rational r = (Rational) o;
        return num == r.num && den == r.den;
    }

    @Override
    public int hashCode() {
        return 31 * num + den;
    }
}
