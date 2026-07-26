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
package org.bytedeco.pytorch.utils.spacy.vocab;

/**
 * Lexical entry for a string orth form.
 */
public final class Lexeme {

    private final String orth;
    private double[] vector;
    private String cluster;
    private boolean isStop;
    private boolean isOov = true;

    public Lexeme(String orth) {
        this.orth = orth;
    }

    public String getKey() {
        return orth;
    }

    public String orth() {
        return orth;
    }

    public String text() {
        return orth;
    }

    public double[] getVector() {
        return vector;
    }

    public void setVector(double[] v) {
        this.vector = v;
        if (v != null) {
            isOov = false;
        }
    }

    public boolean hasVector() {
        return vector != null && vector.length > 0;
    }

    public String getCluster() {
        return cluster;
    }

    public void setCluster(String cluster) {
        this.cluster = cluster;
    }

    public boolean isStop() {
        return isStop;
    }

    public void setStop(boolean stop) {
        isStop = stop;
    }

    public boolean isOov() {
        return isOov;
    }

    public void setOov(boolean oov) {
        isOov = oov;
    }

    public String lower() {
        return orth == null ? "" : orth.toLowerCase(java.util.Locale.ROOT);
    }

    public String shape() {
        if (orth == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder(orth.length());
        for (int i = 0; i < orth.length(); i++) {
            char c = orth.charAt(i);
            if (Character.isUpperCase(c)) {
                sb.append('X');
            } else if (Character.isLowerCase(c)) {
                sb.append('x');
            } else if (Character.isDigit(c)) {
                sb.append('d');
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return "Lexeme(" + orth + ")";
    }
}
