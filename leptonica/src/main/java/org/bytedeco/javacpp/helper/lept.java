/*
 * Copyright (C) 2018 Samuel Audet
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

package org.bytedeco.javacpp.helper;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexable;
import org.bytedeco.javacpp.indexer.UByteIndexer;

// required by javac to resolve circular dependencies
import org.bytedeco.javacpp.lept.*;
import static org.bytedeco.javacpp.lept.dpixClone;
import static org.bytedeco.javacpp.lept.dpixCreate;
import static org.bytedeco.javacpp.lept.dpixCreateTemplate;
import static org.bytedeco.javacpp.lept.dpixDestroy;
import static org.bytedeco.javacpp.lept.dpixGetData;
import static org.bytedeco.javacpp.lept.dpixGetDimensions;
import static org.bytedeco.javacpp.lept.dpixGetWpl;
import static org.bytedeco.javacpp.lept.fpixClone;
import static org.bytedeco.javacpp.lept.fpixCreate;
import static org.bytedeco.javacpp.lept.fpixCreateTemplate;
import static org.bytedeco.javacpp.lept.fpixDestroy;
import static org.bytedeco.javacpp.lept.fpixGetData;
import static org.bytedeco.javacpp.lept.fpixGetDimensions;
import static org.bytedeco.javacpp.lept.fpixGetWpl;
import static org.bytedeco.javacpp.lept.pixClone;
import static org.bytedeco.javacpp.lept.pixCreate;
import static org.bytedeco.javacpp.lept.pixCreateHeader;
import static org.bytedeco.javacpp.lept.pixCreateNoInit;
import static org.bytedeco.javacpp.lept.pixCreateTemplate;
import static org.bytedeco.javacpp.lept.pixCreateTemplateNoInit;
import static org.bytedeco.javacpp.lept.pixDestroy;
import static org.bytedeco.javacpp.lept.pixGetData;
import static org.bytedeco.javacpp.lept.pixGetDepth;
import static org.bytedeco.javacpp.lept.pixGetHeight;
import static org.bytedeco.javacpp.lept.pixGetWidth;
import static org.bytedeco.javacpp.lept.pixGetWpl;

public class lept extends org.bytedeco.javacpp.presets.lept {

    public static abstract class AbstractPIX extends Pointer implements Indexable {
        protected static class DestroyDeallocator extends PIX implements Pointer.Deallocator {
            boolean header = false;
            DestroyDeallocator(PIX p) { this(p, false); }
            DestroyDeallocator(PIX p, boolean header) { super(p); this.header = header; }
            @Override public void deallocate() { if (isNull()) return; if (header) this.data(null); pixDestroy(this); setNull(); }
        }

        public AbstractPIX(Pointer p) { super(p); }

        /**
         * Calls pixCreate(), and registers a deallocator.
         * @return PIX created. Do not call pixDestroy() on it.
         */
        public static PIX create(int width, int height, int depth) {
            PIX p = pixCreate(width, height, depth);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls pixCreateNoInit(), and registers a deallocator.
         * @return PIX created. Do not call pixDestroy() on it.
         */
        public static PIX createNoInit(int width, int height, int depth) {
            PIX p = pixCreateNoInit(width, height, depth);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls pixCreateTemplate(), and registers a deallocator.
         * @return PIX created. Do not call pixDestroy() on it.
         */
        public static PIX createTemplate(PIX pixs) {
            PIX p = pixCreateTemplate(pixs);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls pixCreateTemplateNoInit(), and registers a deallocator.
         * @return PIX created. Do not call pixDestroy() on it.
         */
        public static PIX createTemplateNoInit(PIX pixs) {
            PIX p = pixCreateTemplateNoInit(pixs);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls pixCreateHeader(), and registers a deallocator.
         * @return PIX created. Do not call pixDestroy() on it.
         */
        public static PIX createHeader(int width, int height, int depth) {
            PIX p = pixCreateHeader(width, height, depth);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p, true));
            }
            return p;
        }

        /**
         * Calls pixClone(), and registers a deallocator.
         * @return PIX cloned. Do not call pixDestroy() on it.
         */
        @Override public PIX clone() {
            PIX p = pixClone((PIX)this);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /** @return {@code createBuffer(0)} */
        public ByteBuffer createBuffer() {
            return createBuffer(0);
        }
        /** @return {@link PIX#data()} wrapped in a {@link ByteBuffer} starting at given byte index. */
        public ByteBuffer createBuffer(int index) {
            int h = pixGetHeight((PIX)this);
            int wpl = pixGetWpl((PIX)this);
            BytePointer data = new BytePointer(pixGetData((PIX)this)).position(index).capacity(h * wpl * 4);
            return data.asByteBuffer();
        }

        /** @return {@code createIndexer(true)} */
        public UByteIndexer createIndexer() {
            return createIndexer(true);
        }
        @Override public UByteIndexer createIndexer(boolean direct) {
            int w = pixGetWidth((PIX)this);
            int h = pixGetHeight((PIX)this);
            int d = pixGetDepth((PIX)this);
            int wpl = pixGetWpl((PIX)this);
            long[] sizes = {h, w, d / 8};
            long[] strides = {wpl * 4, d / 8, 1};
            BytePointer data = new BytePointer(pixGetData((PIX)this)).capacity(h * wpl * 4);
            return UByteIndexer.create(data, sizes, strides, direct);
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void destroy() {
            deallocate();
        }
    }

    public static abstract class AbstractFPIX extends Pointer implements Indexable {
        protected static class DestroyDeallocator extends FPIX implements Pointer.Deallocator {
            DestroyDeallocator(FPIX p) { super(p); }
            @Override public void deallocate() { if (isNull()) return; fpixDestroy(this); setNull(); }
        }

        public AbstractFPIX(Pointer p) { super(p); }

        /**
         * Calls fpixCreate(), and registers a deallocator.
         * @return FPIX created. Do not call fpixDestroy() on it.
         */
        public static FPIX create(int width, int height) {
            FPIX p = fpixCreate(width, height);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls fpixCreateTemplate(), and registers a deallocator.
         * @return FPIX created. Do not call fpixDestroy() on it.
         */
        public static FPIX createTemplate(FPIX fpixs) {
            FPIX p = fpixCreateTemplate(fpixs);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls fpixClone(), and registers a deallocator.
         * @return FPIX cloned. Do not call fpixDestroy() on it.
         */
        @Override public FPIX clone() {
            FPIX p = fpixClone((FPIX)this);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /** @return {@code createBuffer(0)} */
        public FloatBuffer createBuffer() {
            return createBuffer(0);
        }
        /** @return {@link FPIX#data()} wrapped in a {@link FloatBuffer} starting at given byte index. */
        public FloatBuffer createBuffer(int index) {
            int[] w = {0}, h = {0};
            fpixGetDimensions((FPIX)this, w, h);
            int wpl = fpixGetWpl((FPIX)this);
            FloatPointer data = new FloatPointer(fpixGetData((FPIX)this)).position(index).capacity(h[0] * wpl);
            return data.asBuffer();
        }

        /** @return {@code createIndexer(true)} */
        public FloatIndexer createIndexer() {
            return createIndexer(true);
        }
        @Override public FloatIndexer createIndexer(boolean direct) {
            int[] w = {0}, h = {0};
            fpixGetDimensions((FPIX)this, w, h);
            int wpl = fpixGetWpl((FPIX)this);
            long[] sizes = {h[0], w[0], wpl / w[0]};
            long[] strides = {wpl, wpl / w[0], 1};
            FloatPointer data = new FloatPointer(fpixGetData((FPIX)this)).capacity(h[0] * wpl);
            return FloatIndexer.create(data, sizes, strides, direct);
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void destroy() {
            deallocate();
        }
    }

    public static abstract class AbstractDPIX extends Pointer implements Indexable {
        protected static class DestroyDeallocator extends DPIX implements Pointer.Deallocator {
            DestroyDeallocator(DPIX p) { super(p); }
            @Override public void deallocate() { if (isNull()) return; dpixDestroy(this); setNull(); }
        }

        public AbstractDPIX(Pointer p) { super(p); }

        /**
         * Calls dpixCreate(), and registers a deallocator.
         * @return DPIX created. Do not call dpixDestroy() on it.
         */
        public static DPIX create(int width, int height) {
            DPIX p = dpixCreate(width, height);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls dpixCreateTemplate(), and registers a deallocator.
         * @return DPIX created. Do not call dpixDestroy() on it.
         */
        public static DPIX createTemplate(DPIX dpixs) {
            DPIX p = dpixCreateTemplate(dpixs);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /**
         * Calls dpixClone(), and registers a deallocator.
         * @return DPIX cloned. Do not call dpixDestroy() on it.
         */
        @Override public DPIX clone() {
            DPIX p = dpixClone((DPIX)this);
            if (p != null) {
                p.deallocator(new DestroyDeallocator(p));
            }
            return p;
        }

        /** @return {@code createBuffer(0)} */
        public DoubleBuffer createBuffer() {
            return createBuffer(0);
        }
        /** @return {@link DPIX#data()} wrapped in a {@link DoubleBuffer} starting at given byte index. */
        public DoubleBuffer createBuffer(int index) {
            int[] w = {0}, h = {0};
            dpixGetDimensions((DPIX)this, w, h);
            int wpl = dpixGetWpl((DPIX)this);
            DoublePointer data = new DoublePointer(dpixGetData((DPIX)this)).position(index).capacity(h[0] * wpl);
            return data.asBuffer();
        }

        /** @return {@code createIndexer(true)} */
        public DoubleIndexer createIndexer() {
            return createIndexer(true);
        }
        @Override public DoubleIndexer createIndexer(boolean direct) {
            int[] w = {0}, h = {0};
            dpixGetDimensions((DPIX)this, w, h);
            int wpl = dpixGetWpl((DPIX)this);
            long[] sizes = {h[0], w[0], wpl / w[0]};
            long[] strides = {wpl, wpl / w[0], 1};
            DoublePointer data = new DoublePointer(dpixGetData((DPIX)this)).capacity(h[0] * wpl);
            return DoubleIndexer.create(data, sizes, strides, direct);
        }

        /**
         * Calls the deallocator, if registered, otherwise has no effect.
         */
        public void destroy() {
            deallocate();
        }
    }
}
