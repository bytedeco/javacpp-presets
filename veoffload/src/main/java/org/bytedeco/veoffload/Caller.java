/*
 * Copyright (C) 2019-2021 Samuel Audet
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

package org.bytedeco.veoffload;

import java.lang.reflect.Field;
import org.bytedeco.javacpp.*;
import org.bytedeco.veoffload.*;
import static org.bytedeco.veoffload.global.veo.*;

public class Caller implements AutoCloseable {
    veo_proc_handle proc;
    long handle;
    veo_thr_ctxt ctx;
    Field pointerArrayField;

    public Caller(int venode, String libname) {
        try {
            proc = veo_proc_create(venode);
            handle = veo_load_library(proc, libname);
            ctx = veo_context_open(proc);
            pointerArrayField = PointerPointer.class.getDeclaredField("pointerArray");
            pointerArrayField.setAccessible(true);
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    public Object call(String symname, Object... args) {
        long sym = veo_get_sym(proc, handle, symname);
        veo_args argp = veo_args_alloc();
        long[] pointers = new long[args.length];
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            if (arg instanceof Integer) {
                veo_args_set_i32(argp, i, (Integer)arg);
            } else if (arg instanceof Long) {
                veo_args_set_i64(argp, i, (Long)arg);
            } else if (arg instanceof Float) {
                veo_args_set_float(argp, i, (Float)arg);
            } else if (arg instanceof Double) {
                veo_args_set_double(argp, i, (Double)arg);
            } else if (arg instanceof Pointer) {
                Pointer p = (Pointer)arg;
                if (p.limit() <= 0) {
                    veo_args_set_i64(argp, i, p.address() + p.position() * p.sizeof());
                } else {
                    long size = (p.limit() - p.position()) * p.sizeof();
                    long[] addr = {0};
                    veo_alloc_mem(proc, addr, size);
                    veo_args_set_i64(argp, i, pointers[i] = addr[0]);
                    if (p instanceof PointerPointer) {
                        PointerPointer pp = (PointerPointer)p;
                        try {
                            Pointer[] array = (Pointer[])pointerArrayField.get(pp);
                            for (int j = 0; j < array.length; j++) {
                                Pointer p2 = array[j];
                                long size2 = (p2.limit() - p2.position()) * p2.sizeof();
                                LongPointer addr2 = new LongPointer(1);
                                veo_alloc_mem(proc, addr2, size2);
                                veo_write_mem(proc, addr2.get(0), p2, size2);
                                veo_write_mem(proc, pointers[i] + j * 8, addr2, 8);
                            }
                        } catch (Exception ex) {
                            throw new RuntimeException(ex);
                        }
                    } else {
                        veo_write_mem(proc, pointers[i], p, size);
                    }
                }
            } else {
                throw new UnsupportedOperationException("Not supported yet.");
            }
        }
        long id = veo_call_async(ctx, sym, argp);
        long[] retval = {0};
        veo_call_wait_result(ctx, id, retval);
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            if (pointers[i] != 0) {
                Pointer p = (Pointer)arg;
                long size = (p.limit() - p.position()) * p.sizeof();
                if (p instanceof PointerPointer) {
                    PointerPointer pp = (PointerPointer)p;
                    try {
                        Pointer[] array = (Pointer[])pointerArrayField.get(pp);
                        for (int j = 0; j < array.length; j++) {
                            Pointer p2 = array[j];
                            long size2 = (p2.limit() - p2.position()) * p2.sizeof();
                            LongPointer addr2 = new LongPointer(1);
                            veo_read_mem(proc, addr2, pointers[i] + j * 8, 8);
                            veo_read_mem(proc, p2, addr2.get(0), size2);
                            veo_free_mem(proc, addr2.get(0));
                        }
                    } catch (Exception ex) {
                        throw new RuntimeException(ex);
                    }
                } else {
                    veo_read_mem(proc, p, pointers[i], size);
                }
                veo_free_mem(proc, pointers[i]);
            }
        }
        veo_args_free(argp);
        return retval[0];
    }

    @Override public void close() {
        veo_context_close(ctx);
    }
}
