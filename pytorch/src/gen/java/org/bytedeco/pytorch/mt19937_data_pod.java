// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.pytorch;

import org.bytedeco.pytorch.Allocator;
import org.bytedeco.pytorch.Function;
import org.bytedeco.pytorch.Module;
import org.bytedeco.javacpp.annotation.Cast;
import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.javacpp.chrono.*;
import static org.bytedeco.javacpp.global.chrono.*;

import static org.bytedeco.pytorch.global.torch.*;


/**
 * Note [Mt19937 Engine implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Originally implemented in:
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/MTARCOK/mt19937ar-cok.c
 * and modified with C++ constructs. Moreover the state array of the engine
 * has been modified to hold 32 bit uints instead of 64 bits.
 *
 * Note that we reimplemented mt19937 instead of using std::mt19937 because,
 * at::mt19937 turns out to be faster in the pytorch codebase. PyTorch builds with -O2
 * by default and following are the benchmark numbers (benchmark code can be found at
 * https://github.com/syed-ahmed/benchmark-rngs):
 *
 * with -O2
 * Time to get 100000000 philox randoms with at::uniform_real_distribution = 0.462759s
 * Time to get 100000000 at::mt19937 randoms with at::uniform_real_distribution = 0.39628s
 * Time to get 100000000 std::mt19937 randoms with std::uniform_real_distribution = 0.352087s
 * Time to get 100000000 std::mt19937 randoms with at::uniform_real_distribution = 0.419454s
 *
 * std::mt19937 is faster when used in conjunction with std::uniform_real_distribution,
 * however we can't use std::uniform_real_distribution because of this bug:
 * http://open-std.org/JTC1/SC22/WG21/docs/lwg-active.html#2524. Plus, even if we used
 * std::uniform_real_distribution and filtered out the 1's, it is a different algorithm
 * than what's in pytorch currently and that messes up the tests in tests_distributions.py.
 * The other option, using std::mt19937 with at::uniform_real_distribution is a tad bit slower
 * than at::mt19937 with at::uniform_real_distribution and hence, we went with the latter.
 *
 * Copyright notice:
 * A C-program for MT19937, with initialization improved 2002/2/10.
 * Coded by Takuji Nishimura and Makoto Matsumoto.
 * This is a faster version by taking Shawn Cokus's optimization,
 * Matthe Bellew's simplification, Isaku Wada's real version.
 *
 * Before using, initialize the state by using init_genrand(seed)
 * or init_by_array(init_key, key_length).
 *
 * Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote
 *   products derived from this software without specific prior written
 *   permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * Any feedback is very welcome.
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 * email: m-mat \ math.sci.hiroshima-u.ac.jp (remove space)
 */

/**
 * mt19937_data_pod is used to get POD data in and out
 * of mt19937_engine. Used in torch.get_rng_state and
 * torch.set_rng_state functions.
 */
@Namespace("at") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class mt19937_data_pod extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public mt19937_data_pod() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public mt19937_data_pod(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public mt19937_data_pod(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public mt19937_data_pod position(long position) {
        return (mt19937_data_pod)super.position(position);
    }
    @Override public mt19937_data_pod getPointer(long i) {
        return new mt19937_data_pod((Pointer)this).offsetAddress(i);
    }

  public native @Cast("uint64_t") long seed_(); public native mt19937_data_pod seed_(long setter);
  public native int left_(); public native mt19937_data_pod left_(int setter);
  public native @Cast("bool") boolean seeded_(); public native mt19937_data_pod seeded_(boolean setter);
  public native @Cast("uint32_t") int next_(); public native mt19937_data_pod next_(int setter);
  public native @ByRef @Cast("std::array<uint32_t,at::MERSENNE_STATE_N>*") IntPointer state_(); public native mt19937_data_pod state_(IntPointer setter);
}
