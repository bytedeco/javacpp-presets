// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.leptonica;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.leptonica.global.leptonica.*;


/**
 * \file watershed.h
 *
 *     Simple data structure to hold watershed data.
 *     All data here is owned by the L_WShed and must be freed.
 */

/** Simple data structure to hold watershed data. */
@Name("L_WShed") @Properties(inherit = org.bytedeco.leptonica.presets.leptonica.class)
public class L_WSHED extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public L_WSHED() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public L_WSHED(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L_WSHED(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public L_WSHED position(long position) {
        return (L_WSHED)super.position(position);
    }
    @Override public L_WSHED getPointer(long i) {
        return new L_WSHED((Pointer)this).offsetAddress(i);
    }

    /** clone of input 8 bpp pixs                */
    public native PIX pixs(); public native L_WSHED pixs(PIX setter);
    /** clone of input 1 bpp seed (marker) pixm  */
    public native PIX pixm(); public native L_WSHED pixm(PIX setter);
    /** minimum depth allowed for a watershed    */
    public native @Cast("l_int32") int mindepth(); public native L_WSHED mindepth(int setter);
    /** 16 bpp label pix                         */
    public native PIX pixlab(); public native L_WSHED pixlab(PIX setter);
    /** scratch pix for computing wshed regions  */
    public native PIX pixt(); public native L_WSHED pixt(PIX setter);
    /** line ptrs for pixs                       */
    public native Pointer lines8(int i); public native L_WSHED lines8(int i, Pointer setter);
    public native @Cast("void**") PointerPointer lines8(); public native L_WSHED lines8(PointerPointer setter);
    /** line ptrs for pixm                       */
    public native Pointer linem1(int i); public native L_WSHED linem1(int i, Pointer setter);
    public native @Cast("void**") PointerPointer linem1(); public native L_WSHED linem1(PointerPointer setter);
    /** line ptrs for pixlab                     */
    public native Pointer linelab32(int i); public native L_WSHED linelab32(int i, Pointer setter);
    public native @Cast("void**") PointerPointer linelab32(); public native L_WSHED linelab32(PointerPointer setter);
    /** line ptrs for pixt                       */
    public native Pointer linet1(int i); public native L_WSHED linet1(int i, Pointer setter);
    public native @Cast("void**") PointerPointer linet1(); public native L_WSHED linet1(PointerPointer setter);
    /** result: 1 bpp pixa of watersheds         */
    public native PIXA pixad(); public native L_WSHED pixad(PIXA setter);
    /** pta of initial seed pixels               */
    public native PTA ptas(); public native L_WSHED ptas(PTA setter);
    /** numa of seed indicators; 0 if completed  */
    public native NUMA nasi(); public native L_WSHED nasi(NUMA setter);
    /** numa of initial seed heights             */
    public native NUMA nash(); public native L_WSHED nash(NUMA setter);
    /** numa of initial minima heights           */
    public native NUMA namh(); public native L_WSHED namh(NUMA setter);
    /** result: numa of watershed levels         */
    public native NUMA nalevels(); public native L_WSHED nalevels(NUMA setter);
    /** number of seeds (markers)                */
    public native @Cast("l_int32") int nseeds(); public native L_WSHED nseeds(int setter);
    /** number of minima different from seeds    */
    public native @Cast("l_int32") int nother(); public native L_WSHED nother(int setter);
    /** lut for pixel indices                    */
    public native @Cast("l_int32*") IntPointer lut(); public native L_WSHED lut(IntPointer setter);
    /** back-links into lut, for updates         */
    public native NUMA links(int i); public native L_WSHED links(int i, NUMA setter);
    public native @Cast("Numa**") PointerPointer links(); public native L_WSHED links(PointerPointer setter);
    /** size of links array                      */
    public native @Cast("l_int32") int arraysize(); public native L_WSHED arraysize(int setter);
    /** set to 1 for debug output                */
    public native @Cast("l_int32") int debug(); public native L_WSHED debug(int setter);
}
