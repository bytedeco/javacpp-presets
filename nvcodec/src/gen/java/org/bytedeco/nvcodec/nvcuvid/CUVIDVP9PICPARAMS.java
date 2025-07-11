// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.nvcodec.nvcuvid;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.nvcodec.global.nvcuvid.*;


/***********************************************************/
/** \struct CUVIDVP9PICPARAMS
/** VP9 picture parameters
/** This structure is used in CUVIDPICPARAMS structure
/***********************************************************/
@Properties(inherit = org.bytedeco.nvcodec.presets.nvcuvid.class)
public class CUVIDVP9PICPARAMS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUVIDVP9PICPARAMS() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUVIDVP9PICPARAMS(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUVIDVP9PICPARAMS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUVIDVP9PICPARAMS position(long position) {
        return (CUVIDVP9PICPARAMS)super.position(position);
    }
    @Override public CUVIDVP9PICPARAMS getPointer(long i) {
        return new CUVIDVP9PICPARAMS((Pointer)this).offsetAddress(i);
    }

    public native @Cast("unsigned int") int width(); public native CUVIDVP9PICPARAMS width(int setter);
    public native @Cast("unsigned int") int height(); public native CUVIDVP9PICPARAMS height(int setter);

    //Frame Indices
    public native @Cast("unsigned char") byte LastRefIdx(); public native CUVIDVP9PICPARAMS LastRefIdx(byte setter);
    public native @Cast("unsigned char") byte GoldenRefIdx(); public native CUVIDVP9PICPARAMS GoldenRefIdx(byte setter);
    public native @Cast("unsigned char") byte AltRefIdx(); public native CUVIDVP9PICPARAMS AltRefIdx(byte setter);
    public native @Cast("unsigned char") byte colorSpace(); public native CUVIDVP9PICPARAMS colorSpace(byte setter);

    public native @Cast("unsigned short") @NoOffset short profile(); public native CUVIDVP9PICPARAMS profile(short setter);
    public native @Cast("unsigned short") @NoOffset short frameContextIdx(); public native CUVIDVP9PICPARAMS frameContextIdx(short setter);
    public native @Cast("unsigned short") @NoOffset short frameType(); public native CUVIDVP9PICPARAMS frameType(short setter);
    public native @Cast("unsigned short") @NoOffset short showFrame(); public native CUVIDVP9PICPARAMS showFrame(short setter);
    public native @Cast("unsigned short") @NoOffset short errorResilient(); public native CUVIDVP9PICPARAMS errorResilient(short setter);
    public native @Cast("unsigned short") @NoOffset short frameParallelDecoding(); public native CUVIDVP9PICPARAMS frameParallelDecoding(short setter);
    public native @Cast("unsigned short") @NoOffset short subSamplingX(); public native CUVIDVP9PICPARAMS subSamplingX(short setter);
    public native @Cast("unsigned short") @NoOffset short subSamplingY(); public native CUVIDVP9PICPARAMS subSamplingY(short setter);
    public native @Cast("unsigned short") @NoOffset short intraOnly(); public native CUVIDVP9PICPARAMS intraOnly(short setter);
    public native @Cast("unsigned short") @NoOffset short allow_high_precision_mv(); public native CUVIDVP9PICPARAMS allow_high_precision_mv(short setter);
    public native @Cast("unsigned short") @NoOffset short refreshEntropyProbs(); public native CUVIDVP9PICPARAMS refreshEntropyProbs(short setter);
    public native @Cast("unsigned short") @NoOffset short reserved2Bits(); public native CUVIDVP9PICPARAMS reserved2Bits(short setter);

    public native @Cast("unsigned short") short reserved16Bits(); public native CUVIDVP9PICPARAMS reserved16Bits(short setter);

    public native @Cast("unsigned char") byte refFrameSignBias(int i); public native CUVIDVP9PICPARAMS refFrameSignBias(int i, byte setter);
    @MemberGetter public native @Cast("unsigned char*") BytePointer refFrameSignBias();

    public native @Cast("unsigned char") byte bitDepthMinus8Luma(); public native CUVIDVP9PICPARAMS bitDepthMinus8Luma(byte setter);
    public native @Cast("unsigned char") byte bitDepthMinus8Chroma(); public native CUVIDVP9PICPARAMS bitDepthMinus8Chroma(byte setter);
    public native @Cast("unsigned char") byte loopFilterLevel(); public native CUVIDVP9PICPARAMS loopFilterLevel(byte setter);
    public native @Cast("unsigned char") byte loopFilterSharpness(); public native CUVIDVP9PICPARAMS loopFilterSharpness(byte setter);

    public native @Cast("unsigned char") byte modeRefLfEnabled(); public native CUVIDVP9PICPARAMS modeRefLfEnabled(byte setter);
    public native @Cast("unsigned char") byte log2_tile_columns(); public native CUVIDVP9PICPARAMS log2_tile_columns(byte setter);
    public native @Cast("unsigned char") byte log2_tile_rows(); public native CUVIDVP9PICPARAMS log2_tile_rows(byte setter);

    public native @Cast("unsigned char") @NoOffset byte segmentEnabled(); public native CUVIDVP9PICPARAMS segmentEnabled(byte setter);
    public native @Cast("unsigned char") @NoOffset byte segmentMapUpdate(); public native CUVIDVP9PICPARAMS segmentMapUpdate(byte setter);
    public native @Cast("unsigned char") @NoOffset byte segmentMapTemporalUpdate(); public native CUVIDVP9PICPARAMS segmentMapTemporalUpdate(byte setter);
    public native @Cast("unsigned char") @NoOffset byte segmentFeatureMode(); public native CUVIDVP9PICPARAMS segmentFeatureMode(byte setter);
    public native @Cast("unsigned char") @NoOffset byte reserved4Bits(); public native CUVIDVP9PICPARAMS reserved4Bits(byte setter);


    public native @Cast("unsigned char") byte segmentFeatureEnable(int i, int j); public native CUVIDVP9PICPARAMS segmentFeatureEnable(int i, int j, byte setter);
    @MemberGetter public native @Cast("unsigned char(* /*[8]*/ )[4]") BytePointer segmentFeatureEnable();
    public native short segmentFeatureData(int i, int j); public native CUVIDVP9PICPARAMS segmentFeatureData(int i, int j, short setter);
    @MemberGetter public native @Cast("short(* /*[8]*/ )[4]") ShortPointer segmentFeatureData();
    public native @Cast("unsigned char") byte mb_segment_tree_probs(int i); public native CUVIDVP9PICPARAMS mb_segment_tree_probs(int i, byte setter);
    @MemberGetter public native @Cast("unsigned char*") BytePointer mb_segment_tree_probs();
    public native @Cast("unsigned char") byte segment_pred_probs(int i); public native CUVIDVP9PICPARAMS segment_pred_probs(int i, byte setter);
    @MemberGetter public native @Cast("unsigned char*") BytePointer segment_pred_probs();
    public native @Cast("unsigned char") byte reservedSegment16Bits(int i); public native CUVIDVP9PICPARAMS reservedSegment16Bits(int i, byte setter);
    @MemberGetter public native @Cast("unsigned char*") BytePointer reservedSegment16Bits();

    public native int qpYAc(); public native CUVIDVP9PICPARAMS qpYAc(int setter);
    public native int qpYDc(); public native CUVIDVP9PICPARAMS qpYDc(int setter);
    public native int qpChDc(); public native CUVIDVP9PICPARAMS qpChDc(int setter);
    public native int qpChAc(); public native CUVIDVP9PICPARAMS qpChAc(int setter);

    public native @Cast("unsigned int") int activeRefIdx(int i); public native CUVIDVP9PICPARAMS activeRefIdx(int i, int setter);
    @MemberGetter public native @Cast("unsigned int*") IntPointer activeRefIdx();
    public native @Cast("unsigned int") int resetFrameContext(); public native CUVIDVP9PICPARAMS resetFrameContext(int setter);
    public native @Cast("unsigned int") int mcomp_filter_type(); public native CUVIDVP9PICPARAMS mcomp_filter_type(int setter);
    public native @Cast("unsigned int") int mbRefLfDelta(int i); public native CUVIDVP9PICPARAMS mbRefLfDelta(int i, int setter);
    @MemberGetter public native @Cast("unsigned int*") IntPointer mbRefLfDelta();
    public native @Cast("unsigned int") int mbModeLfDelta(int i); public native CUVIDVP9PICPARAMS mbModeLfDelta(int i, int setter);
    @MemberGetter public native @Cast("unsigned int*") IntPointer mbModeLfDelta();
    public native @Cast("unsigned int") int frameTagSize(); public native CUVIDVP9PICPARAMS frameTagSize(int setter);
    public native @Cast("unsigned int") int offsetToDctParts(); public native CUVIDVP9PICPARAMS offsetToDctParts(int setter);
    public native @Cast("unsigned int") int reserved128Bits(int i); public native CUVIDVP9PICPARAMS reserved128Bits(int i, int setter);
    @MemberGetter public native @Cast("unsigned int*") IntPointer reserved128Bits();

}
