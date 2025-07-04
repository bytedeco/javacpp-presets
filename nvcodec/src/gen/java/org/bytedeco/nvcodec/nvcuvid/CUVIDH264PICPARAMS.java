// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.nvcodec.nvcuvid;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.nvcodec.global.nvcuvid.*;


/******************************************************/
/** \struct CUVIDH264PICPARAMS
/** H.264 picture parameters
/** This structure is used in CUVIDPICPARAMS structure
/******************************************************/
@Properties(inherit = org.bytedeco.nvcodec.presets.nvcuvid.class)
public class CUVIDH264PICPARAMS extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUVIDH264PICPARAMS() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUVIDH264PICPARAMS(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUVIDH264PICPARAMS(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUVIDH264PICPARAMS position(long position) {
        return (CUVIDH264PICPARAMS)super.position(position);
    }
    @Override public CUVIDH264PICPARAMS getPointer(long i) {
        return new CUVIDH264PICPARAMS((Pointer)this).offsetAddress(i);
    }

    // SPS
    public native int log2_max_frame_num_minus4(); public native CUVIDH264PICPARAMS log2_max_frame_num_minus4(int setter);
    public native int pic_order_cnt_type(); public native CUVIDH264PICPARAMS pic_order_cnt_type(int setter);
    public native int log2_max_pic_order_cnt_lsb_minus4(); public native CUVIDH264PICPARAMS log2_max_pic_order_cnt_lsb_minus4(int setter);
    public native int delta_pic_order_always_zero_flag(); public native CUVIDH264PICPARAMS delta_pic_order_always_zero_flag(int setter);
    public native int frame_mbs_only_flag(); public native CUVIDH264PICPARAMS frame_mbs_only_flag(int setter);
    public native int direct_8x8_inference_flag(); public native CUVIDH264PICPARAMS direct_8x8_inference_flag(int setter);
    public native int num_ref_frames(); public native CUVIDH264PICPARAMS num_ref_frames(int setter);             // NOTE: shall meet level 4.1 restrictions
    public native @Cast("unsigned char") byte residual_colour_transform_flag(); public native CUVIDH264PICPARAMS residual_colour_transform_flag(byte setter);
    public native @Cast("unsigned char") byte bit_depth_luma_minus8(); public native CUVIDH264PICPARAMS bit_depth_luma_minus8(byte setter);    // Must be 0 (only 8-bit supported)
    public native @Cast("unsigned char") byte bit_depth_chroma_minus8(); public native CUVIDH264PICPARAMS bit_depth_chroma_minus8(byte setter);  // Must be 0 (only 8-bit supported)
    public native @Cast("unsigned char") byte qpprime_y_zero_transform_bypass_flag(); public native CUVIDH264PICPARAMS qpprime_y_zero_transform_bypass_flag(byte setter);
    // PPS
    public native int entropy_coding_mode_flag(); public native CUVIDH264PICPARAMS entropy_coding_mode_flag(int setter);
    public native int pic_order_present_flag(); public native CUVIDH264PICPARAMS pic_order_present_flag(int setter);
    public native int num_ref_idx_l0_active_minus1(); public native CUVIDH264PICPARAMS num_ref_idx_l0_active_minus1(int setter);
    public native int num_ref_idx_l1_active_minus1(); public native CUVIDH264PICPARAMS num_ref_idx_l1_active_minus1(int setter);
    public native int weighted_pred_flag(); public native CUVIDH264PICPARAMS weighted_pred_flag(int setter);
    public native int weighted_bipred_idc(); public native CUVIDH264PICPARAMS weighted_bipred_idc(int setter);
    public native int pic_init_qp_minus26(); public native CUVIDH264PICPARAMS pic_init_qp_minus26(int setter);
    public native int deblocking_filter_control_present_flag(); public native CUVIDH264PICPARAMS deblocking_filter_control_present_flag(int setter);
    public native int redundant_pic_cnt_present_flag(); public native CUVIDH264PICPARAMS redundant_pic_cnt_present_flag(int setter);
    public native int transform_8x8_mode_flag(); public native CUVIDH264PICPARAMS transform_8x8_mode_flag(int setter);
    public native int MbaffFrameFlag(); public native CUVIDH264PICPARAMS MbaffFrameFlag(int setter);
    public native int constrained_intra_pred_flag(); public native CUVIDH264PICPARAMS constrained_intra_pred_flag(int setter);
    public native int chroma_qp_index_offset(); public native CUVIDH264PICPARAMS chroma_qp_index_offset(int setter);
    public native int second_chroma_qp_index_offset(); public native CUVIDH264PICPARAMS second_chroma_qp_index_offset(int setter);
    public native int ref_pic_flag(); public native CUVIDH264PICPARAMS ref_pic_flag(int setter);
    public native int frame_num(); public native CUVIDH264PICPARAMS frame_num(int setter);
    public native int CurrFieldOrderCnt(int i); public native CUVIDH264PICPARAMS CurrFieldOrderCnt(int i, int setter);
    @MemberGetter public native IntPointer CurrFieldOrderCnt();
    // DPB
    public native @ByRef CUVIDH264DPBENTRY dpb(int i); public native CUVIDH264PICPARAMS dpb(int i, CUVIDH264DPBENTRY setter);
    @MemberGetter public native CUVIDH264DPBENTRY dpb();          // List of reference frames within the DPB
    // Quantization Matrices (raster-order)
    public native @Cast("unsigned char") byte WeightScale4x4(int i, int j); public native CUVIDH264PICPARAMS WeightScale4x4(int i, int j, byte setter);
    @MemberGetter public native @Cast("unsigned char(* /*[6]*/ )[16]") BytePointer WeightScale4x4();
    public native @Cast("unsigned char") byte WeightScale8x8(int i, int j); public native CUVIDH264PICPARAMS WeightScale8x8(int i, int j, byte setter);
    @MemberGetter public native @Cast("unsigned char(* /*[2]*/ )[64]") BytePointer WeightScale8x8();
    // FMO/ASO
    public native @Cast("unsigned char") byte fmo_aso_enable(); public native CUVIDH264PICPARAMS fmo_aso_enable(byte setter);
    public native @Cast("unsigned char") byte num_slice_groups_minus1(); public native CUVIDH264PICPARAMS num_slice_groups_minus1(byte setter);
    public native @Cast("unsigned char") byte slice_group_map_type(); public native CUVIDH264PICPARAMS slice_group_map_type(byte setter);
    public native byte pic_init_qs_minus26(); public native CUVIDH264PICPARAMS pic_init_qs_minus26(byte setter);
    public native @Cast("unsigned int") int slice_group_change_rate_minus1(); public native CUVIDH264PICPARAMS slice_group_change_rate_minus1(int setter);
        @Name("fmo.slice_group_map_addr") public native @Cast("unsigned long long") long fmo_slice_group_map_addr(); public native CUVIDH264PICPARAMS fmo_slice_group_map_addr(long setter);
        @Name("fmo.pMb2SliceGroupMap") public native @Cast("const unsigned char*") BytePointer fmo_pMb2SliceGroupMap(); public native CUVIDH264PICPARAMS fmo_pMb2SliceGroupMap(BytePointer setter);
    public native @Cast("unsigned int") int Reserved(int i); public native CUVIDH264PICPARAMS Reserved(int i, int setter);
    @MemberGetter public native @Cast("unsigned int*") IntPointer Reserved();
    // SVC/MVC
        public native @ByRef CUVIDH264MVCEXT mvcext(); public native CUVIDH264PICPARAMS mvcext(CUVIDH264MVCEXT setter);
        public native @ByRef CUVIDH264SVCEXT svcext(); public native CUVIDH264PICPARAMS svcext(CUVIDH264SVCEXT setter);
}
