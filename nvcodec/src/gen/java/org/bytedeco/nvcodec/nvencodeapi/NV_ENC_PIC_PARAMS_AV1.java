// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.nvcodec.nvencodeapi;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;
import org.bytedeco.nvcodec.nvcuvid.*;
import static org.bytedeco.nvcodec.global.nvcuvid.*;

import static org.bytedeco.nvcodec.global.nvencodeapi.*;


/**
* \struct _NV_ENC_PIC_PARAMS_AV1
* AV1 specific enc pic params. sent on a per frame basis.
*/
@Properties(inherit = org.bytedeco.nvcodec.presets.nvencodeapi.class)
public class NV_ENC_PIC_PARAMS_AV1 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NV_ENC_PIC_PARAMS_AV1() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NV_ENC_PIC_PARAMS_AV1(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NV_ENC_PIC_PARAMS_AV1(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NV_ENC_PIC_PARAMS_AV1 position(long position) {
        return (NV_ENC_PIC_PARAMS_AV1)super.position(position);
    }
    @Override public NV_ENC_PIC_PARAMS_AV1 getPointer(long i) {
        return new NV_ENC_PIC_PARAMS_AV1((Pointer)this).offsetAddress(i);
    }

    /** [in]: Specifies the display POC syntax This is required to be set if client is handling the picture type decision. */
    public native @Cast("uint32_t") int displayPOCSyntax(); public native NV_ENC_PIC_PARAMS_AV1 displayPOCSyntax(int setter);
    /** [in]: Set to 1 for a reference picture. This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") int refPicFlag(); public native NV_ENC_PIC_PARAMS_AV1 refPicFlag(int setter);
    /** [in]: Specifies the temporal id of the picture */
    public native @Cast("uint32_t") int temporalId(); public native NV_ENC_PIC_PARAMS_AV1 temporalId(int setter);
    /** [in]: Forces an intra refresh with duration equal to intraRefreshFrameCnt.
                                                                        forceIntraRefreshWithFrameCnt cannot be used if B frames are used in the GOP structure specified */
    public native @Cast("uint32_t") int forceIntraRefreshWithFrameCnt(); public native NV_ENC_PIC_PARAMS_AV1 forceIntraRefreshWithFrameCnt(int setter);
    /** [in]: Encode frame as Golden Frame. This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") @NoOffset int goldenFrameFlag(); public native NV_ENC_PIC_PARAMS_AV1 goldenFrameFlag(int setter);
    /** [in]: Encode frame as Alternate Reference Frame. This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") @NoOffset int arfFrameFlag(); public native NV_ENC_PIC_PARAMS_AV1 arfFrameFlag(int setter);
    /** [in]: Encode frame as Alternate Reference 2 Frame. This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") @NoOffset int arf2FrameFlag(); public native NV_ENC_PIC_PARAMS_AV1 arf2FrameFlag(int setter);
    /** [in]: Encode frame as Backward Reference Frame. This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") @NoOffset int bwdFrameFlag(); public native NV_ENC_PIC_PARAMS_AV1 bwdFrameFlag(int setter);
    /** [in]: Encode frame as overlay frame. A previously encoded frame with the same displayPOCSyntax value should be present in reference frame buffer.
                                                                        This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") @NoOffset int overlayFrameFlag(); public native NV_ENC_PIC_PARAMS_AV1 overlayFrameFlag(int setter);
    /** [in]: When ovelayFrameFlag is set to 1, this flag controls the value of the show_existing_frame syntax element associated with the overlay frame.
                                                                        This flag is added to the interface as a placeholder. Its value is ignored for now and always assumed to be set to 1.
                                                                        This is ignored if NV_ENC_INITIALIZE_PARAMS::enablePTD is set to 1. */
    public native @Cast("uint32_t") @NoOffset int showExistingFrameFlag(); public native NV_ENC_PIC_PARAMS_AV1 showExistingFrameFlag(int setter);
    /** [in]: encode frame independently from previously encoded frames */
    public native @Cast("uint32_t") @NoOffset int errorResilientModeFlag(); public native NV_ENC_PIC_PARAMS_AV1 errorResilientModeFlag(int setter);

    /** [in]: Set to 1 if client wants to overwrite the default tile configuration with the tile parameters specified below
                                                                        When forceIntraRefreshWithFrameCnt is set it will have priority over tileConfigUpdate setting */
    public native @Cast("uint32_t") @NoOffset int tileConfigUpdate(); public native NV_ENC_PIC_PARAMS_AV1 tileConfigUpdate(int setter);
    /** [in]: Set 1 to enable custom tile configuration: numTileColumns and numTileRows must have non zero values and tileWidths and tileHeights must point to a valid address  */
    public native @Cast("uint32_t") @NoOffset int enableCustomTileConfig(); public native NV_ENC_PIC_PARAMS_AV1 enableCustomTileConfig(int setter);
    /** [in]: Set to 1 if client wants to update previous film grain parameters: filmGrainParams must point to a valid address and encoder must have been configured with film grain enabled  */
    public native @Cast("uint32_t") @NoOffset int filmGrainParamsUpdate(); public native NV_ENC_PIC_PARAMS_AV1 filmGrainParamsUpdate(int setter);
    /** [in]: Reserved bitfields and must be set to 0 */
    public native @Cast("uint32_t") @NoOffset int reservedBitFields(); public native NV_ENC_PIC_PARAMS_AV1 reservedBitFields(int setter);
    /** [in]: This parameter in conjunction with the flag enableCustomTileConfig and the array tileWidths[] specifies the way in which the picture is divided into tile columns.
                                                                        When enableCustomTileConfig == 0, the picture will be uniformly divided into numTileColumns tile columns. If numTileColumns is not a power of 2,
                                                                        it will be rounded down to the next power of 2 value. If numTileColumns == 0, the picture will be coded with the smallest number of vertical tiles as allowed by standard.
                                                                        When enableCustomTileConfig == 1, numTileColumns must be > 0 and <= NV_MAX_TILE_COLS_AV1 and tileWidths must point to a valid array of numTileColumns entries.
                                                                        Entry i specifies the width in 64x64 CTU unit of tile colum i. The sum of all the entries should be equal to the picture width in 64x64 CTU units. */
    public native @Cast("uint32_t") int numTileColumns(); public native NV_ENC_PIC_PARAMS_AV1 numTileColumns(int setter);
    /** [in]: This parameter in conjunction with the flag enableCustomTileConfig and the array tileHeights[] specifies the way in which the picture is divided into tiles rows
                                                                        When enableCustomTileConfig == 0, the picture will be uniformly divided into numTileRows tile rows. If numTileRows is not a power of 2,
                                                                        it will be rounded down to the next power of 2 value. If numTileRows == 0, the picture will be coded with the smallest number of horizontal tiles as allowed by standard.
                                                                        When enableCustomTileConfig == 1, numTileRows must be > 0 and <= NV_MAX_TILE_ROWS_AV1 and tileHeights must point to a valid array of numTileRows entries.
                                                                        Entry i specifies the height in 64x64 CTU unit of tile row i. The sum of all the entries should be equal to the picture hieght in 64x64 CTU units. */
    public native @Cast("uint32_t") int numTileRows(); public native NV_ENC_PIC_PARAMS_AV1 numTileRows(int setter);
    /** [in]: Reserved and must be set to 0. */
    public native @Cast("uint32_t") int reserved(); public native NV_ENC_PIC_PARAMS_AV1 reserved(int setter);
    /** [in]: If enableCustomTileConfig == 1, tileWidths[i] specifies the width of tile column i in 64x64 CTU unit, with 0 <= i <= numTileColumns -1. */
    public native @Cast("uint32_t*") IntPointer tileWidths(); public native NV_ENC_PIC_PARAMS_AV1 tileWidths(IntPointer setter);
    /** [in]: If enableCustomTileConfig == 1, tileHeights[i] specifies the height of tile row i in 64x64 CTU unit, with 0 <= i <= numTileRows -1. */
    public native @Cast("uint32_t*") IntPointer tileHeights(); public native NV_ENC_PIC_PARAMS_AV1 tileHeights(IntPointer setter);
    /** [in]: Specifies the number of elements allocated in  obuPayloadArray array. */
    public native @Cast("uint32_t") int obuPayloadArrayCnt(); public native NV_ENC_PIC_PARAMS_AV1 obuPayloadArrayCnt(int setter);
    /** [in]: Reserved and must be set to 0. */
    public native @Cast("uint32_t") int reserved1(); public native NV_ENC_PIC_PARAMS_AV1 reserved1(int setter);
    /** [in]: Array of OBU payloads which will be inserted for this frame. */
    public native NV_ENC_SEI_PAYLOAD obuPayloadArray(); public native NV_ENC_PIC_PARAMS_AV1 obuPayloadArray(NV_ENC_SEI_PAYLOAD setter);
    /** [in]: If filmGrainParamsUpdate == 1, filmGrainParams must point to a valid NV_ENC_FILM_GRAIN_PARAMS_AV1 structure */
    public native NV_ENC_FILM_GRAIN_PARAMS_AV1 filmGrainParams(); public native NV_ENC_PIC_PARAMS_AV1 filmGrainParams(NV_ENC_FILM_GRAIN_PARAMS_AV1 setter);
    /** [in]: Reserved and must be set to 0. */
    public native @Cast("uint32_t") int reserved2(int i); public native NV_ENC_PIC_PARAMS_AV1 reserved2(int i, int setter);
    @MemberGetter public native @Cast("uint32_t*") IntPointer reserved2();
    /** [in]: Reserved and must be set to NULL. */
    public native Pointer reserved3(int i); public native NV_ENC_PIC_PARAMS_AV1 reserved3(int i, Pointer setter);
    @MemberGetter public native @Cast("void**") PointerPointer reserved3();
}
