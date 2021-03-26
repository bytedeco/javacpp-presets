package org.bytedeco.nvcodec.samples.callback;

import org.bytedeco.nvcodec.samples.decoder.NvDecoder;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.nvcodec.nvcuvid.CUVIDPICPARAMS;
import org.bytedeco.nvcodec.nvcuvid.PFNVIDDECODECALLBACK;

public class PfnVideoDecodeCallback extends PFNVIDDECODECALLBACK {
    private NvDecoder decoder;
    private static PfnVideoDecodeCallback instance;

    public static PfnVideoDecodeCallback getInstance() {
        if (instance == null) {
            instance = new PfnVideoDecodeCallback().retainReference();
        }

        return instance;
    }

    public void setDecoder(NvDecoder decoder) {
        this.decoder = decoder;
    }

    private PfnVideoDecodeCallback() {
        super();
    }

    @Override
    public int call(Pointer pointer, CUVIDPICPARAMS cuvidpicparams) {
        return this.decoder.handlePictureDecode(cuvidpicparams);
    }
}
