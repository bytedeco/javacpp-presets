package org.bytedeco.kvazaar.samples.yuv;


public class YuvFrame {
    private byte[] lumaPlane;
    private byte[] crPlane;
    private byte[] cbPlane;

    public byte[] getLumaPlane() {
        return lumaPlane;
    }

    public void setLumaPlane(byte[] lumaPlane) {
        this.lumaPlane = lumaPlane;
    }

    public byte[] getCrPlane() {
        return crPlane;
    }

    public void setCrPlane(byte[] crPlane) {
        this.crPlane = crPlane;
    }

    public byte[] getCbPlane() {
        return cbPlane;
    }

    public void setCbPlane(byte[] cbPlane) {
        this.cbPlane = cbPlane;
    }
    
    
}
