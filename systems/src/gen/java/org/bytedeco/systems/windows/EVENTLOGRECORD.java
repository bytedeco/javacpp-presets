// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


//
// Structure that defines the header of the Eventlog record. This is the
// fixed-sized portion before all the variable-length strings, binary
// data and pad bytes.
//
// TimeGenerated and TimeWritten are the time the event was put into the log at the server end.
//

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class EVENTLOGRECORD extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public EVENTLOGRECORD() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public EVENTLOGRECORD(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public EVENTLOGRECORD(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public EVENTLOGRECORD position(long position) {
        return (EVENTLOGRECORD)super.position(position);
    }
    @Override public EVENTLOGRECORD getPointer(long i) {
        return new EVENTLOGRECORD((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int Length(); public native EVENTLOGRECORD Length(int setter);        // Length of full record
    public native @Cast("DWORD") int Reserved(); public native EVENTLOGRECORD Reserved(int setter);      // Used by the service
    public native @Cast("DWORD") int RecordNumber(); public native EVENTLOGRECORD RecordNumber(int setter);  // Absolute record number
    public native @Cast("DWORD") int TimeGenerated(); public native EVENTLOGRECORD TimeGenerated(int setter); // Seconds since 1-1-1970
    public native @Cast("DWORD") int TimeWritten(); public native EVENTLOGRECORD TimeWritten(int setter);   // Seconds since 1-1-1970
    public native @Cast("DWORD") int EventID(); public native EVENTLOGRECORD EventID(int setter);
    public native @Cast("WORD") short EventType(); public native EVENTLOGRECORD EventType(short setter);
    public native @Cast("WORD") short NumStrings(); public native EVENTLOGRECORD NumStrings(short setter);
    public native @Cast("WORD") short EventCategory(); public native EVENTLOGRECORD EventCategory(short setter);
    public native @Cast("WORD") short ReservedFlags(); public native EVENTLOGRECORD ReservedFlags(short setter);
    public native @Cast("DWORD") int ClosingRecordNumber(); public native EVENTLOGRECORD ClosingRecordNumber(int setter); // Reserved
    public native @Cast("DWORD") int StringOffset(); public native EVENTLOGRECORD StringOffset(int setter);  // Offset from beginning of record
    public native @Cast("DWORD") int UserSidLength(); public native EVENTLOGRECORD UserSidLength(int setter);
    public native @Cast("DWORD") int UserSidOffset(); public native EVENTLOGRECORD UserSidOffset(int setter);
    public native @Cast("DWORD") int DataLength(); public native EVENTLOGRECORD DataLength(int setter);
    public native @Cast("DWORD") int DataOffset(); public native EVENTLOGRECORD DataOffset(int setter);    // Offset from beginning of record
    //
    // Then follow:
    //
    // WCHAR SourceName[]
    // WCHAR Computername[]
    // SID   UserSid
    // WCHAR Strings[]
    // BYTE  Data[]
    // CHAR  Pad[]
    // DWORD Length;
    //
}
