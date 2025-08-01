// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnx;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.onnx.global.onnx.*;


// Defined in command_line_interface.cc
@Namespace("google::protobuf::compiler") @Opaque @Properties(inherit = org.bytedeco.onnx.presets.onnx.class)
public class CommandLineInterface extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public CommandLineInterface() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CommandLineInterface(Pointer p) { super(p); }
}
