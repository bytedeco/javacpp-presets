import org.bytedeco.javacpp.*;
import org.bytedeco.libfreenect2.*;
import static org.bytedeco.libfreenect2.global.freenect2.*;

/**
 *
 * @author Jeremy Laviole
 */
public class freenect2Example {
    public static void main(String[] args) {
        Freenect2 freenect2Context;
        try {
            Loader.load(org.bytedeco.libfreenect2.global.freenect2.class);
            // Context is shared accross cameras.
            freenect2Context = new Freenect2();
        } catch (Exception e) {
            System.out.println("Exception in the TryLoad !" + e);
            e.printStackTrace();
            return;
        }
        Freenect2Device device = null;
        PacketPipeline pipeline = null;
        String serial = "";

        // Only CPU pipeline tested.
        pipeline = new CpuPacketPipeline();
//        pipeline = new libfreenect2::OpenGLPacketPipeline();
//        pipeline = new libfreenect2::OpenCLPacketPipeline(deviceId);
//        pipeline = new libfreenect2::CudaPacketPipeline(deviceId);

        if (serial == "") {
            serial = freenect2Context.getDefaultDeviceSerialNumber().getString();
            System.out.println("Serial:" + serial);
        }

        device = freenect2Context.openDevice(serial, pipeline);
        // [listeners]
        int types = 0;
        types |= Frame.Color;
        types |= Frame.Ir | Frame.Depth;

        SyncMultiFrameListener listener = new SyncMultiFrameListener(types);

        device.setColorFrameListener(listener);
        device.setIrAndDepthFrameListener(listener);

        device.start();

        System.out.println("Serial: " + device.getSerialNumber().getString());
        System.out.println("Firmware: " + device.getFirmwareVersion().getString());
/// [start]

        FrameMap frames = new FrameMap();
        // Fetch 100 frames.
        int frameCount = 0;
        for (int i = 0; i < 100; i++) {
            System.out.println("getting frame " + frameCount);
            if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 sconds
            {
                System.out.println("timeout!");
                return;
            }

            Frame rgb = frames.get(Frame.Color);
            Frame ir = frames.get(Frame.Ir);
            Frame depth = frames.get(Frame.Depth);
/// [loop start]
            System.out.println("RGB image, w:" + rgb.width() + " " + rgb.height());
            byte[] imgData = new byte[1000];
            rgb.data().get(imgData);
            for (int pix = 0; pix < 10; pix++) {
                System.out.print(imgData[pix] + " ");
            }
            System.out.println();
            frameCount++;
            listener.release(frames);
            continue;
        }
        device.stop();
        device.close();
    }
}
