import java.lang.System;
import org.bytedeco.javacpp.*;
import org.bytedeco.openpose.*;
import org.bytedeco.opencv.opencv_core.Mat;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.openpose.global.openpose.ThreadManagerMode;
import static org.bytedeco.openpose.global.openpose.OP_CV2OPCONSTMAT;
import static org.bytedeco.openpose.global.openpose.OP_OP2CVCONSTMAT;
import static org.bytedeco.openpose.global.openpose.PoseMode;
import static org.bytedeco.openpose.global.openpose.ScaleMode;
import static org.bytedeco.openpose.global.openpose.RenderMode;
import static org.bytedeco.openpose.global.openpose.PoseModel;
import static org.bytedeco.openpose.global.openpose.HeatMapType;
import static org.bytedeco.openpose.global.openpose.POSE_DEFAULT_ALPHA_KEYPOINT;
import static org.bytedeco.openpose.global.openpose.POSE_DEFAULT_ALPHA_HEAT_MAP;
import org.bytedeco.openpose.OpWrapper;
import org.bytedeco.openpose.OpString;
import org.bytedeco.openpose.WrapperStructFace;
import org.bytedeco.openpose.WrapperStructHand;
import org.bytedeco.openpose.WrapperStructPose;
import org.bytedeco.openpose.Matrix;


public class openpose {
    static WrapperStructPose mkWrapperStructPose() {
        String modelFolder = System.getenv("MODEL_FOLDER");
        if (modelFolder == null) {
            System.err.println("MODEL_FOLDER must be set");
            System.exit(-1);
        }
        WrapperStructPose structPose = new WrapperStructPose();
        structPose.modelFolder(new OpString(modelFolder));
        return structPose;
    }

    public static void main(String[] args) {
        // Parse command arguments
        boolean doHands = false;
        boolean doFace = false;
        String[] pargs = new String[2];
        int pargsIdx = 0;
        for (String arg : args) {
            if (arg.startsWith("-")) {
                if (arg.equals("--hands")) {
                    doHands = true;
                } if (arg.equals("--face")) {
                    doFace = true;
                } else {
                    System.err.println(
                        "Usage: <openpose sample> [--hands] [--face] IMAGE_IN IMAGE_OUT"
                    );
                    System.exit(-1);
                }
            } else {
                pargs[pargsIdx] = arg;
                pargsIdx += 1;
            }
        }
        // Configure OpenPose
        OpWrapper opWrapper = new OpWrapper(ThreadManagerMode.Asynchronous);
        opWrapper.disableMultiThreading();
        opWrapper.configure(mkWrapperStructPose());
        if (doFace) {
            WrapperStructFace face = new WrapperStructFace();
            face.enable(true);
            opWrapper.configure(face);
        }
        if (doHands) {
            WrapperStructHand hand = new WrapperStructHand();
            hand.enable(true);
            opWrapper.configure(hand);
        }
        // Start OpenPose
        opWrapper.start();
        Mat ocvIm = imread(pargs[0]);
        Matrix opIm = OP_CV2OPCONSTMAT(ocvIm);
        Datum dat = new Datum();
        dat.cvInputData(opIm);
        Datums dats = new Datums();
        dats.put(dat);
        opWrapper.emplaceAndPop(dats);
        dat = dats.get(0);
        // Print keypoints
        FloatArray poseArray = dat.poseKeypoints();
        IntPointer dimSizes = poseArray.getSize();
        int numPeople = dimSizes.get(0);
        int numJoints = dimSizes.get(1);
        for (int i = 0; i < numPeople; i++) {
            System.out.printf("Person %d\n", i);
            for (int j = 0; j < numJoints; j++) {
                System.out.printf(
                    "Limb %d\tx: %f\ty: %f\t: %f\n",
                    j,
                    poseArray.get(new int[] {i, j, 0})[0],
                    poseArray.get(new int[] {i, j, 1})[0],
                    poseArray.get(new int[] {i, j, 2})[0]
                );
            }
        }
        // Save output
        Mat ocvMatOut = OP_OP2CVCONSTMAT(dat.cvOutputData());
        imwrite(pargs[1], ocvMatOut);
    }
}
