import org.bytedeco.shakapackager.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;



public class Example {
  

    public static void main(String[] args) throws Exception {
        System.out.println("Testing Shaka Packager");
        if (args.length < 1) {
            System.out.println("Missing input video file");
            System.exit(-1);
        }
        int ret = -1,  v_stream_idx = -1;


        String vf_path = args[0];
        List<StreamDescriptor> streams = new ArrayList<>();

        TestParams ts = new TestParams();
        ts.dump_stream_info(true);

        StreamDescriptor st = new StreamDescriptor();
        st.input("test.mp4");
        st.stream_selector("video");
        st.output("output_video.mp4");
        streams.add(st);

        StreamDescriptor ste = new StreamDescriptor();
        ste.input("test.mp4");
        ste.stream_selector("audio");
        ste.output("output_audio.mp4");
        streams.add(ste);

        final StreamDescriptor rectsPointer = new StreamDescriptor(streams.size());
        for (int i=0; i<streams.size(); ++i)
            rectsPointer.position(i).put(streams.get(i));


        Packager packager = new Packager();
        System.out.println("Shaka Packager version : " + Packager.GetLibraryVersion());
        PackagingParams packaging_params = new PackagingParams();
        packaging_params.test_params(ts);
        packaging_params.chunking_params(new ChunkingParams().segment_duration_in_seconds(5.0));
        System.out.println("Testing Shaka Packager");
        Status s = packager.Initialize(packaging_params,rectsPointer);
        s = packager.Run();
        System.out.println("test status : " + s.ToString());
    }
}
