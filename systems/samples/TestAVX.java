import org.bytedeco.javacpp.*;
import org.bytedeco.systems.global.*;

public class TestAVX {
    public static void main(String[] args) {
        int AVXbit = 0x10000000;
        int AVX2bit = 0x20;
        boolean hasAVX = false;
        boolean hasAVX2 = false;
        String platform = Loader.getPlatform();

        if (platform.startsWith("linux-x86")) {
            int[] eax = {0}, ebx = {0}, ecx = {0}, edx = {0};
            linux.__cpuid_count(1, 0, eax, ebx, ecx, edx);
            hasAVX = (ecx[0] & AVXbit) != 0;
            linux.__cpuid_count(7, 0, eax, ebx, ecx, edx);
            hasAVX2 = hasAVX && (ebx[0] & AVX2bit) != 0;
        } else if (platform.startsWith("macosx-x86")) {
            int[] eax = {0}, ebx = {0}, ecx = {0}, edx = {0};
            macosx.__cpuid_count(1, 0, eax, ebx, ecx, edx);
            hasAVX = (ecx[0] & AVXbit) != 0;
            macosx.__cpuid_count(7, 0, eax, ebx, ecx, edx);
            hasAVX2 = hasAVX && (ebx[0] & AVX2bit) != 0;
        } else if (platform.startsWith("windows-x86")) {
            int[] cpuinfo = new int[4];
            windows.__cpuidex(cpuinfo, 1, 0);
            hasAVX = (cpuinfo[2] & AVXbit) != 0;
            windows.__cpuidex(cpuinfo, 7, 0);
            hasAVX2 = hasAVX && (cpuinfo[1] & AVX2bit) != 0;
        }

        System.out.println("hasAVX = " + hasAVX);
        System.out.println("hasAVX2 = " + hasAVX2);
    }
}
