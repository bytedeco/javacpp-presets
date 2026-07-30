package samples.demo.kvcache;

import org.bytedeco.pytorch.llm.kvcache.CoWBlockManager;

import java.util.*;
import java.util.concurrent.*;

public class RadixTreeStressTest {
    public static void main(String[] args) throws InterruptedException {
        int totalBlocks = 3000;
        CoWBlockManager manager = new CoWBlockManager(totalBlocks, 32, 16, 128, 0);

        // 模拟 3 轮对话，每轮增加一些内容
        List<Long> round1 = Arrays.asList(1001L, 1002L, 1003L); // System + Q1
        List<Long> round2 = Arrays.asList(1001L, 1002L, 1003L, 2001L, 2002L); // Round 1 + Q2

        ExecutorService executor = Executors.newFixedThreadPool(8);

        for (int i = 0; i < 500; i++) {
            final int id = i;
            executor.submit(() -> {
                String sid = "session-" + id;
                // 模拟不同用户，有些共享 round1，有些共享 round2
                List<Long> myPath = (id % 2 == 0) ? round1 : round2;

                List<Integer> blocks = manager.matchAndAllocatePath(myPath, sid, null);

                if (id % 100 == 0) {
                    System.out.printf("Session %d allocated %d blocks. Current Free: %d%n",
                            id, blocks.size(), manager.getFreeBlockCount());
                }

                // 模拟处理时间
                try { Thread.sleep(50); } catch (InterruptedException e) {}

                manager.releaseSession(sid);
            });
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);
    }
}