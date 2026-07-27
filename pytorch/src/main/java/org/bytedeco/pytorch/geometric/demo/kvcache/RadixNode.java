//package org.bytedeco.pytorch.geometric.demo.kvcache;
//
//import java.util.Map;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.concurrent.atomic.AtomicInteger;
//
//class RadixNode {
//    long blockHash;
//    int physicalBlockId;
//    Map<Long, RadixNode> children = new ConcurrentHashMap<>();
//    AtomicInteger refCount = new AtomicInteger(0);
//}