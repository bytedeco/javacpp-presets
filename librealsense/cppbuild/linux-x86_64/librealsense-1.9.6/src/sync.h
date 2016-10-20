// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#ifndef LIBREALSENSE_SYNC_H
#define LIBREALSENSE_SYNC_H

#include "archive.h"
#include <atomic>
#include "timestamps.h"

namespace rsimpl
{
    class syncronizing_archive : public frame_archive
    {
    private:
        // This data will be left constant after creation, and accessed from all threads
        subdevice_mode_selection modes[RS_STREAM_NATIVE_COUNT];
        rs_stream key_stream;
        std::vector<rs_stream> other_streams;

        // This data will be read and written exclusively from the application thread
        frameset frontbuffer;

        // This data will be read and written by all threads, and synchronized with a mutex
        std::vector<frame> frames[RS_STREAM_NATIVE_COUNT];
        std::condition_variable_any cv;
        
        void get_next_frames();
        void dequeue_frame(rs_stream stream);
        void discard_frame(rs_stream stream);
        void cull_frames();

        timestamp_corrector            ts_corrector;
    public:
        syncronizing_archive(const std::vector<subdevice_mode_selection> & selection, 
            rs_stream key_stream, 
            std::atomic<uint32_t>* max_size,
            std::atomic<uint32_t>* event_queue_size,
            std::atomic<uint32_t>* events_timeout,
            std::chrono::high_resolution_clock::time_point capture_started = std::chrono::high_resolution_clock::now());
        
        // Application thread API
        void wait_for_frames();
        bool poll_for_frames();

        frameset * wait_for_frames_safe();
        bool poll_for_frames_safe(frameset ** frames);

        const byte * get_frame_data(rs_stream stream) const;
        double get_frame_timestamp(rs_stream stream) const;
        unsigned long long get_frame_number(rs_stream stream) const;
        long long get_frame_system_time(rs_stream stream) const;
        int get_frame_stride(rs_stream stream) const;
        int get_frame_bpp(rs_stream stream) const;

        frameset * clone_frontbuffer();

        // Frame callback thread API
        void commit_frame(rs_stream stream);

        void flush() override;

        void correct_timestamp(rs_stream stream);
        void on_timestamp(rs_timestamp_data data);

    };
}

#endif
