/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

/** @file frame_listener_impl.h Implementation of the frame listener classes. */

#ifndef FRAME_LISTENER_IMPL_H_
#define FRAME_LISTENER_IMPL_H_

#include <map>

#include <libfreenect2/config.h>
#include <libfreenect2/frame_listener.hpp>

namespace libfreenect2
{
///@addtogroup frame
///@{

/** Storage of multiple different types of frames. */
// typedef std::map<Frame::Type, Frame*> FrameMap;

class SyncMultiFrameListenerImpl;

/** Collect multiple types of frames. */
class LIBFREENECT2_API SyncMultiFrameListener : public FrameListener
{
public:
  /**
   * @param frame_types Use bitwise or to combine multiple types, e.g. `Frame::Ir | Frame::Depth`.
   */
  SyncMultiFrameListener(unsigned int frame_types);
  virtual ~SyncMultiFrameListener();

  /** Test if there are new frames. Non-blocking. */
  bool hasNewFrame() const;

  /** Wait milliseconds for new frames.
   * @param[out] frame Caller is responsible to release the frames.
   * @param milliseconds Timeout. This parameter is ignored if not built with C++11 threading support.
   * @return true if a frame is received; false if not.
   */
  bool waitForNewFrame(std::map<Frame::Type, Frame*> &frame, int milliseconds);

  /** Wait indefinitely for new frames.
   * @param[out] frame Caller is responsible to release the frames.
   */
  void waitForNewFrame(std::map<Frame::Type, Frame*> &frame);

  /** Shortcut to delete all frames */
  void release(std::map<Frame::Type, Frame*> &frame);

  virtual bool onNewFrame(Frame::Type type, Frame *frame);
private:
  SyncMultiFrameListenerImpl *impl_;
};

///@}
} /* namespace libfreenect2 */
#endif /* FRAME_LISTENER_IMPL_H_ */
