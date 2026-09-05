/**
******************************************************************************
* Xenia : Xbox 360 Emulator Research Project                                 *
******************************************************************************
* Copyright 2024 Ben Vanik. All rights reserved.                             *
* Released under the BSD license - see LICENSE in the root for more details. *
******************************************************************************
*/

#include "xenia/apu/xma_context_master.h"

#include <cstring>
#include <tuple>

#include "xenia/apu/xma_decoder.h"
#include "xenia/apu/xma_helpers.h"
#include "xenia/base/bit_stream.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"
#include "xenia/base/ring_buffer.h"

extern "C" {
#if XE_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable : 4101 4244 5033)
#endif
#include "third_party/FFmpeg/libavcodec/avcodec.h"
#include "third_party/FFmpeg/libavutil/channel_layout.h"
#include "third_party/FFmpeg/libavutil/error.h"
#if XE_COMPILER_MSVC
#pragma warning(pop)
#endif
}  // extern "C"

// Credits for most of this code goes to:
// https://github.com/koolkdev/libertyv/blob/master/libav_wrapper/xma2dec.c

namespace xe {
namespace apu {

XmaContextMaster::XmaContextMaster() = default;

XmaContextMaster::~XmaContextMaster() {
  if (av_context_) {
    avcodec_free_context(&av_context_);
  }
  if (av_frame_) {
    av_frame_free(&av_frame_);
  }
  // if (current_frame_) {
  //   delete[] current_frame_;
  //  }
}

int XmaContextMaster::Setup(uint32_t id, Memory* memory, uint32_t guest_ptr) {
  id_ = id;
  memory_ = memory;
  guest_ptr_ = guest_ptr;

  // Allocate ffmpeg stuff:
  av_packet_ = av_packet_alloc();
  assert_not_null(av_packet_);
  // chrispy: preallocate this buffer so that ffmpeg isn't reallocating it for
  // every packet, these allocations were causing RtlSubsegmentInitialize
  av_packet_->buf = av_buffer_alloc(128 * 1024);
  // find the XMA2 audio decoder
  av_codec_ = avcodec_find_decoder(AV_CODEC_ID_XMAFRAMES);
  if (!av_codec_) {
    XELOGE("XmaContext {}: Codec not found", id);
    return 1;
  }

  av_context_ = avcodec_alloc_context3(av_codec_);
  if (!av_context_) {
    XELOGE("XmaContext {}: Couldn't allocate context", id);
    return 1;
  }

  // Initialize these to 0. They'll actually be set later.
  av_context_->ch_layout = AVChannelLayout{};
  av_context_->sample_rate = 0;

  av_frame_ = av_frame_alloc();
  if (!av_frame_) {
    XELOGE("XmaContext {}: Couldn't allocate frame", id);
    return 1;
  }

  // FYI: We're purposely not opening the codec here. That is done later.
  return 0;
}

bool XmaContextMaster::Work() {
  if (!is_enabled() || !is_allocated()) {
    return false;
  }

  std::lock_guard<xe_mutex> lock(lock_);
  set_is_enabled(false);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  const XMA_CONTEXT_DATA initial_data = data;

  if (!data.output_buffer_valid) {
    return true;
  }

  RingBuffer output_rb = PrepareOutputRingBuffer(&data);

  if (data.IsConsumeOnlyContext()) {
    if (current_frame_remaining_subframes_ == 0) {
      return true;
    }
    XELOGAPU("XmaContext {}: Consume-only context, draining subframes", this->id());
    Consume(&output_rb, &data);
    data.output_buffer_write_offset =
        output_rb.write_offset() / XmaContext::kOutputBytesPerBlock;
    data.Store(context_ptr);
    return true;
  }

  // Minimum free blocks needed before attempting a decode.
  const uint32_t effective_sdc =
      std::max(static_cast<uint32_t>(1), data.subframe_decode_count);
  const int32_t minimum_subframe_decode_count =
      static_cast<int32_t>(effective_sdc) + data.output_buffer_padding;

  if (minimum_subframe_decode_count >
      this->remaining_subframe_blocks_in_output_buffer_) {
    XELOGD("XmaContext {}: No space for subframe decoding {}/{}!", this->id(),
           minimum_subframe_decode_count,
           this->remaining_subframe_blocks_in_output_buffer_);
    data.Store(context_ptr);
    return true;
  }

  while (this->remaining_subframe_blocks_in_output_buffer_ >=
         minimum_subframe_decode_count) {
    XELOGAPU(
        "XmaContext {}: Write Count: {}, Capacity: {} - {} {} Subframes: {} "
        "Padding: {}",
        this->id(), (uint32_t)output_rb.write_count(),
        this->remaining_subframe_blocks_in_output_buffer_,
        data.input_buffer_0_valid + (data.input_buffer_1_valid << 1),
        data.output_buffer_valid, data.subframe_decode_count,
        data.output_buffer_padding);

    const uint32_t pre_decode_offset = data.input_buffer_read_offset;
    const uint8_t pre_remaining_subframes = current_frame_remaining_subframes_;

    Decode(&data);
    Consume(&output_rb, &data);

    if (!data.IsAnyInputBufferValid() || data.error_status == 4) {
      XELOGAPU(
          "XmaContext {}: Work loop exit - buffers_valid={} error_status={}",
          this->id(), data.IsAnyInputBufferValid(), data.error_status);
      break;
    }

    if (pre_remaining_subframes == 0 &&
        data.input_buffer_read_offset == pre_decode_offset &&
        current_frame_remaining_subframes_ == 0) {
      XELOGAPU(
          "XmaContext {}: Decode stalled at offset {} (no progress), "
          "waiting for next buffer",
          this->id(), pre_decode_offset);
      break;
    }
  }

  data.output_buffer_write_offset =
      output_rb.write_offset() / XmaContext::kOutputBytesPerBlock;

  XELOGAPU("XmaContext {}: Read Output: {} Write Output: {}", this->id(),
           data.output_buffer_read_offset, data.output_buffer_write_offset);

  if (output_rb.empty()) {
    XELOGAPU("XmaContext {}: Output ring buffer empty, invalidating output",
             this->id());
    data.output_buffer_valid = 0;
  }

  data.Store(context_ptr);
  return true;
}

void XmaContextMaster::Enable() {
  std::lock_guard<xe_mutex> lock(lock_);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);

  XELOGAPU("XmaContext: kicking context {} (buffer {} {}/{} bits)", this->id(),
           data.current_buffer, data.input_buffer_read_offset,
           (data.current_buffer == 0 ? data.input_buffer_0_packet_count
                                     : data.input_buffer_1_packet_count) *
               kBitsPerPacket);

  data.Store(context_ptr);

  set_is_enabled(true);
}

void XmaContextMaster::Clear() {
  std::lock_guard<xe_mutex> lock(lock_);
  XELOGAPU("XmaContext: reset context {}", this->id());

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);

  data.input_buffer_0_valid = 0;
  data.input_buffer_1_valid = 0;
  data.output_buffer_valid = 0;

  data.output_buffer_read_offset = 0;
  data.output_buffer_write_offset = 0;

  data.Store(context_ptr);
}

void XmaContextMaster::Disable() {
  std::lock_guard<xe_mutex> lock(lock_);
  XELOGAPU("XmaContext: disabling context {}", this->id());
  set_is_enabled(false);
}

void XmaContextMaster::Release() {
  // Lock it in case the decoder thread is working on it now.
  std::lock_guard<xe_mutex> lock(lock_);
  assert_true(is_allocated_ == true);

  set_is_allocated(false);
  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));  // Zero it.
}

void XmaContextMaster::SwapInputBuffer(XMA_CONTEXT_DATA* data) {
  // No more frames.
  if (data->current_buffer == 0) {
    data->input_buffer_0_valid = 0;
  } else {
    data->input_buffer_1_valid = 0;
  }
  data->current_buffer ^= 1;
  data->input_buffer_read_offset = 0;
}

bool XmaContextMaster::TrySetupNextLoop(XMA_CONTEXT_DATA* data,
                                        bool ignore_input_buffer_offset) {
  // Setup the input buffer offset if next loop exists.
  // TODO(Pseudo-Kernel): Need to handle loop in the following cases.
  // 1. loop_start == loop_end == 0
  // 2. loop_start > loop_end && loop_count > 0
  if (data->loop_count > 0 && data->loop_start < data->loop_end &&
      (ignore_input_buffer_offset ||
       data->input_buffer_read_offset >= data->loop_end)) {
    // Loop back to the beginning.
    data->input_buffer_read_offset = data->loop_start;
    if (data->loop_count < 255) {
      data->loop_count--;
    }
    return true;
  }
  return false;
}

/*
void XmaContext::NextPacket(
    uint8_t* input_buffer,
    uint32_t input_size,
    uint32_t input_buffer_read_offset) {
*/
void XmaContextMaster::NextPacket(XMA_CONTEXT_DATA* data) {
  // auto packet_idx = GetFramePacketNumber(input_buffer, input_size,
  // input_buffer_read_offset);

  // packet_idx++;
  // if (packet_idx++ >= input_size)
}

int XmaContextMaster::GetSampleRate(int id) {
  switch (id) {
    case 0:
      return 24000;
    case 1:
      return 32000;
    case 2:
      return 44100;
    case 3:
      return 48000;
  }
  assert_always();
  return 0;
}

bool XmaContextMaster::ValidFrameOffset(uint8_t* block, size_t size_bytes,
                                        size_t frame_offset_bits) {
  uint32_t packet_num =
      GetFramePacketNumber(block, size_bytes, frame_offset_bits);
  if (packet_num == -1) {
    // Invalid packet number
    return false;
  }

  uint8_t* packet = block + (packet_num * XmaContext::kBytesPerPacket);
  size_t relative_offset_bits = frame_offset_bits % XmaContext::kBitsPerPacket;

  uint32_t first_frame_offset = xma::GetPacketFrameOffset(packet);
  if (first_frame_offset == -1 || first_frame_offset > kBitsPerPacket) {
    // Packet only contains a partial frame, so no frames can start here.
    return false;
  }

  BitStream stream(packet, kBitsPerPacket);
  stream.SetOffset(first_frame_offset);
  while (true) {
    if (stream.offset_bits() == relative_offset_bits) {
      return true;
    }

    if (stream.BitsRemaining() < 15) {
      // Not enough room for another frame header.
      return false;
    }

    uint64_t size = stream.Read(15);
    if ((size - 15) > stream.BitsRemaining()) {
      // Last frame.
      return false;
    } else if (size == 0x7FFF) {
      // Invalid frame (and last of this packet)
      return false;
    }

    stream.Advance(size - 16);

    // Read the trailing bit to see if frames follow
    if (stream.Read(1) == 0) {
      break;
    }
  }

  return false;
}

void XmaContextMaster::Decode(XMA_CONTEXT_DATA* data) {
  SCOPE_profile_cpu_f("apu");

  // No available data.
  if (!data->IsAnyInputBufferValid()) {
    XELOGAPU("XmaContext {}: Decode skipped - no valid input buffers", this->id());
    return;
  }

  if (current_frame_remaining_subframes_ > 0) {
    return;
  }

  if (!data->IsCurrentInputBufferValid()) {
    XELOGAPU("XmaContext {}: Current buffer {} invalid, swapping to other",
             this->id(), data->current_buffer);
    SwapInputBuffer(data);
    if (!data->IsCurrentInputBufferValid()) {
      XELOGAPU("XmaContext {}: Both buffers invalid after swap, aborting",
               this->id());
      return;
    }
  }

  // Translate pointers for future use.
  uint8_t* in0 = data->input_buffer_0_valid
                     ? memory()->TranslatePhysical(data->input_buffer_0_ptr)
                     : nullptr;
  uint8_t* in1 = data->input_buffer_1_valid
                     ? memory()->TranslatePhysical(data->input_buffer_1_ptr)
                     : nullptr;
  uint8_t* current_input_buffer = data->current_buffer ? in1 : in0;

  XELOGAPU("Processing context {} (offset {}, buffer {}, ptr {:p})", this->id(),
           data->input_buffer_read_offset, data->current_buffer,
           static_cast<void*>(current_input_buffer));

  size_t input_buffer_0_size =
      data->input_buffer_0_packet_count * XmaContext::kBytesPerPacket;
  size_t input_buffer_1_size =
      data->input_buffer_1_packet_count * XmaContext::kBytesPerPacket;

  size_t current_input_size =
      data->current_buffer ? input_buffer_1_size : input_buffer_0_size;
  size_t current_input_packet_count = current_input_size / XmaContext::kBytesPerPacket;

  assert_false(data->stop_when_done);
  assert_false(data->interrupt_when_done);
  static int total_samples = 0;
  bool reuse_input_buffer = false;
  
  // Update loop status before processing
  UpdateLoopStatus(data);
  
  // Process one frame at a time (subframe processing handled in Work/Consume)
  {
    if (!data->input_buffer_0_valid && !data->input_buffer_1_valid) {
      // Out of data.
      return;
    }

    // Setup the input buffer if we are at loop_end.
    // The input buffer must not be swapped out until all loops are processed.
    reuse_input_buffer = TrySetupNextLoop(data, false);

    // assert_true(packets_skip_ == 0);
    // assert_true(split_frame_len_ == 0);
    // assert_true(split_frame_len_partial_ == 0);

    // Where are we in the buffer (in XMA jargon)
    int packet_idx, frame_idx, frame_count;
    uint8_t* packet;
    bool frame_last_split;

    BitStream stream(current_input_buffer, current_input_size * 8);
    stream.SetOffset(data->input_buffer_read_offset);

    // if we had a buffer swap try to skip packets first
    if (packets_skip_ > 0) {
      packet_idx =
          GetFramePacketNumber(current_input_buffer, current_input_size,
                               data->input_buffer_read_offset);
      while (packets_skip_ > 0) {
        packets_skip_--;
        packet_idx++;
        if (packet_idx >= current_input_packet_count) {
          if (!reuse_input_buffer) {
            // Last packet. Try setup once more.
            reuse_input_buffer = TrySetupNextLoop(data, true);
          }
          if (!reuse_input_buffer) {
            SwapInputBuffer(data);
          }
          return;
        }
      }
      // invalid frame pointer but needed for us
      data->input_buffer_read_offset = packet_idx * kBitsPerPacket;
      // continue;
    }

    if (split_frame_len_) {
      // handle a frame that was split over two packages
      packet_idx =
          GetFramePacketNumber(current_input_buffer, current_input_size,
                               data->input_buffer_read_offset);
      packet = current_input_buffer + packet_idx * XmaContext::kBytesPerPacket;
      std::tie(frame_count, frame_last_split) = GetPacketFrameCount(packet);
      frame_idx = -1;

      stream =
          BitStream(current_input_buffer, (packet_idx + 1) * kBitsPerPacket);
      stream.SetOffset(packet_idx * kBitsPerPacket + 32);

      if (split_frame_len_ > xma::kMaxFrameLength) {
        // TODO write CopyPeekMethod
        auto offset = stream.offset_bits();
        stream.Copy(
            xma_frame_.data() + 1 +
                ((split_frame_len_partial_ + split_frame_padding_start_) / 8),
            15 - split_frame_len_partial_);
        stream.SetOffset(offset);
        BitStream slen(xma_frame_.data() + 1, 15 + split_frame_padding_start_);
        slen.Advance(split_frame_padding_start_);
        split_frame_len_ = static_cast<int>(slen.Read(15));
      }

      if (frame_count > 0) {
        assert_true(xma::GetPacketFrameOffset(packet) - 32 ==
                    split_frame_len_ - split_frame_len_partial_);
      }

      auto offset = stream.Copy(
          xma_frame_.data() + 1 +
              ((split_frame_len_partial_ + split_frame_padding_start_) / 8),
          split_frame_len_ - split_frame_len_partial_);
      assert_true(offset ==
                  (split_frame_padding_start_ + split_frame_len_partial_) % 8);
    } else {
      if (data->input_buffer_read_offset % kBitsPerPacket == 0) {
        // Invalid offset. Go ahead and set it.
        int packet_number =
            GetFramePacketNumber(current_input_buffer, current_input_size,
                                 data->input_buffer_read_offset);

        if (packet_number == -1) {
          return;
        }

        auto offset =
            xma::GetPacketFrameOffset(current_input_buffer +
                                      XmaContext::kBytesPerPacket * packet_number) +
            data->input_buffer_read_offset;
        if (offset == -1) {
          // No more frames.
          SwapInputBuffer(data);
          // TODO partial frames? end?
          XELOGE("XmaContext {}: TODO partial frames? end?", this->id());
          assert_always("TODO");
          return;
        } else {
          data->input_buffer_read_offset = offset;
        }
      }

      if (!ValidFrameOffset(current_input_buffer, current_input_size,
                            data->input_buffer_read_offset)) {
        XELOGAPU("XmaContext {}: Invalid read offset {}!", this->id(),
                 data->input_buffer_read_offset);
        SwapInputBuffer(data);
        return;
      }

      // Where are we in the buffer (in XMA jargon)
      std::tie(packet_idx, frame_idx) =
          GetFrameNumber(current_input_buffer, current_input_size,
                         data->input_buffer_read_offset);
      // TODO handle
      assert_true(packet_idx >= 0);
      assert_true(frame_idx >= 0);
      packet = current_input_buffer + packet_idx * XmaContext::kBytesPerPacket;
      // frames that belong to this packet
      std::tie(frame_count, frame_last_split) = GetPacketFrameCount(packet);
      assert_true(frame_count >= 0);  // TODO end

      PrepareDecoder(packet, data->sample_rate, bool(data->is_stereo));

      // Current frame is split to next packet:
      bool frame_is_split = frame_last_split && (frame_idx >= frame_count - 1);

      stream =
          BitStream(current_input_buffer, (packet_idx + 1) * kBitsPerPacket);
      stream.SetOffset(data->input_buffer_read_offset);
      // int frame_len;
      // int frame_len_partial
      split_frame_len_partial_ = static_cast<int>(stream.BitsRemaining());
      if (split_frame_len_partial_ >= 15) {
        split_frame_len_ = static_cast<int>(stream.Peek(15));
      } else {
        // assert_always();
        split_frame_len_ = xma::kMaxFrameLength + 1;
      }
      assert_true(frame_is_split ==
                  (split_frame_len_ > split_frame_len_partial_));

      // TODO fix bitstream copy
      std::memset(xma_frame_.data(), 0, xma_frame_.size());

      {
        auto offset =
            stream.Copy(xma_frame_.data() + 1,
                        std::min(split_frame_len_, split_frame_len_partial_));
        assert_true(offset < 8);
        split_frame_padding_start_ = static_cast<uint8_t>(offset);
      }

      if (frame_is_split) {
        // go to next xma packet of this stream
        packets_skip_ = xma::GetPacketSkipCount(packet) + 1;
        while (packets_skip_ > 0) {
          packets_skip_--;
          packet += XmaContext::kBytesPerPacket;
          packet_idx++;
          if (packet_idx >= current_input_packet_count) {
            if (!reuse_input_buffer) {
              // Last packet. Try setup once more.
              reuse_input_buffer = TrySetupNextLoop(data, true);
            }
            if (!reuse_input_buffer) {
              SwapInputBuffer(data);
            }
            return;
          }
        }
        // TODO guest might read this:
        data->input_buffer_read_offset = packet_idx * kBitsPerPacket;
        // Return after processing split frame
        return;
      }
    }

    av_packet_->data = xma_frame_.data();
    av_packet_->size = static_cast<int>(
        1 + ((split_frame_padding_start_ + split_frame_len_) / 8) +
        (((split_frame_padding_start_ + split_frame_len_) % 8) ? 1 : 0));

    auto padding_end = av_packet_->size * 8 -
                       (8 + split_frame_padding_start_ + split_frame_len_);
    assert_true(padding_end < 8);
    xma_frame_[0] =
        ((split_frame_padding_start_ & 7) << 5) | ((padding_end & 7) << 2);

    split_frame_len_ = 0;
    split_frame_len_partial_ = 0;
    split_frame_padding_start_ = 0;

    auto ret = avcodec_send_packet(av_context_, av_packet_);
    if (ret < 0) {
      XELOGE("XmaContext {}: Error sending packet for decoding", id());
      // TODO bail out
      assert_always();
    }
    ret = avcodec_receive_frame(av_context_, av_frame_);
    if (ret == AVERROR(EAGAIN)) {
      return;
    }
    if (ret < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(ret, errbuf, sizeof(errbuf));
      XELOGE("XmaContext {}: Error during decoding: {} ({})", id(), errbuf,
             ret);
      return;
    }
    assert_true(ret == 0);

    {
      // copy over 1 frame
      // update input buffer read offset

      assert_true(av_context_->sample_fmt == AV_SAMPLE_FMT_FLTP);

      ConvertFrame((const uint8_t**)av_frame_->data, bool(data->is_stereo),
                   raw_frame_.data());

      // Set up subframe processing instead of writing entire frame at once
      current_frame_remaining_subframes_ = 4 << data->is_stereo;

      // Check if this is the loop end frame
      bool is_loop_end_frame = false;
      if (data->loop_count > 0) {
        const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);
        is_loop_end_frame = (data->input_buffer_read_offset == loop_end);
      }

      // Loop end: limit output to subframes 0..loop_subframe_end.
      if (is_loop_end_frame) {
        loop_frame_output_limit_ = (data->loop_subframe_end + 1) << data->is_stereo;
        XELOGAPU(
            "XmaContext {}: Loop end frame - limiting output to {} subframes "
            "(loop_subframe_end={})",
            id(), loop_frame_output_limit_, data->loop_subframe_end);
      } else {
        loop_frame_output_limit_ = 0;
      }

      // Loop start: skip leading subframes per loop_subframe_skip.
      if (loop_start_skip_pending_) {
        const uint8_t skip = data->loop_subframe_skip << data->is_stereo;
        if (skip < current_frame_remaining_subframes_) {
          XELOGAPU(
              "XmaContext {}: Loop start - skipping {} leading subframes "
              "(loop_subframe_skip={})",
              id(), skip, data->loop_subframe_skip);
          current_frame_remaining_subframes_ -= skip;
        }
        loop_start_skip_pending_ = false;
      }

      total_samples += id_ == 0 ? kSamplesPerFrame : 0;

      // Update input buffer read offset for next frame
      uint32_t offset = data->input_buffer_read_offset;
      offset = static_cast<uint32_t>(
          GetNextFrame(current_input_buffer, current_input_size, offset));
      
      if (frame_idx + 1 >= frame_count) {
        // Skip to next packet (no split frame)
        packets_skip_ = xma::GetPacketSkipCount(packet) + 1;
        while (packets_skip_ > 0) {
          packets_skip_--;
          packet_idx++;
          if (packet_idx >= current_input_packet_count) {
            if (!reuse_input_buffer) {
              // Last packet. Try setup once more.
              reuse_input_buffer = TrySetupNextLoop(data, true);
            }
            if (!reuse_input_buffer) {
              SwapInputBuffer(data);
            }
            // Return after setting up subframe state
            return;
          }
        }
        packet = current_input_buffer + packet_idx * XmaContext::kBytesPerPacket;
        offset =
            xma::GetPacketFrameOffset(packet) + packet_idx * XmaContext::kBitsPerPacket;
      }
      
      if (offset == 0 || frame_idx == -1) {
        // Next packet but we already skipped to it
        if (packet_idx >= current_input_packet_count) {
          // Buffer is fully used
          if (!reuse_input_buffer) {
            // Last packet. Try setup once more.
            reuse_input_buffer = TrySetupNextLoop(data, true);
          }
          if (!reuse_input_buffer) {
            SwapInputBuffer(data);
          }
          // Buffer fully used, return
          return;
        }
        offset =
            xma::GetPacketFrameOffset(packet) + packet_idx * kBitsPerPacket;
      }
      
      // TODO buffer bounds check
      assert_true(data->input_buffer_read_offset < offset);
      data->input_buffer_read_offset = offset;
      
      // Return after processing one frame to allow subframe consumption
      return;
    }
  }

  // assert_true((split_frame_len_ != 0) == (data->input_buffer_read_offset ==
  // 0));

  // The game will kick us again with a new output buffer later.
  // It's important that we only invalidate this if we actually wrote to it!!
  // Note: output buffer validation is handled in Work function
  // if (output_rb.write_offset() == output_rb.read_offset()) {
  //   data->output_buffer_valid = 0;
  // }
}

size_t XmaContextMaster::GetNextFrame(uint8_t* block, size_t size,
                                      size_t bit_offset) {
  // offset = xma::GetPacketFrameOffset(packet);
  // TODO meh
  // auto next_packet = bit_offset - bit_offset % kBitsPerPacket +
  // kBitsPerPacket;
  auto packet_idx = GetFramePacketNumber(block, size, bit_offset);

  BitStream stream(block, size * 8);
  stream.SetOffset(bit_offset);

  if (stream.BitsRemaining() < 15) {
    return 0;
  }

  uint64_t len = stream.Read(15);
  if ((len - 15) > stream.BitsRemaining()) {
    // assert_always("TODO");
    //  *bit_offset = next_packet;
    //  return false;
    //  return next_packet;
    return 0;
  } else if (len >= xma::kMaxFrameLength) {
    // assert_always("TODO");
    //  *bit_offset = next_packet;
    //  return false;
    return 0;
    // return next_packet;
  }

  stream.Advance(len - (15 + 1));
  // Read the trailing bit to see if frames follow
  if (stream.Read(1) == 0) {
    return 0;
  }

  bit_offset += len;
  if (packet_idx < GetFramePacketNumber(block, size, bit_offset)) {
    return 0;
  }
  return bit_offset;
}

int XmaContextMaster::GetFramePacketNumber(uint8_t* block, size_t size,
                                           size_t bit_offset) {
  size *= 8;
  if (bit_offset >= size) {
    // Not good :(
    assert_always();
    return -1;
  }

  size_t byte_offset = bit_offset >> 3;
  size_t packet_number = byte_offset / XmaContext::kBytesPerPacket;

  return (uint32_t)packet_number;
}

std::tuple<int, int> XmaContextMaster::GetFrameNumber(uint8_t* block,
                                                      size_t size,
                                                      size_t bit_offset) {
  auto packet_idx = GetFramePacketNumber(block, size, bit_offset);

  if (packet_idx < 0 || (packet_idx + 1) * XmaContext::kBytesPerPacket > size) {
    assert_always();
    return {packet_idx, -2};
  }

  if (bit_offset == 0) {
    return {packet_idx, -1};
  }

  uint8_t* packet = block + (packet_idx * XmaContext::kBytesPerPacket);
  auto first_frame_offset = xma::GetPacketFrameOffset(packet);
  BitStream stream(block, size * 8);
  stream.SetOffset(packet_idx * XmaContext::kBitsPerPacket + first_frame_offset);

  int frame_idx = 0;
  while (true) {
    if (stream.BitsRemaining() < 15) {
      break;
    }

    if (stream.offset_bits() == bit_offset) {
      break;
    }

    uint64_t size = stream.Read(15);
    if ((size - 15) > stream.BitsRemaining()) {
      // Last frame.
      break;
    } else if (size == 0x7FFF) {
      // Invalid frame (and last of this packet)
      break;
    }

    stream.Advance(size - (15 + 1));

    // Read the trailing bit to see if frames follow
    if (stream.Read(1) == 0) {
      break;
    }
    frame_idx++;
  }
  return {packet_idx, frame_idx};
}

std::tuple<int, bool> XmaContextMaster::GetPacketFrameCount(uint8_t* packet) {
  auto first_frame_offset = xma::GetPacketFrameOffset(packet);
  if (first_frame_offset > kBitsPerPacket - 33) {
    // frame offset is beyond packet end
    return {0, false};
  }

  BitStream stream(packet, kBitsPerPacket);
  stream.SetOffset(first_frame_offset);
  int frame_count = 0;

  while (true) {
    frame_count++;
    if (stream.BitsRemaining() < 15) {
      return {frame_count, true};
    }

    uint64_t size = stream.Read(15);
    if ((size - 15) > stream.BitsRemaining()) {
      return {frame_count, true};
    } else if (size == 0x7FFF) {
      assert_always();
      return {frame_count, true};
    }

    stream.Advance(size - (15 + 1));

    if (stream.Read(1) == 0) {
      return {frame_count, false};
    }
  }
}

int XmaContextMaster::PrepareDecoder(uint8_t* packet, int sample_rate,
                                     bool is_two_channel) {
  // Sanity check: Packet metadata is always 1 for XMA2/0 for XMA
  assert_true((packet[2] & 0x7) == 1 || (packet[2] & 0x7) == 0);

  sample_rate = GetSampleRate(sample_rate);

  // Re-initialize the context with new sample rate and channels.
  uint32_t channels = is_two_channel ? 2 : 1;
  if (av_context_->sample_rate != sample_rate ||
      av_context_->ch_layout.nb_channels != (int)channels) {
    // We have to recreate the codec context so it'll realloc whatever data it
    // needs.
    avcodec_free_context(&av_context_);
    av_context_ = avcodec_alloc_context3(av_codec_);

    av_context_->sample_rate = sample_rate;
    av_channel_layout_default(&av_context_->ch_layout, channels);
    av_context_->flags2 |= AV_CODEC_FLAG2_SKIP_MANUAL;

    if (avcodec_open2(av_context_, av_codec_, NULL) < 0) {
      XELOGE("XmaContext: Failed to reopen FFmpeg context");
      return -1;
    }
    return 1;
  }
  return 0;
}

RingBuffer XmaContextMaster::PrepareOutputRingBuffer(XMA_CONTEXT_DATA* data) {
  // Use fully qualified constant names to avoid ambiguity
  const uint32_t output_capacity =
      data->output_buffer_block_count * XmaContext::kOutputBytesPerBlock;
  const uint32_t output_read_offset =
      data->output_buffer_read_offset * XmaContext::kOutputBytesPerBlock;
  const uint32_t output_write_offset =
      data->output_buffer_write_offset * XmaContext::kOutputBytesPerBlock;

  if (output_capacity > XmaContext::kOutputMaxSizeBytes) {
    XELOGW(
        "XmaContext {}: Output buffer uses more space than expected! "
        "(Actual: {} Max: {})",
        this->id(), output_capacity, XmaContext::kOutputMaxSizeBytes);
  }

  uint8_t* output_buffer = this->memory()->TranslatePhysical(data->output_buffer_ptr);

  RingBuffer output_rb(output_buffer, output_capacity);
  output_rb.set_read_offset(output_read_offset);
  output_rb.set_write_offset(output_write_offset);
  this->remaining_subframe_blocks_in_output_buffer_ =
      (int32_t)output_rb.write_count() / XmaContext::kOutputBytesPerBlock;

  return output_rb;
}

void XmaContextMaster::Consume(RingBuffer* XE_RESTRICT output_rb,
                               const XMA_CONTEXT_DATA* const XE_RESTRICT data) {
  if (this->current_frame_remaining_subframes_ == 0) {
    return;
  }

  // Check if the loop end truncation limit has been reached.
  if (this->loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes =
        (XmaContext::kBytesPerFrameChannel / XmaContext::kOutputBytesPerBlock) << data->is_stereo;
    const uint8_t consumed =
        total_subframes - this->current_frame_remaining_subframes_;
    if (consumed >= this->loop_frame_output_limit_) {
      XELOGAPU(
          "XmaContext {}: Loop end truncation: discarding {} remaining "
          "subframes (limit {})",
          this->id(), this->current_frame_remaining_subframes_, this->loop_frame_output_limit_);
      this->remaining_subframe_blocks_in_output_buffer_ -=
          data->output_buffer_padding;
      this->current_frame_remaining_subframes_ = 0;
      this->loop_frame_output_limit_ = 0;
      return;
    }
  }

  // Guard against subframe_decode_count == 0 which would cause zero progress
  // and an infinite loop in Work(). Treat 0 as 1 (minimum progress).
  const uint8_t effective_sdc =
      std::max(static_cast<uint32_t>(1), data->subframe_decode_count);
  int8_t subframes_to_write = std::min(
      (int8_t)this->current_frame_remaining_subframes_, (int8_t)effective_sdc);

  // Clamp to loop end limit if active.
  if (this->loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes =
        (XmaContext::kBytesPerFrameChannel / XmaContext::kOutputBytesPerBlock) << data->is_stereo;
    const uint8_t consumed =
        total_subframes - this->current_frame_remaining_subframes_;
    const int8_t remaining_until_limit =
        (int8_t)(this->loop_frame_output_limit_ - consumed);
    if (subframes_to_write > remaining_until_limit) {
      subframes_to_write = remaining_until_limit;
    }
  }

  // Calculate the number of subframes already consumed
  const uint8_t total_subframes_per_frame = (XmaContext::kBytesPerFrameChannel / XmaContext::kOutputBytesPerBlock) << data->is_stereo;
  const uint8_t consumed_subframes = total_subframes_per_frame - this->current_frame_remaining_subframes_;
  
  // Calculate read offset in bytes (ensure it's non-negative)
  const uint32_t raw_frame_read_offset = consumed_subframes * XmaContext::kOutputBytesPerBlock;

  output_rb->Write(
      this->raw_frame_.data() + raw_frame_read_offset,
      subframes_to_write * XmaContext::kOutputBytesPerBlock);

  // Reserve extra blocks as headroom when unk_skip_decode is set.
  // Only apply when the frame is fully consumed to avoid double-counting.
  const int8_t headroom =
      (this->current_frame_remaining_subframes_ - subframes_to_write == 0)
          ? data->output_buffer_padding
          : 0;

  this->remaining_subframe_blocks_in_output_buffer_ -= subframes_to_write + headroom;
  this->current_frame_remaining_subframes_ -= subframes_to_write;

  XELOGAPU("XmaContext {}: Consume: {} - {} - {} - {} - {}", this->id(),
           this->remaining_subframe_blocks_in_output_buffer_,
           data->output_buffer_write_offset, data->output_buffer_read_offset,
           output_rb->write_offset(), this->current_frame_remaining_subframes_);
}

void XmaContextMaster::UpdateLoopStatus(XMA_CONTEXT_DATA* data) {
  if (data->loop_count == 0) {
    return;
  }

  const uint32_t loop_start = std::max(kBitsPerPacketHeader, data->loop_start);
  const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);

  XELOGAPU("XmaContext {}: Looped Data: {} < {} (Start: {}) Remaining: {}",
           id(), data->input_buffer_read_offset, data->loop_end,
           data->loop_start, data->loop_count);

  if (data->input_buffer_read_offset != loop_end) {
    return;
  }

  data->input_buffer_read_offset = loop_start;
  loop_start_skip_pending_ = true;

  if (data->loop_count != 255) {
    data->loop_count--;
  }
}

}  // namespace apu
}  // namespace xe
