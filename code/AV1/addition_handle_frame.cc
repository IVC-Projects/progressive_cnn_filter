/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/common/addition_handle_frame.h"

uint8_t *getYbuf(uint8_t *yPxl, int height, int width, int stride) {
  if (!yPxl) return nullptr;
  int buflen = height * width;
  unsigned char *buf = new unsigned char[buflen];  //定义一块一帧图像大小的buf
  unsigned char *p = buf;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      unsigned char uctemp = (unsigned char)(*(yPxl + x));
      *p = uctemp;
      p++;
    }
    yPxl += stride;  //一行的一个Y值对应下一行同样位置的增量
  }
  return buf;
}

/*Feed full frame image into the network*/
void addition_handle_frame(AV1_COMMON *cm) {
  YV12_BUFFER_CONFIG *pcPicYuvRec = &cm->cur_frame->buf;

  if (!cm->seq_params.use_highbitdepth) {
    uint8_t *py = pcPicYuvRec->y_buffer;
    uint8_t *bkuPy = py;

    int height = pcPicYuvRec->y_height;
    int width = pcPicYuvRec->y_width;
    int stride = pcPicYuvRec->y_stride;
    uint8_t **buf = new uint8_t *[height];
    for (int i = 0; i < height; i++) {
      buf[i] = new uint8_t[width];
    }
    if (cm->current_frame.order_hint % 16 == 0 &&
        cm->current_frame.order_hint != 0)
      buf = TF_Predict(py, height, width, stride, cm->base_qindex - 80,
                       cm->current_frame.frame_type);
    else
      buf = TF_Predict(py, height, width, stride, cm->base_qindex,
                       cm->current_frame.frame_type);

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        *(bkuPy + j) = buf[i][j];  // Fill in the luma buffer again
                                   // fwrite(bkuPy + j, 1, 1, ff);
      }
      bkuPy += stride;
    }
    // fclose(ff);
  } else {
    uint16_t *py = CONVERT_TO_SHORTPTR(pcPicYuvRec->y_buffer);
    uint16_t *bkuPy = py;

    int height = pcPicYuvRec->y_height;
    int width = pcPicYuvRec->y_width;
    int stride = pcPicYuvRec->y_stride;

    uint16_t **buf = new uint16_t *[height];
    for (int i = 0; i < height; i++) {
      buf[i] = new uint16_t[width];
    }

    buf = TF_Predict_hbd(py, height, width, stride);

    // FILE *ff = fopen("end.yuv", "wb");
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        *(bkuPy + j) = buf[i][j];  // Fill in the luma buffer again
        // fwrite(bkuPy + j, 1, 1, ff);
      }
      bkuPy += stride;
    }
  }
  finish_python();
}


