/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <assert.h>
#include <string.h>
#include <fstream>
#include <iostream>

#include <helper_cuda.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

static int debug_level = 0;
#define DEBUG(level, format, ...) if(debug_level >= level) printf(format __VA_OPT__(,) __VA_ARGS__)

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

void ensureFileOpens(char const *filename){
  std::ifstream infile(filename, std::ifstream::in);

  if (infile.good())
  {
    std::cout << "boxFilterNPP opened: <" << filename
              << "> successfully!" << std::endl;
    infile.close();
  }
  else
  {
    std::cout << "boxFilterNPP unable to open: <" << filename << ">"
              << std::endl;
    infile.close();
    exit(EXIT_FAILURE);
  }
  return;
}

__global__ void watermark_kernel(uchar *dImg, uchar *dWatermark, int nPixels)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPixels; i += blockDim.x * gridDim.x) 
      {
        // for row r, col c, channel ch:
        //   cv::Mat data field layout is array[(nchannels*mat.step*r) + (nchannels*c) + ch];
        //   See: https://stackoverflow.com/questions/37040787/opencv-in-memory-mat-representation
        int bIdx = 3*i;
        int gIdx = 3*i+1;
        int rIdx = 3*i+2;
        uchar bWater = dWatermark[4*i];
        uchar gWater = dWatermark[4*i+1];
        uchar rWater = dWatermark[4*i+2];
        float alpha = ((float) dWatermark[4*i+3])/255.0f;
        dImg[bIdx] = (uchar) ((1.0-alpha)*dImg[bIdx] + alpha*bWater);
        dImg[gIdx] = (uchar) ((1.0-alpha)*dImg[gIdx] + alpha*gWater);
        dImg[rIdx] = (uchar) ((1.0-alpha)*dImg[rIdx] + alpha*rWater);
      }
}


void launch_watermark_kernel(cv::Mat mImg, cv::Mat mWatermark, cv::Mat &mOut)
{
  cv::Mat mWatermarkResized;
  cv::resize(mWatermark, mWatermarkResized, mImg.size(), cv::INTER_LINEAR);

  DEBUG(1, "watermark channels: %d\n", mWatermark.channels());
  DEBUG(1, "mImg step: %d\n", mImg.step);
  DEBUG(1, "mWatermarkResized step: %d\n", mWatermarkResized.step);

  assert(mImg.isContinuous());
  assert(mWatermark.isContinuous());

  uchar *dImg;
  uchar *dWatermark;

  assert(mImg.channels() == mWatermarkResized.channels()-1);
  int imgSize = mImg.total()*mImg.elemSize();
  int watermarkSize = mWatermarkResized.total()*mWatermarkResized.elemSize();
  cudaError_t err = cudaMalloc((void **) &dImg, imgSize);
  if(err != cudaSuccess){
    fprintf(stderr, "cudaMalloc dImg failed: %s\n", cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **) &dWatermark, watermarkSize);
  if(err != cudaSuccess){
    fprintf(stderr, "cudaMalloc dWatermark failed: %s\n", cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(dImg, mImg.data, imgSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "cudaMemcpy dImg mImg Error:\n");
      fprintf(stderr, "%s\n", cudaGetErrorString(err));
      std::exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(dWatermark, mWatermarkResized.data, watermarkSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "cudaMemcpy dWatermark mWatermarkResized Error:\n");
      fprintf(stderr, "%s\n", cudaGetErrorString(err));
      std::exit(EXIT_FAILURE);
  }

  int blockSize, gridSize;
  err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, watermark_kernel);

  DEBUG(1, "pre kernel launch\n");
  int nPixels = mImg.total();
  watermark_kernel<<<gridSize, blockSize>>>(dImg, dWatermark, nPixels);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "CUDA Runtime Error:\n");
      fprintf(stderr, "%s\n", cudaGetErrorString(err));
      std::exit(EXIT_FAILURE);
  }
  DEBUG(1, "post kernel launch\n");
  uchar *data = (uchar *) malloc(imgSize);
  err = cudaMemcpy(data, dImg, imgSize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "cudaMemcpy Error:\n");
      fprintf(stderr, "%s\n", cudaGetErrorString(err));
      std::exit(EXIT_FAILURE);
  }
  cv::Mat out(mImg.size(), mImg.type(), data);
  mOut = out;
  return;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  findCudaDevice(argc, (const char **)argv);

  if (printfNPPinfo(argc, argv) == false)
  {
    exit(EXIT_SUCCESS);
  }

  if(argc != 3 && argc != 4){
    printf("Usage: %s <8bit-rgb-img-filename.png> <8bit-rgba-watermark-filename.png> [output.png]\n\tOutput defaults to output.png\n", argv[0]);
    exit(1);
  }

  ensureFileOpens(argv[1]);
  ensureFileOpens(argv[2]);
  std::string sImgFile = argv[1];
  std::string sWatermarkFile = argv[2];
  std::string sOutputFile = (argc == 4) ? argv[3] : "output.png";

  cv::Mat mImg = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat mWatermark = cv::imread(argv[2], cv::IMREAD_UNCHANGED);

  cv::Mat mOut;
  launch_watermark_kernel(mImg, mWatermark, mOut);
  imwrite(sOutputFile, mOut);

  return 0;
}
