// Image Rotation with DPC++

#include <CL/sycl.hpp>
#include <array>
#include <math.h>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful header files for image convolution
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

using Duration = std::chrono::duration<double>;
class Timer
{
public:
  Timer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed()
  {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

private:
  std::chrono::steady_clock::time_point start;
};

static const char *inputImagePath = "./Images/cat.bmp";

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

#if 1
class IntMatrix
{
public:
  size_t row, column;
  std::vector<int> elements;

  IntMatrix(size_t r, size_t c, int initVal)
  {
    row = r;
    column = c;
    elements = std::vector<int>(r * c, initVal);
  }
  int e(size_t r, size_t c)
  {
    return elements[r * column + c];
  }
};
// matrice shapes for this example.
constexpr size_t a_rows = 200;
constexpr size_t a_columns = 400;
constexpr size_t b_columns = 600;

// matrix A size: a_rows x a_columns
// matrix B size: a_columns x b_columns
// matrices C an D size: a_rows x b_columns

#endif

float4 *pixel2rgba(float *image_in, size_t ImageRows, size_t ImageCols, image_channel_order chan_order)
{
  // allocate spaces
  float4 *ret = (float4 *)malloc(ImageRows * ImageCols * sizeof(float4));
  if (chan_order == image_channel_order::luminance)
  {
    return ret;
  }
  else
  {
    std::cout << "ERROR: unknown image channel order" << std::endl;
    free(ret);
    return NULL;
  }
}

//************************************
// Image Rotation in DPC++ on device:
//************************************
void ImageRotation(queue &q, void *image_in, void *image_out,
               size_t ImageRows, size_t ImageCols)
{
  // We create images for the input and output data.
  // Images objects are created from a host pointer together with provided
  // channel order and channel type.
  // image_in is a host side buffer of size ImageRows x ImageCols
  // each data item in image_in is float, representing a pixel
  // In the example file cat.bmp, each pixel is of 8-bit color, so we just
  // use "r" as channel order which replicates the value in all R component
  // in the image object
  // The channel type is set as fp32
  //
  image<2> srcImage(image_in, image_channel_order::r, image_channel_type::fp32,
                    range<2>(ImageCols, ImageRows));

  image<2> dstImage(image_out, image_channel_order::r, image_channel_type::fp32,
                    range<2>(ImageCols, ImageRows));

  // Create the range object for the pixel data managed by the image.
  range<2> num_items{ImageCols, ImageRows};

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h)
           {
      // Create an accessor to image with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the image.
      // When accessing images, the accessor element type is used to specify 
      // how the image should be read from or written to. 
      // It can be either int4, uint4 or float4. 
      accessor<float4, 2, access::mode::read, access::target::image> srcPtr(
        srcImage, h);

      accessor<float4, 2, access::mode::write, access::target::image> dstPtr(
        dstImage, h);

      // Samplers are used to specify the way in which the coordinates map to
      // a particular pixel in the image. 
      // In our example, we specify 
      //  (1) the sampler will not use normalized co-ordinates, 
      //  (2) addresses outside the image bounds should clamp to the edge of the image 
      //  (3) and floating-point co-ordinates should take the nearest pixel's data,
      //      rather that applying (for example) a linear filter.
      sampler mysampler(coordinate_normalization_mode::unnormalized,
                    addressing_mode::clamp, filtering_mode::nearest);

      // angle to rotate image by
      float theta = 315.0;

      // Use parallel_for to run image convolution in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // DPC++ supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](id<2> item) 
      { 

        // get row and col of the pixel assigned to this work item
        int source_x = item[0];
        int source_y = item[1];

        // DPC++ Coordinate objects for source and desintation locations
        int2 source_coords;
        int2 destination_coords;

        // Set source coordinates 
        source_coords[0] = source_x;
        source_coords[1] = source_y;

        // Initalize output pixel
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

        // Get source pixel and assign value to sum
        float4 pixel = srcPtr.read(source_coords, mysampler);
        sum[0] = pixel[0];

        // Calculate new position of pixel
        float destination_x = cos(theta)*source_x - sin(theta)*source_y;
        float destination_y = sin(theta)*source_x + cos(theta)*source_y;

        // Set destination coordinates 
        destination_coords[0] = int(destination_x);
        destination_coords[1] = int(destination_y);

        // Range checking
        if (destination_coords[0] >= 0 && destination_coords[0] < ImageCols &&
            destination_coords[1] >= 0 && destination_coords[1] < ImageRows){
              printf("Pixel %d, %d, (%d) rotated to %d, %d \n", source_x, source_y, sum[0], destination_coords[0], destination_coords[1]);
              dstPtr.write(destination_coords, sum);
          }
      }
    ); });
}

int main()
{
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  // default_selector d_selector;
  cpu_selector d_selector;
#endif

  float *hInputImage;
  float *hOutputImage;

  int imageRows;
  int imageCols;
  int i;

  /* Read in the BMP image */
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  /* Allocate space for the output image */
  hOutputImage = (float *)malloc(imageRows * imageCols * sizeof(float));
  for (i = 0; i < imageRows * imageCols; i++)
    hOutputImage[i] = 0.0;

  Timer t;

  try
  {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Image Rotation in DPC++
    ImageRotation
  (q, hInputImage, hOutputImage, imageRows, imageCols);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught for image convolution.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Output image saved as: cat-rotated.bmp\n");
  writeBmpFloat(hOutputImage, "cat-rotated.bmp", imageRows, imageCols,
                inputImagePath);

  return 0;
}
