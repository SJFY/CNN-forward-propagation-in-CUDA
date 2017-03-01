#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>
#include <cuda_profiler_api.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

#define KERNEL_WIDTH 5 //becasue filter height and width are always 5
__constant__ int devW1d[4] = {5,5,1,32};
__constant__ int devW2d[4] = {5,5,32,64};
//Can fit 1st conv filter into constant memory. Can fit 2nd conv
//filter (0x32000 bytes) into constant memory
__constant__ float constantW[5*5*1*32];

__constant__ float dev_cons_Wunroll[15200];

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32}; //32 filters
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

//https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
/*Dev count = 1
 maxThreads = 1024
 Shared Mem per Block (B) = 49152 (48KB)
 Regs per Block = 65536
 Warp Size = 32
 Total Const Mem (B) = 65536
 Total Global Mem (B) = 3405643776
 */
void device_query(){
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("Dev count = %d\n",dev_count);
    cudaDeviceProp dev_prop;
    for(int i=0; i < dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        //to print device resources and capabilities
        printf("maxThreads = %d\n",dev_prop.maxThreadsPerBlock);
        printf("Shared Mem per Block (B) = %d\n",(int) dev_prop.sharedMemPerBlock);
        printf("Regs per Block = %d\n",dev_prop.regsPerBlock);
        printf("Warp Size = %d\n",dev_prop.warpSize);
        printf("Total Const Mem (B) = %u\n",(unsigned int) dev_prop.totalConstMem);
        printf("Total Global Mem (B) = %u\n",(unsigned int) dev_prop.totalGlobalMem);
        printf("Overlap possible = %d",dev_prop.deviceOverlap);
    }
}

//Each thread in 32x32 block will load 1 element of X into shared memory
//No need to load W as its in constant memory
__global__ void MatrixMultiply(const float* A, const float* B, float* C,
                          const int numARows, const int numAColumns,
                          const int numBRows, const int numBColumns,
                          const int numCRows, const int numCColumns, const int num_of_X_tiles){
    const int TILE_WIDTH = 32; //Set Tile_Width as 32
    //Assertions to check my matrix sizes
    assert(numAColumns == numBRows);
    assert(numARows == numCRows);
    assert(numBColumns == numCColumns);
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;
    int Col = tx + TILE_WIDTH * blockIdx.x;
    int Row = ty + TILE_WIDTH * blockIdx.y;
    float Cvalue = 0;
    //Load only X into shared memory
    for(int m=0; m < num_of_X_tiles; ++m ){
        if((Row < numCRows) && (m*TILE_WIDTH+tx) < numAColumns)
            subTileA[ty][tx] = A[Row*numAColumns+m*TILE_WIDTH+tx];
        else
            subTileA[ty][tx] = 0;
        if((Col < numCColumns) && (m*TILE_WIDTH+ty) < numBRows)
            subTileB[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
        else
            subTileB[ty][tx] = 0;
        
        __syncthreads();
        for(int k=0; k < TILE_WIDTH; ++k){
            Cvalue += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }
    if((Row < numCRows) && (Col < numCColumns))
        C[Row*numCColumns + Col] = Cvalue;
}

/*This kernel will remap X from X[batch_size, height, width, # of input feature map] to X[batchsize,# of feature map, height, width]
dim3 ReMapBlock(32,32,1);
//Each Thread block processes all channels in 28 and 28 image 
Remap_x = ceil( xdims[2]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is column number //1
Remap_y = ceil( xdims[1]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is row number    //1
Remap_z = ceil( xdims[0]/1.0f);  // BlockIDx.z*blockDim.z+threadIdx.z is batch number  //10
dim3 ReMapGrid(Remap_x, Remap_y, Remap_z);*/
//Took 16us for 1st convo, took 82us for 2nd convo
__global__ void n_RemapX(const float*X, const int* xdims, float* Xremapped){
	//if(xdims[3] == 1)
	//	return;
	//inputindex =  i * xdims[1] * xdims[2] * xdims[3]
    //				+ height * xdims[2] * xdims[3]
    //				+  width * xdims[3] + channel;
	int batch = blockIdx.z*blockDim.z + threadIdx.z;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int channel = 0; //Channel is 0 at start
	
	const int inputindex = 	batch * xdims[1] * xdims[2] * xdims[3] +  		// cancelled out ( blockIdx.z/xdims[1] ) * xdims[1]
							row * xdims[2] * xdims[3] +
							col * xdims[3] + channel;
	//outputindex = i * xdims[1] * xdims[2] * xdims[3] +
	//				channel * xdims[1] * xdims[2] +
	//				height * xdims[2] + width;
	const int outputindex = batch * xdims[1] * xdims[2] * xdims[3] + 		//( blockIdx.z/xdims[1] ) * xdims[1]
							channel * xdims[1] * xdims[2] +
							row * xdims[2] + col;
	if(row < xdims[1] && col < xdims[2] && batch < xdims[0] ){
		for(channel = 0; channel < xdims[3]; channel++ )
			Xremapped[outputindex + channel*xdims[1]*xdims[2] ] = X[inputindex + channel];
	}
}

/*This kernel will remap X from X[batch_size, height, width, # of input feature map] to X[batchsize,# of feature map, height, width]
dim3 ReMapBlock(32,32,1);
//Each Thread block processes all batches in FOR loop 
Remap_x_coal = ceil( xdims[3]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is channel 		  		//1
Remap_y_coal = ceil( xdims[2]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is column number			//1
Remap_z_coal = ceil( xdims[1]/1.0f);  // BlockIDx.z*blockDim.z+threadIdx.z is row number			//10
dim3 ReMapGrid(Remap_x, Remap_y, Remap_z);*/
//Took 140us for 1st convo, took 55us for 2nd convo
__global__ void n_RemapX_memory_coalesced(const float*X, const int* xdims, float* Xremapped){
	//inputindex =  i * xdims[1] * xdims[2] * xdims[3]
    //				+ height * xdims[2] * xdims[3]
    //				+  width * xdims[3] + channel;
	int batch = 0; //Batch is 0 at start for every block
	int row = blockIdx.z*blockDim.z + threadIdx.z;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int channel = blockIdx.x*blockDim.x + threadIdx.x;
	
	const int inputindex = 	batch * xdims[1] * xdims[2] * xdims[3] +  		// cancelled out ( blockIdx.z/xdims[1] ) * xdims[1]
							row * xdims[2] * xdims[3] +
							col * xdims[3] + channel;
	//outputindex = i * xdims[1] * xdims[2] * xdims[3] +
	//				channel * xdims[1] * xdims[2] +
	//				height * xdims[2] + width;
	const int outputindex = batch * xdims[1] * xdims[2] * xdims[3] + 		//( blockIdx.z/xdims[1] ) * xdims[1]
							channel * xdims[1] * xdims[2] +
							row * xdims[2] + col;
	if(row < xdims[1] && col < xdims[2] && channel < xdims[3] ){
		for(batch = 0; batch < xdims[0]; batch++ )
			Xremapped[outputindex + batch*xdims[1]*xdims[2]*xdims[3] ] = X[inputindex + batch*xdims[1]*xdims[2]*xdims[3]]; //X[inputindex];
	}
}

/*Read 1 input feature map from remappedX and store it in unrolled format.
 1 thread block processes 1 feature map X ie 28*28*1 for 1st convo

 Plan: Use 1 thread to read and copy 1 element of X feature map into shared memory,
 then do syncthread. 
 Then use each thread to read 1 element from shared memory and write in 
 upto 5 locations in X_unrolled table.
 After this kernel Unrolled X table should be complete.
 dim3 unrolBlock(1024,1,1);   //1024 threads in X direction
 unrol_x = ceil( xdims[3] * ydims[2]*ydims[1]/1024.0f ) //BlockIDx.x*blockDim.x+threadIdx.x is column number
 //Notused unrol_y = ceil( xdims[1]/1.0f ) //BlockIDx.y*blockDim.y+threadIdx.y is row number
 //Notused unrol_z = ceil( xdims[3]/1.0f ) //BlockIDx.z*blockDim.z+threadIdx.z is channel number
 dim3 unrolGrid(unrol_x, 1, 1); //One block processing 1024 elements in each channel in each batch
 
     //Unrolled table height is kernel^2*xdims[3] = 25 * no of channels
	//There will be batch_size (num_of_input_maps) number of tables 
    //Unrolled table Width is number of possible kernel locations = (xdims[1] - kernel+1)^2
    //Each Thread Block should process 1 feature map ie 1 channel
    //No of blocks in X is (1*24*24)/1024
    //No of blocks in Y is 1
 */

__global__ void n_unrolX_prof(const float* X, const int* xdims, const int batch, float* X_unrol){
	//Batch is a variable given to every thread block
	const int t = blockIdx.x*blockDim.x + threadIdx.x;
	const int W_out = xdims[1] - 5 + 1;
	const int W_unroll = W_out*W_out; 	//This is width of unrolled kernel
	if( t < xdims[3] * W_unroll ){
		int channel = t/W_unroll; 		//channel number or feature map number
		int s = t%W_unroll;				//column to write within the feature map
		int row = s/W_out; 				//row in feature map to read
		int col = s%W_out;				//col in feature map to read
		int h_unroll = row*W_out + col;	//base in unrolled kernel
		
		const int inputindex = 	batch * xdims[1] * xdims[2] * xdims[3] + 
								channel * xdims[1] * xdims[2] +
								row * xdims[2] + col;
		int w_base = channel*25;
		for(int p=0; p <5; p++)
			for(int q=0; q <5; q++){ // in X
				int xunroll = w_base + p*5 + q;
				X_unrol[batch *(25*xdims[3]*W_unroll) + xunroll*W_unroll + h_unroll] = X[	batch*xdims[1] * xdims[2] * xdims[3] + channel*(xdims[1]*xdims[2])+(row+p)*xdims[1] + (col+q)];
			}
	}
}								
/*
__global__ void n_unrolX(const float* X, const int* xdims, const int batch, float* X_unrol){
	//Batch is a variable given to every thread block
	const int t = blockIdx.x*blockDim.x + threadIdx.x;
	const int W_out = xdims[1] - 5 + 1;
	const int W_unroll = W_out*W_out; 	//This is width of unrolled kernel

	//Slide down 5 times, so each thread moves down 5 times
		for(int i = 0; i < 5; i++){
			row=row+i*5;
			if(row < xdims[1] && col < xdims[2] && channel < xdims[3] ){ //Only proceed if thread is valid
				float tempreg = X[inputindex];
				X_unrol[row*W_unroll+col] = tempreg;
				if(col%5 == 1)
					X_unrol[(row+1) * W_unroll+ (col-1)] = tempreg;
				if(col%5 == 2){
					X_unrol[(row+1) * W_unroll+ (col-1)] = tempreg;
					X_unrol[(row+2) * W_unroll+ (col-2)] = tempreg;
				}
				if(col%5 == 3){
					X_unrol[(row+1) * W_unroll+ (col-1)] = tempreg;
					X_unrol[(row+2) * W_unroll+ (col-2)] = tempreg;
					X_unrol[(row+3) * W_unroll+ (col-3)] = tempreg;
				}
				if(col%5 == 4){
					X_unrol[(row+1) * W_unroll+ (col-1)] = tempreg;
					X_unrol[(row+2) * W_unroll+ (col-2)] = tempreg;
					X_unrol[(row+3) * W_unroll+ (col-3)] = tempreg;
					X_unrol[(row+4) * W_unroll+ (col-4)] = tempreg;
				}
			}
		}
	}
}
*/
/*Read X input feature maps and store them in unrolled table.
 1 thread block processes 1 feature map X
Plan: Use 1 thread to read and copy 1 element of X feature map into shared memory,
 then do syncthread. 
 Then use each thread to read 1 element from shared memory and write in 
 upto 5 locations in X_unrolled table.
 After this kernel Unrolled X table should be complete. */
/*__global__ void n_unrolX_old(const float* X, const int* Xdims, float* X_unrol){
    __shared__ float temp[ 28 *28*1]; //Gave it the highest possible Xdims[1]*Xdims[2]*Xdims[3]
    //We will not be using the full shared array with smaller feature maps
	int col_i = threadIdx.x + blockIdx.x * blockDim.x;
    int row_i = threadIdx.y*5 + blockIdx.y * blockDim.y; // Go to every 5th row
    int offset = row_i * Xdims[2] + col_i;
	//Dont use shared memory for now
    __shared__ float temp[28*28]; //Gave it the highest possible Xdims[1]*Xdims[2]*Xdims[3]
	if (row_i < xdims[1] && col_i < xdims[2] )
		temp[row*28+col] = X[inputindex];
	__syncthreads();
	//Now the whole feature map is copied into shared memory.
	//Now the whole feature map is copied into shared memory.
    //Now all threads can use shared memory to make unrol table.
    //28 Threads. Each Thread writes a full column, and tries to fill columns to its left.
    //This reduces control divergence. Only first 4 threads will have control divergence
    //ONly threads in Row0,5,10,15 etc should do work. Doesnt affect divergence.
    int width = (Xdims[1] - 5 + 1)*(Xdims[1] - 5 + 1); //This is width of unrolled kernel
    int boundary = Xdims[1] - 5 + 1;
    if(col_i + 5 < Xdims[2] && row_i % 5 == 0 && row_i+5 < Xdims[1]){
        X_unrol[row_i*width+col_i] = X[offset];
        if(col_i -1 >= 0)
            X_unrol[ (row_i+1) *width+ (col_i-1) ] = X[offset];
        if(col_i -2 >= 0)
            X_unrol[ (row_i+2) *width+ (col_i-2) ] = X[offset];
        if(col_i -3 >= 0)
            X_unrol[ (row_i+3) *width+ (col_i-3) ] = X[offset];
        if(col_i -4 >= 0)
            X_unrol[ (row_i+4) *width+ (col_i-4) ] = X[offset];
    }
    //This covers the right hand side boundary condition
    if(col_i == boundary && row_i % 5 == 0 && row_i + 5 < Xdims[1] ){
        X_unrol[row_i     *width+col_i] = X[offset];
        X_unrol[(row_i+1) *width+col_i] = X[offset+1];
        X_unrol[(row_i+2) *width+col_i] = X[offset+2];
        X_unrol[(row_i+3) *width+col_i] = X[offset+3];
        X_unrol[(row_i+4) *width+col_i] = X[offset+4];
    }
    //This covers the bottom boundary condition
    if(col_i + 5 < Xdims[2] && row_i == boundary){
        X_unrol[row_i*width+col_i] = X[offset];
        if(col_i -1 >= 0)
            X_unrol[ (row_i+1) *width+col_i -1 ] = X[offset];
        if(col_i -2 >= 0)
            X_unrol[ (row_i+2) *width+col_i -2 ] = X[offset];
        if(col_i -3 >= 0)
            X_unrol[ (row_i+3) *width+col_i -3 ] = X[offset];
        if(col_i -4 >= 0)
            X_unrol[ (row_i+4) *width+col_i -4 ] = X[offset];
    }
}*/

//3D matrix multiply
__global__ void z_MatrixMultiply(const float* A, const float* B, float* C,
                          const int numARows, const int numAColumns,
                          const int numBRows, const int numBColumns,
                          const int numCRows, const int numCColumns, const int num_of_X_tiles, const int* outputdims, const int* inputdims){
    const int TILE_WIDTH = 32; //Set Tile_Width as 32
    //Assertions to check my matrix sizes
    assert(numAColumns == numBRows);
    assert(numARows == numCRows);
    assert(numBColumns == numCColumns);
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;
    int Col = tx + TILE_WIDTH * blockIdx.x;
    int Row = ty + TILE_WIDTH * blockIdx.y;
    float Cvalue = 0;
    //Load only X into shared memory
    for(int m=0; m < num_of_X_tiles; ++m ){
        if((Row < numCRows) && (m*TILE_WIDTH+tx) < numAColumns)
            subTileA[ty][tx] = A[Row*numAColumns+m*TILE_WIDTH+tx];
        else
            subTileA[ty][tx] = 0;
        if((Col < numCColumns) && (m*TILE_WIDTH+ty) < numBRows)
            subTileB[ty][tx] = B[blockIdx.z * outputdims[1] * outputdims[2] * 25 * inputdims[3]  + (m*TILE_WIDTH+ty)*numBColumns+Col];
        else
            subTileB[ty][tx] = 0;
        
        __syncthreads();
        for(int k=0; k < TILE_WIDTH; ++k){
            Cvalue += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }
    if((Row < numCRows) && (Col < numCColumns)){
    	int h = Col/outputdims[2];
    	int w = Col%outputdims[2];
    	int i = blockIdx.z;
    	int m = Row;
    	int yoffset = ((i * outputdims[1] + h) * outputdims[2] + w) * outputdims[3] + m;
        //C[yoffset] = Cvalue;
        C[yoffset] = (Cvalue < 0) ? 0: Cvalue;	// Doing relu4 right here

    }
}

/*
    int x_blocks = ceil(numCColumns/32.0f);
    int y_blocks = ceil(numCRows/32.0f);
    dim3 z_DimGrid(x_blocks, y_blocks, 1);
    dim3 z_DimBlock(32, 32, 1);
	for(int batch =0; batch < ydims[0]; batch++){
*/
__global__ void z_MatrixMultiply_perbatch(const float* A, const float* B, float* C,
                          const int numARows, const int numAColumns,
                          const int numBRows, const int numBColumns,
                          const int numCRows, const int numCColumns, const int num_of_X_tiles, const int* outputdims, const int* inputdims, int batch){
    const int TILE_WIDTH = 32; //Set Tile_Width as 32
    //Assertions to check my matrix sizes
    assert(numAColumns == numBRows);
    assert(numARows == numCRows);
    assert(numBColumns == numCColumns);
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;
    int Col = tx + TILE_WIDTH * blockIdx.x;
    int Row = ty + TILE_WIDTH * blockIdx.y;
    float Cvalue = 0;
    //Load only X into shared memory
    for(int m=0; m < num_of_X_tiles; ++m ){
        if((Row < numCRows) && (m*TILE_WIDTH+tx) < numAColumns)
            subTileA[ty][tx] = A[Row*numAColumns+m*TILE_WIDTH+tx];
        else
            subTileA[ty][tx] = 0;
        if((Col < numCColumns) && (m*TILE_WIDTH+ty) < numBRows)
            subTileB[ty][tx] = B[batch * outputdims[1] * outputdims[2] * 25 * inputdims[3]  + (m*TILE_WIDTH+ty)*numBColumns+Col];
        else
            subTileB[ty][tx] = 0;
        
        __syncthreads();
        for(int k=0; k < TILE_WIDTH; ++k){
            Cvalue += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }
    if((Row < numCRows) && (Col < numCColumns)){
      int h = Col/outputdims[2];
      int w = Col%outputdims[2];
      int i = batch;
      int m = Row;
     // int yoffset = ((i * outputdims[1] + h) * outputdims[2] + w) * outputdims[3] + m;
      int yoffset = i*outputdims[1]*outputdims[2]*outputdims[3] + m * outputdims[1]* outputdims[2]+h*outputdims[2]+w;
        //C[yoffset] = Cvalue;
        C[yoffset] = (Cvalue < 0) ? 0: Cvalue;	// Doing relu4 right here
    }
}

__global__ void z_MatrixMultiply_perbatch_constant(const float* B, float* C,
                          const int numARows, const int numAColumns,
                          const int numBRows, const int numBColumns,
                          const int numCRows, const int numCColumns, const int num_of_X_tiles, const int* outputdims, const int* inputdims, int batch){
    const int TILE_WIDTH = 32; //Set Tile_Width as 32
    //Assertions to check my matrix sizes
    assert(numAColumns == numBRows);
    assert(numARows == numCRows);
    assert(numBColumns == numCColumns);
    //__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x;  int by = blockIdx.y;
    int Col = tx + TILE_WIDTH * blockIdx.x;
    int Row = ty + TILE_WIDTH * blockIdx.y;
    float Cvalue = 0;
    //Load only X into shared memory
    for(int m=0; m < num_of_X_tiles; ++m ){
       /* if((Row < numCRows) && (m*TILE_WIDTH+tx) < numAColumns)
            subTileA[ty][tx] = dev_cons_Wunroll[Row*numAColumns+m*TILE_WIDTH+tx];
        else
            subTileA[ty][tx] = 0;*/
        if((Col < numCColumns) && (m*TILE_WIDTH+ty) < numBRows)
            subTileB[ty][tx] = B[batch * outputdims[1] * outputdims[2] * 25 * inputdims[3]  + (m*TILE_WIDTH+ty)*numBColumns+Col];
        else
            subTileB[ty][tx] = 0;
        
        __syncthreads();
        for(int k=0; k < TILE_WIDTH; ++k){
          if((m*TILE_WIDTH+k)<numAColumns){
            //Cvalue += subTileA[ty][k] * subTileB[k][tx];
            Cvalue += dev_cons_Wunroll[Row*numAColumns+m*TILE_WIDTH+k] * subTileB[k][tx];
          }
          else{
            Cvalue += 0;
          }
        }
        __syncthreads();
    }
    if((Row < numCRows) && (Col < numCColumns)){
      int h = Col/outputdims[2];
      int w = Col%outputdims[2];
      int i = batch;
      int m = Row;
      //int yoffset = ((i * outputdims[1] + h) * outputdims[2] + w) * outputdims[3] + m;
      //remapped yoffset
      int yoffset = i*outputdims[1]*outputdims[2]*outputdims[3] + m * outputdims[1]* outputdims[2]+h*outputdims[2]+w;
        //C[yoffset] = Cvalue;
        C[yoffset] = (Cvalue < 0) ? 0: Cvalue;
    }
}

/*
	dim3 z_unrolGrid(xdims[3], xdims[0], 1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, */
__global__ void z_unrolX(const float* X, const int* Xdims, float* X_unrol){
    __shared__ float temp[28*28]; //one block only take care of one input feature map, shared memory only store for one inputfeture map elements, highest is first convolution
    
    int channel_i = blockIdx.x; //input no., in the case of 2ed convolution, it's from 0 to 31, 32 input in total
    
    int Xoffset = blockIdx.y * Xdims[1] * Xdims[2] * Xdims[3] + threadIdx.y * Xdims[2] * Xdims[3] + threadIdx.x * Xdims[3] + channel_i;
    //blockIdx.y is batch no., 
	int tempoffset = threadIdx.y * Xdims[2] + threadIdx.x; // 2 dimention block, shared memory is 2D, one thread copy one element into shared memory, in 2ed convolution, totaly 12*12 elements= threads number in block
    temp[tempoffset] = X[Xoffset];
    //temp[tempoffset] = tempoffset;
    __syncthreads();

    //Row in grid means batch, row 0 means batch 0. Column means input no., 
    //Blocks in Row 0 Column 0 will write the first 25 (0~24)rows of unrow matrix of batch0
	//Blocks in Row 0 Column 1 will write the second 25 (25~49)rows of unrow matrix of batch0
	//Block dims is 12 * 12 in second convolution, but only 8*8 threads work, each thread write a whole column of 25 elements.
	//unrow matrix is global memory, write in the global memory, different thread write into different element, no need to atomic add
	//threads only need to read data from shared memory and write into global memory, fast enough
    //Problem:
	//1.may have control divergence, not all threads in one block are working
	//2.second convolution, the block is too small, only 144 threads, and have 32 * batch no. blocks in total, maybe too mank blocks but each block has few threads
   
    int outputwidth = Xdims[1] - 5 + 1; //output feature map width, equal to height
    int unrollwidth = outputwidth * outputwidth;
	if((threadIdx.x < outputwidth) && (threadIdx.y < outputwidth)){  // we only need output feature map elements number of threads to work, 8*8 in the second convolution
    	//recall that blockIdx.x is input map no., blockIdx.y is batch no.
		int unroll_col = threadIdx.y * outputwidth + threadIdx.x;   // col index in unroll matrix
    	int unroll_row_base = 5 * 5 * Xdims[3] * blockIdx.y + blockIdx.x * 5 * 5;
    	int i = 0;

    	for(int p = 0; p < 5; p++)
    		for(int q = 0; q < 5; q++){
    		X_unrol[(unroll_row_base + i) * unrollwidth + unroll_col] = temp[(threadIdx.y + p) * Xdims[2] + (threadIdx.x + q) ];   // not sure about the p and q, I think it doesn't matter
				//X_unrol[(unroll_row_base + i) * unrollwidth + unroll_col] = 8;
				//X_unrol[(unroll_row_base + i) * unrollwidth + unroll_col] += 1;
				i++;
    		}
    }
}

/*
	dim3 z_unrolGrid(xdims[3], xdims[0], 1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, */

__global__ void z_unrolX_remapped(const float* X, const int* Xdims, float* X_unrol){
    __shared__ float temp[28*28]; //one block only take care of one input feature map, shared memory only store for one inputfeture map elements, highest is first convolution
    
    int channel_i = blockIdx.x; //input no., in the case of 2ed convolution, it's from 0 to 31, 32 input in total
    //remapped index
	//outputindex = i * xdims[1] * xdims[2] * xdims[3] +
	//				channel * xdims[1] * xdims[2] +
	//				height * xdims[2] + width;
	int Xoffset = blockIdx.y * Xdims[1] * Xdims[2] * Xdims[3] + channel_i* Xdims[2] * Xdims[1] + threadIdx.y * Xdims[2] + threadIdx.x ;
    //blockIdx.y is batch no., 
	int tempoffset = threadIdx.y * Xdims[2] + threadIdx.x; // 2 dimention block, shared memory is 2D, one thread copy one element into shared memory, in 2ed convolution, totaly 12*12 elements= threads number in block
    temp[tempoffset] = X[Xoffset];
    //temp[tempoffset] = tempoffset;
    __syncthreads();

    //Row in grid means batch, row 0 means batch 0. Column means input no., 
    //Blocks in Row 0 Column 0 will write the first 25 (0~24)rows of unrow matrix of batch0
	//Blocks in Row 0 Column 1 will write the second 25 (25~49)rows of unrow matrix of batch0
	//Block dims is 12 * 12 in second convolution, but only 8*8 threads work, each thread write a whole column of 25 elements.
	//unrow matrix is global memory, write in the global memory, different thread write into different element, no need to atomic add
	//threads only need to read data from shared memory and write into global memory, fast enough
    //Problem:
	//1.may have control divergence, not all threads in one block are working
	//2.second convolution, the block is too small, only 144 threads, and have 32 * batch no. blocks in total, maybe too mank blocks but each block has few threads
   
    int outputwidth = Xdims[1] - 5 + 1; //output feature map width, equal to height
    int unrollwidth = outputwidth * outputwidth;
	if((threadIdx.x < outputwidth) && (threadIdx.y < outputwidth)){  // we only need output feature map elements number of threads to work, 8*8 in the second convolution
    	//recall that blockIdx.x is input map no., blockIdx.y is batch no.
		int unroll_col = threadIdx.y * outputwidth + threadIdx.x;   // col index in unroll matrix
    	int unroll_row_base = 5 * 5 * Xdims[3] * blockIdx.y + blockIdx.x * 5 * 5;
    	int i = 0;

    	for(int p = 0; p < 5; p++)
    		for(int q = 0; q < 5; q++){
    		X_unrol[(unroll_row_base + i) * unrollwidth + unroll_col] = temp[(threadIdx.y + p) * Xdims[2] + (threadIdx.x + q) ];   // not sure about the p and q, I think it doesn't matter
				//X_unrol[(unroll_row_base + i) * unrollwidth + unroll_col] = 8;
				//X_unrol[(unroll_row_base + i) * unrollwidth + unroll_col] += 1;
				i++;

    		}
    }

}

__global__ void z_unrollw(const float* originallayout, float* multiplylayout, const int* filterdims){
	//filterdims is wdims, col and row is in unrollw
	//recall the threadIdx.x and threadIdx.y is 5, threadIdx.z is input no., blockIdx.x is output no.
	int col = threadIdx.z * 5 * 5 + threadIdx.y * 5 + threadIdx.x; // not sure about the x and y, 90% sure
	int row = blockIdx.x;    //for filters calculating output map0, stored in row0.
	//this multiplylayout is 2D array, have total 25*wdims[2]*wdims[3] elements, same as threads in grid.
  multiplylayout[row * filterdims[2] * 5 * 5 + col] = originallayout[threadIdx.y * filterdims[1] * filterdims[2] * filterdims[3] + threadIdx.x * filterdims[2] * filterdims[3] + threadIdx.z * filterdims[3] + blockIdx.x];
}
__global__ void conv_forward_kernel(const float *X, const int Xdims[4], float *W, bool conv1d,
                                    float *Y, const int Ydims[4]) {
    
    //if(conv1d == true)
    //use devW1d
    //else use devW2d
    //Wdims[0] and Wdims[1] is always 5
    __shared__ float Xs[32][32];
    __shared__ float Ws[32][32];
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < Xdims[2] && row < Xdims[1])       //loading X feature map
        Xs[row][col] = X[row * Xdims[2] + col];
    else
        Xs[row][col] = 0;
    if ((col < KERNEL_WIDTH) && (row < KERNEL_WIDTH))       //loading W conv kernel
        Ws[row][col] = W[row * KERNEL_WIDTH + col];
    else
        Ws[row][col] = 0;
    __syncthreads();
    //list reduction optimization
    float temp[] = {0};  //Temp array with size = num of feature map
    /*for(int i =0 ; i < KERNEL_WIDTH * KERNEL_WIDTH; i++){
        temp[feature_map_index] += Xs[row][col]*Ws[row][col];
    }*/
    
}
__global__ void n_averagepool_coalesced(const float* X, const int* xdims, const int pool_size,
                                    float* Y, const int* ydims, const int batch){
     //inputindex =  i * xdims[1] * xdims[2] * xdims[3]
     //       + row * xdims[2] * xdims[3]
     //       +  col * xdims[3] + channel;
     int col = blockIdx.x*blockDim.x + threadIdx.x;
     int row = blockIdx.y*blockDim.y + threadIdx.y;
     int channel = blockIdx.z*blockDim.z + threadIdx.z;
     //int batch = 0; //Batch is 0 at start
    /* int outputindex = batch * ydims[1]*ydims[2]*ydims[3] +
                        channel * ydims[1]*ydims[2] +
                        row * ydims[1] + col;*/
     int inputindex  = batch * xdims[1]*xdims[2]*xdims[3] +
                        channel * xdims[1]*xdims[2] +
                        2*row * xdims[1] + 2*col;
     int outputindex =  batch * ydims[1] * ydims[2] * ydims[3] +
                        row * ydims[2] * ydims[3] +
                        col * ydims[3] + channel;
    /* int inputindex = batch * xdims[1]* xdims[2]* xdims[3]+
                      2*row*xdims[2]*xdims[3] +
                      2*col*xdims[3] + channel;*/
     
     if (col < ydims[2] && row < ydims[1] && channel < ydims[3]){
         
         float Cvalue = 0;
         Cvalue += X[inputindex]/4.0;
         Cvalue += X[inputindex+xdims[1]]/4.0;
         Cvalue += X[inputindex+1]/4.0;
         Cvalue += X[inputindex+1+xdims[1]]/4.0;
         
         /*for(int p=0; p < pool_size; p++){ //p is width//poolsize = 2 always
             for(int q=0; q < pool_size; q++){ //q is height
                int inputindex = batch * xdims[1]*xdims[2]*xdims[3]+
                                (row+q)*xdims[2]*xdims[3] +
                                (col+p)*xdims[3] + channel;
                Cvalue += X[inputindex]/4.0f;
             //Cvalue = Cvalue + A[Row*numAColumns+k] * B[Col+k*numBColumns];
             //Cvalue=Cvalue+A[k][Row] * B[Col][k];
             }
         }*/
         Y[outputindex] = Cvalue; //inputindex; //C[output]= Cvalue;
     }
 }


/*This kernel will read convolution result in memory coalesced way and
 do averagepool.
 //One thread represents 1 element in output matrix and merges 4 pixels into 1
 dim3 avgPoolBlock(32,32,1);
 int avgpool_x_coal = ceil( bdims[2]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is column number 	//1
 int avgpool_y_coal = ceil( bdims[1]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is row number		//1
 int avgpool_z_coal = ceil( bdims[3]/1.0f);  // BlockIDx.z*blockDim.z+threadIdx.z is channel   //32 then 64*/
 __global__ void n_averagepool_output_remapped(const float* X, const int* xdims, const int pool_size,
                                    float* Y, const int* ydims, const int batch){
     //inputindex =  i * xdims[1] * xdims[2] * xdims[3]
     //       + row * xdims[2] * xdims[3]
     //       +  col * xdims[3] + channel;
     int col = blockIdx.x*blockDim.x + threadIdx.x;
     int row = blockIdx.y*blockDim.y + threadIdx.y;
     int channel = blockIdx.z*blockDim.z + threadIdx.z;
     //int batch = 0; //Batch is 0 at start
     int outputindex = batch * ydims[1]*ydims[2]*ydims[3] +
                        channel * ydims[1]*ydims[2] +
                        row * ydims[1] + col;
     int inputindex  = batch * xdims[1]*xdims[2]*xdims[3] +
                        channel * xdims[1]*xdims[2] +
                        2*row * xdims[1] + 2*col;
    /* int outputindex =  batch * ydims[1] * ydims[2] * ydims[3] +
                        row * ydims[2] * ydims[3] +
                        col * ydims[3] + channel;*/
    /* int inputindex = batch * xdims[1]* xdims[2]* xdims[3]+
                      2*row*xdims[2]*xdims[3] +
                      2*col*xdims[3] + channel;*/
     
     if (col < ydims[2] && row < ydims[1] && channel < ydims[3]){
         
         float Cvalue = 0;
         Cvalue += X[inputindex]/4.0;
         Cvalue += X[inputindex+xdims[1]]/4.0;
         Cvalue += X[inputindex+1]/4.0;
         Cvalue += X[inputindex+1+xdims[1]]/4.0;
         
         /*for(int p=0; p < pool_size; p++){ //p is width//poolsize = 2 always
             for(int q=0; q < pool_size; q++){ //q is height
                int inputindex = batch * xdims[1]*xdims[2]*xdims[3]+
                                (row+q)*xdims[2]*xdims[3] +
                                (col+p)*xdims[3] + channel;
                Cvalue += X[inputindex]/4.0f;
             //Cvalue = Cvalue + A[Row*numAColumns+k] * B[Col+k*numBColumns];
             //Cvalue=Cvalue+A[k][Row] * B[Col][k];
             }
         }*/
         Y[outputindex] = Cvalue; //inputindex; //C[output]= Cvalue;
     }
 }

__global__ void fullyforward(const float *X, const int* Xdims, const float *W, const int* Wdims,
                              float *Y, const int* Ydims ){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int i = index / Ydims[1];  //batch number
  int j = index % Ydims[1];  //element number inside a batch
  float sum = 0;
  if(( index< Ydims[0]*Ydims[1])&&(i<10)&&(j<Ydims[1])){
  for (int k =0; k < Xdims[1]; k++ ){
    sum += X[i * Xdims[1] + k] * W[k * Wdims[1] + j];
  }  
  Y[i * Wdims[1] + j] = sum;
  }

}

//1st kernel will put convolution filters into matrix
//2nd kernel puts input feature maps into a matrix
//3rd kernel will do the multiplication


// From book chapter Figure 16.4
//X is input feature map
//W is convolution filter weights
//Y is output feature map
//W is stored as [p,q,c,m] = [conv height, conv width, channel, output feature map]
//Y is stored as [i,h,w,m]
//X is stored as [batches, height, width, channels ]
//Xdims[0] is # of input feature maps; Xdims[1] is height; Xdims[2] is width; Xdims[3] is channels
//Wdims[0] is height=5; Wdims[1] is width=5; Wdims[2] is channels; Wdims[3] is # of filters per output feature map
//Ydims[0] is # of X input feature maps; Ydims[1] is height after convo; Ydims[2] is width after convo; Ydims[3] is # of output feature maps
static void conv_forward_valid (const float *X, const int xdims[4],
                               const float *W, /*const bool conv1d*/ int wdims[4], const float *W2, int wdims2[4], float *Y,
                               const int ydims[4], const int ydims2[4]) {
    const auto filter_h   = KERNEL_WIDTH; //wdims[0]; //Height of convolution kernel = 5
    const auto filter_w   = KERNEL_WIDTH; //wdims[1]; //Width = 5
    //const auto in_channel = wdims[2]; //Channel
    
    //Putting wdims into constant memory. It can have only 2 possible values
    float *devX;
	float *devXremapped;
    float *devX_unrol;
    float *devW;
	float *devW_unroll;
    float *devY;
    int *devXdims;
	int *devWdims;
    int *devYdims;
    const int Wsize = sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH * wdims[2] * wdims[3];
    const int Xsize = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
    const int Xunrol_size = sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH *xdims[3] * xdims[0] * (xdims[1]-5+1)*(xdims[1]-5+1) ;
    const int Ysize = sizeof(float) * ydims[1] * ydims[2] * ydims[3] * ydims[0];
    printf("xdims[0]=%d, xdims[1]=%d, xdims[2]=%d, xdims[3]=%d\n",xdims[0],xdims[1],xdims[2],xdims[3]);
    printf("ydims[0]=%d, ydims[1]=%d, ydims[2]=%d, ydims[3]=%d\n",ydims[0],ydims[1],ydims[2],ydims[3]);
    printf("wdims[0]=%d, wdims[1]=%d, wdims[2]=%d, wdims[3]=%d\n",wdims[0],wdims[1],wdims[2],wdims[3]);
    printf("Wsize = %d\n", Wsize);
    printf("Xsize = %d\n", Xsize);
    printf("Xsize_unrol = %d\n", Xunrol_size);
    printf("Ysize = %d\n", Ysize);
    check_success(cudaMalloc(&devX, Xsize));
	check_success(cudaMalloc(&devXremapped, Xsize));
    check_success(cudaMalloc(&devX_unrol, Xunrol_size));
    check_success(cudaMalloc(&devW, Wsize));
	  check_success(cudaMalloc(&devW_unroll, Wsize));     // new added to change the way filters store in memory
    check_success(cudaMalloc(&devWdims, sizeof(int)*4));
    check_success(cudaMalloc(&devY, Ysize));
    check_success(cudaMalloc(&devXdims, sizeof(int)*4));
    check_success(cudaMalloc(&devYdims, sizeof(int)*4));

    check_success(cudaMemcpy(devX, X, Xsize, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devW, W, Wsize, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devXdims, &xdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devYdims, &ydims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devWdims, &wdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));



	
	
/*
	for(int w = 0; w < xdims[0]; w++)
		for(int x = 0; x < xdims[1]; x++)
			for(int y =0; y < xdims[2]; y++)
				for(int z =0; z <xdims[3]; z++){
					int index = w*xdims[1]*xdims[2]*xdims[3] + x*xdims[2]*xdims[3] + y*xdims[3] + z;
						X[index] = index;
					}
	check_success(cudaMemcpy(devX, X, Xsize, cudaMemcpyHostToDevice));
	dim3 z_unrolGrid(xdims[3], xdims[0], 1);  	// one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX<<<z_unrolGrid,z_unrolBlock>>>(devX, devXdims, devX_unrol);
    cudaDeviceSynchronize();
	float* zoe_X_unrolled = (float*) malloc(Xunrol_size);
	check_success(cudaMemcpy(zoe_X_unrolled, devX_unrol, Xunrol_size, cudaMemcpyDeviceToHost));
	
*/	

	
	
	
	dim3 ReMapBlock(32, 32, 1);
	int Remap_x_coal = ceil( xdims[3]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is channel 		  		//1
	int Remap_y_coal = ceil( xdims[2]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is column number			//1
	int Remap_z_coal = ceil( xdims[1]/1.0f);  // BlockIDx.z*blockDim.x+threadIdx.z is row number			//28
	dim3 ReMapGrid(Remap_x_coal, Remap_y_coal, Remap_z_coal);
	n_RemapX_memory_coalesced<<<ReMapGrid, ReMapBlock>>>(devX, devXdims, devXremapped);
//	float* Xremapped = (float*) malloc(Xsize);
/*	check_success(cudaMemcpy(Xremapped, devXremapped, Xsize, cudaMemcpyDeviceToHost));
	for(int w = 0; w < xdims[0]; w++)
		for(int x = 0; x < xdims[1]; x++)
			for(int y =0; y < xdims[2]; y++)
				for(int z =0; z <xdims[3]; z++){
					int index = w*xdims[1]*xdims[2]*xdims[3] + x*xdims[2]*xdims[3] + y*xdims[3] + z;
					if( (int) Xremapped[index] != 9)
						printf("w,x,y,z = %d,%d,%d,%d	index = %d    value is %d\n", w,x,y,z, index, (int) Xremapped[index] );
					}*/
	/*for(int w = 0; w < xdims[0]; w++)
		for(int z =0; z <xdims[3]; z++)
			for(int x = 0; x < xdims[1]; x++)
				for(int y =0; y < xdims[2]; y++){
					int outputindex = w*xdims[1]*xdims[2]*xdims[3] + z*xdims[1]*xdims[2] + x*xdims[2]+ y;
					int inputindex = w*xdims[1]*xdims[2]*xdims[3] + x*xdims[2]*xdims[3] + y*xdims[3] + z;
					if( (int) Xremapped[outputindex] != inputindex)
						printf("w,x,y,z = %d,%d,%d,%d	expectedindex = %d    value is %d\n", w,x,y,z, inputindex, (int) Xremapped[outputindex] );
					}*/
	
	
    //Unrolled table height is kernel^2*xdims[3] = 25 * no of channels
	//There will be batch_size (num_of_input_maps) number of tables 
    //Unrolled table Width is number of possible kernel locations = (xdims[1] - kernel+1)^2
    //Each Thread Block should process 1 feature map ie 1 channel
    //No of blocks in X is (1*24*24)/1024
    //No of blocks in Y is 1


/*	dim3 unrolBlock(1024, 1, 1);   //1024 threads in X direction
	int  unrol_x = ceil( xdims[3] * ydims[2]*ydims[1]/1024.0f ); //BlockIDx.x*blockDim.x+threadIdx.x is column number
	dim3 unrolGrid(unrol_x, 1, 1); //One block for each channel in each batch
    for(int batch = 0; batch < xdims[0]; batch++){
		n_unrolX_prof<<<unrolGrid,unrolBlock>>>(devXremapped, devXdims, batch, devX_unrol);
	}
	cudaDeviceSynchronize();*/


	/*float* X_unrolled = (float*) malloc(Xunrol_size);
	check_success(cudaMemcpy(X_unrolled, devX_unrol, Xunrol_size, cudaMemcpyDeviceToHost));
	for(int w = 0; w < xdims[0]; w++)
		for(int c = 0; c < xdims[3]; c++)
			for(int y = 0; y < 5*5; y++){
				for(int x = 0; x < ydims[1]*ydims[1]; x++){
					int inputindex = w*25*ydims[1]*ydims[1]*xdims[3] + c*25*ydims[1]*ydims[1] + y*ydims[1]*ydims[1] + x;
					//printf("%4d,", (int) X_unrolled[inputindex] );
					//if( X_unrolled[inputindex] != zoe_X_unrolled[inputindex])
					//	printf("%4d vs %4d ", (int) X_unrolled[inputindex], (int) zoe_X_unrolled[inputindex] );
		}
	}*/
    

/* //Z-unroll matrix kernel 
   //when we call this unroll kernel, we input all the x, including batch and channel, and output a whole unroll matrix, including batch, and channel
   //later when we do the matrix multiplication, we only have one big unroll matrix as matrix B.
    dim3 z_unrolGrid(xdims[3],xdims[0],1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX<<<z_unrolGrid,z_unrolBlock>>>(devX, devXdims, devX_unrol);
    cudaDeviceSynchronize();
	float* zoe_X_unrolled = (float*) malloc(Xunrol_size);
	check_success(cudaMemcpy(zoe_X_unrolled, devX_unrol, Xunrol_size, cudaMemcpyDeviceToHost));
	for(int w = 0; w < xdims[0]; w++)
		for(int y = 0; y < 5*5; y++){
			for(int x = 0; x < ydims[1]*ydims[2]; x++){
				int inputindex = w*25*ydims[1]*ydims[1] + y*ydims[1]*ydims[1] + x;
				if( X_unrolled[inputindex] != zoe_X_unrolled[inputindex])
					printf("%4d vs %4d ", (int) X_unrolled[inputindex], (int) zoe_X_unrolled[inputindex] );
		}
		printf("\n");
	}*/
  	//Z-unroll remapped kernel,same grid dims, only use devXremapped
    dim3 z_unrolGrid(xdims[3],xdims[0],1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX_remapped<<<z_unrolGrid,z_unrolBlock>>>(devXremapped, devXdims, devX_unrol);
    cudaDeviceSynchronize();
	
	
    //total thread number is 25 * wdims[2]*wdims[3], one thead copy one element from W into unrollw
    dim3 z_unrollwGrid(wdims[3],1,1);
    dim3 z_unrollwBlock(5,5,wdims[2]); // at most 25*32<1024, legal
    z_unrollw<<<z_unrollwGrid, z_unrollwBlock>>>(devW, devW_unroll, devWdims);


    
    //Use Tiled multiplication
    int numARows = ydims[3];
    int numAColumns = (KERNEL_WIDTH*KERNEL_WIDTH*xdims[3]); //Not copying weights again
    int numBRows = (KERNEL_WIDTH*KERNEL_WIDTH*xdims[3]); 	//Took out xdims[0]);   
    int numBColumns = ydims[1]*ydims[1];					//(xdims[1] - KERNEL_WIDTH+1)*(xdims[1] - KERNEL_WIDTH+1);
    int numCRows = ydims[3];
    int numCColumns = (xdims[1]-KERNEL_WIDTH+1)*(xdims[1]-KERNEL_WIDTH+1);
    int num_of_X_tiles = ceil(numBRows/32.0);
    //1 Thread processes 1 element in output table Y
    //Output columns = kernel^2 = 5*5
    //Output rows = number of output feature maps made for each digit = ydims[3]
    int x_blocks = ceil(numCColumns/32.0f);
    int y_blocks = ceil(numCRows/32.0f);
    dim3 z_DimGrid(x_blocks, y_blocks, 1);
    dim3 z_DimBlock(32, 32, 1);
    

    //copy devW_unroll back to cpu, then copy to constant memory.
    //when use this constant memory, only change multiply kernel
    if(Wsize<=(64*25*sizeof(float))){   //first convolution, and put all w into constant memory.
        float* cons_wunroll = (float*)malloc(Wsize);
        check_success(cudaMemcpy(cons_wunroll, devW_unroll, Wsize, cudaMemcpyDeviceToHost));
        check_success(cudaMemcpyToSymbol(dev_cons_Wunroll,cons_wunroll, Wsize));
        
        for(int batch =0; batch < ydims[0]; batch++){
            z_MatrixMultiply_perbatch_constant<<<z_DimGrid,z_DimBlock>>>(devX_unrol, devY,
                                                                         numARows, numAColumns,
                                                                         numBRows, numBColumns,
                                                                         numCRows, numCColumns,
                                                                         num_of_X_tiles, devYdims, devXdims,batch);
        }
    }
    else{//second convolution, only fisrt 29 rows of w canbe put into constant memory, the rest are still in the global memory
        for(int batch =0; batch < ydims[0]; batch++){
            z_MatrixMultiply_perbatch<<<z_DimGrid,z_DimBlock>>>(devW_unroll, devX_unrol, devY,
                                                                numARows, numAColumns,
                                                                numBRows, numBColumns,
                                                                numCRows, numCColumns,
                                                                num_of_X_tiles, devYdims, devXdims,batch);
        }
        
    }

    
  

	
	/*dim3 DimGrid(x_blocks, y_blocks, 1);
    dim3 DimBlock(32, 32, 1);
    if(conv1d)
        MatrixMultiply<<<DimGrid,DimBlock>>>(constantW, devX_unrol, devY,
                                             numARows, numAColumns,
                                             numBRows, numBColumns,
                                             numCRows, numCColumns,
                                             num_of_X_tiles);
    else
        //    A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns
        MatrixMultiply<<<DimGrid,DimBlock>>>(devW, devX_unrol, devY,
                                             numARows, numAColumns,
                                             numBRows, numBColumns,
                                             numCRows, numCColumns,
                                             num_of_X_tiles);
											 */


//    device_query();
//    conv_forward_kernel<<<DimGrid,DimBlock>>>(devX,devXdims,devW,conv1d, devY,devYdims);

    
/* Original Serial Code
 
 //Total number of W filters is total input feature maps * total output feature maps
  for (int i =0; i < ydims[0]; i++ ) { //Number of input feature maps X
    for (int m =0; m < ydims[3]; m++ ) { //Number of output feature maps Y made for each digit
      for (int h =0; h < ydims[1]; h++ ) { //Height of output Y (=Input height - conv kernel height +1)
        for (int w =0; w < ydims[2]; w++ ) { //Width of ouptut Y (=Input width - conv kernel width +1)
          for (int p =0; p < filter_h; p++) { //Height of weight convolution kernel W
            for (int q =0; q < filter_w; q++) { //Width of weigth convolution kernel W
              for (int c =0; c < in_channel; c++) { //Each channel of W
                const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                     (h + p) * xdims[2] * xdims[3] +
                                     (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                                     q * wdims[2] * wdims[3] + c * wdims[3] + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
*/
    cudaDeviceSynchronize();
    check_success(cudaFree(devX));
    check_success(cudaFree(devX_unrol));
    check_success(cudaFree(devXremapped));
    //check_success(cudaFree(devXdims));
    //check_success(cudaFree(devYdims));
/*
    check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));

   
    for(int c = 0; c < 1; c++){
        for(int y = 0; y < ydims[2]; y++){
            for(int x = 0; x < ydims[1]; x++){
                int index =  c * ydims[1] * ydims[2] + y * ydims[1] + x;
                printf("%f at location x,y,c = %d,%d,%d\n", Y[index],x,y,c );
            }
        }
    }*/

    const int pool_size = 2;
    //Updating old xdims and ydims, so they can be sent to average pool
    const int adims[] = {ydims[0], ydims[1], ydims[2], ydims[3] };
    const int bdims[] = {ydims[0], ydims[1]/pool_size, ydims[2]/pool_size, ydims[3] };
    //xdims[0] = ydims[0]; xdims[1] = ydims[1]; xdims[2] = ydims[2]; xdims[3] = ydims[3];
    //ydims[1] = ydims[1]/pool_size; ydims[2] = ydims[2]/pool_size; //No change to ydims[0] and ydims[3]
    const int Asize = adims[0]*adims[1]*adims[2]*adims[3]*sizeof(float);
    const int Bsize = bdims[0]*bdims[1]*bdims[2]*bdims[3]*sizeof(float);
    float* devB_after_pool;
    int* devBdims;
    check_success(cudaMalloc(&devB_after_pool,  Bsize));  //allocate space for afterpoolY
    check_success(cudaMalloc(&devBdims, sizeof(int)*4));
    check_success(cudaMemcpy(devBdims, &bdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    dim3 avgPoolBlock(32,32,1);
    int avgpool_x_coal = ceil( bdims[2]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is column number   //1
    int avgpool_y_coal = ceil( bdims[1]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is row number    //1
    int avgpool_z_coal = ceil( bdims[3]/1.0f);  // BlockIDx.z*blockDim.z+threadIdx.z is channel   //32 then 64
    dim3 avgPoolGrid(avgpool_x_coal, avgpool_y_coal, avgpool_z_coal);
    for(int batch =0; batch < ydims[0]; batch++){
        n_averagepool_output_remapped<<<avgPoolGrid,avgPoolBlock>>>(devY, devYdims, pool_size,
                                                        devB_after_pool, devBdims, batch);
    }
    cudaDeviceSynchronize();
    //check_success(cudaFree(devWdims));
    //Removed to test averagepool check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));
   
    //check_success(cudaMemcpy(Y, devB_after_pool, Bsize, cudaMemcpyDeviceToHost));
   /* for(int c = 0; c < 1; c++){
        for(int y = 0; y < bdims[1]; y++){
            for(int x = 0; x < bdims[2]; x++){
                int index =  c * bdims[1] * bdims[2] + y * bdims[1] + x;
                printf("%f at location x,y,c = %d,%d,%d\n", Y[index],x,y,c );
            }
        }
    }
*/
    //Now start the second convolution, bdims now is used as xdims, wdims2 is wdims, 
    //devB_after_pool used as devX, fisrt no need to remap X again
    //unroll X,
    
    //ydims2[] is set during function call
    float *devW2;
    int *devWdims2;
    int *devYdims2;
    float *devY2;
    float *devX_unrol2;
    float *devW_unroll2;
    int Wsize2 = sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH * wdims2[2] * wdims2[3];
    int Ysize2 = sizeof(float) * ydims2[1] * ydims2[2] * ydims2[3] * ydims2[0];
    int Xunrol_size2 = sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH *bdims[3] * bdims[0] * (bdims[1]-5+1)*(bdims[1]-5+1) ;
    check_success(cudaMalloc(&devW_unroll2, Wsize2));
    check_success(cudaMalloc(&devX_unrol2, Xunrol_size2));
    check_success(cudaMalloc(&devWdims2, sizeof(int)*4));
    check_success(cudaMalloc(&devW2, Wsize2));
    check_success(cudaMalloc(&devYdims2, sizeof(int)*4));
    check_success(cudaMalloc(&devY2, Ysize2));
    check_success(cudaMemcpy(devWdims2, &wdims2[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devW2, W2, Wsize2, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devYdims2, &ydims2[0], sizeof(int)*4, cudaMemcpyHostToDevice));

    
    dim3 z_unrolGrid2(bdims[3],bdims[0],1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock2(bdims[1], bdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX_remapped<<<z_unrolGrid2,z_unrolBlock2>>>(devB_after_pool, devBdims, devX_unrol2);
    cudaDeviceSynchronize();

    dim3 z_unrollwGrid2(wdims2[3],1,1);
    dim3 z_unrollwBlock2(5,5,wdims2[2]); // at most 25*32<1024, legal
    z_unrollw<<<z_unrollwGrid2, z_unrollwBlock2>>>(devW2, devW_unroll2, devWdims2);

    numARows = ydims2[3];
    numAColumns = (KERNEL_WIDTH*KERNEL_WIDTH*bdims[3]); //Not copying weights again
    numBRows = (KERNEL_WIDTH*KERNEL_WIDTH*bdims[3]);  //Took out xdims[0]);   
    numBColumns = ydims2[1]*ydims2[1];          //(xdims[1] - KERNEL_WIDTH+1)*(xdims[1] - KERNEL_WIDTH+1);
    numCRows = ydims2[3];
    numCColumns = (bdims[1]-KERNEL_WIDTH+1)*(bdims[1]-KERNEL_WIDTH+1);
    num_of_X_tiles = ceil(numBRows/32.0);
    //1 Thread processes 1 element in output table Y
    //Output columns = kernel^2 = 5*5
    //Output rows = number of output feature maps made for each digit = ydims[3]
    x_blocks = ceil(numCColumns/32.0f);
    y_blocks = ceil(numCRows/32.0f);
    dim3 z_DimGrid2(x_blocks, y_blocks, 1);
    dim3 z_DimBlock2(32, 32, 1);
    

    //copy devW_unroll back to cpu, then copy to constant memory.
    //when use this constant memory, only change multiply kernel
    if(Wsize2<=(64*25*sizeof(float))){   //first convolution, and put all w into constant memory.
        float* cons_wunroll = (float*)malloc(Wsize);
        check_success(cudaMemcpy(cons_wunroll, devW_unroll, Wsize, cudaMemcpyDeviceToHost));
        check_success(cudaMemcpyToSymbol(dev_cons_Wunroll,cons_wunroll, Wsize));
        
        for(int batch =0; batch < ydims[0]; batch++){
            z_MatrixMultiply_perbatch_constant<<<z_DimGrid,z_DimBlock>>>(devX_unrol, devY,
                                                                         numARows, numAColumns,
                                                                         numBRows, numBColumns,
                                                                         numCRows, numCColumns,
                                                                         num_of_X_tiles, devYdims, devXdims,batch);
        }
    }
    else{//second convolution, only fisrt 29 rows of w canbe put into constant memory, the rest are still in the global memory
        for(int batch =0; batch < ydims2[0]; batch++){
            z_MatrixMultiply_perbatch<<<z_DimGrid2,z_DimBlock2>>>(devW_unroll2, devX_unrol2, devY2,
                                                                  numARows, numAColumns,
                                                                  numBRows, numBColumns,
                                                                  numCRows, numCColumns,
                                                                  num_of_X_tiles, devYdims2, devBdims,batch);
        }
        
    }

    cudaDeviceSynchronize();

    
    //Updating old xdims and ydims, so they can be sent to average pool
   // const int adims[] = {ydims[0], ydims[1], ydims[2], ydims[3] };
    const int bdims2[] = {ydims2[0], ydims2[1]/pool_size, ydims2[2]/pool_size, ydims2[3] };
    
   // const int Asize = adims[0]*adims[1]*adims[2]*adims[3]*sizeof(float);
    const int Bsize2 = bdims2[0]*bdims2[1]*bdims2[2]*bdims2[3]*sizeof(float);
    float* devB_after_pool2;
    int* devBdims2;
    check_success(cudaMalloc(&devB_after_pool2,  Bsize2));  //allocate space for afterpoolY
    check_success(cudaMalloc(&devBdims2, sizeof(int)*4));
    check_success(cudaMemcpy(devBdims2, &bdims2[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    dim3 avgPoolBlock2(32,32,1);
    avgpool_x_coal = ceil( bdims2[2]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is column number   //1
    avgpool_y_coal = ceil( bdims2[1]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is row number    //1
    avgpool_z_coal = ceil( bdims2[3]/1.0f);  // BlockIDx.z*blockDim.z+threadIdx.z is channel   //32 then 64
    dim3 avgPoolGrid2(avgpool_x_coal, avgpool_y_coal, avgpool_z_coal);
    for(int batch =0; batch < ydims2[0]; batch++){
        n_averagepool_coalesced<<<avgPoolGrid2,avgPoolBlock2>>>(devY2, devYdims2, pool_size,
                                                        devB_after_pool2, devBdims2, batch);
    }
    cudaDeviceSynchronize();






    //cudaDeviceReset();
    //printf("5555");
  /*  check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));

   
    for(int c = 0; c < 1; c++){
        for(int y = 0; y < ydims[2]; y++){
            for(int x = 0; x < ydims[1]; x++){
                int index =  c * ydims[1] * ydims[2] + y * ydims[1] + x;
                printf("%f at location x,y,c = %d,%d,%d\n", Y[index],x,y,c );
            }
        }
    }  */

    check_success(cudaMemcpy(Y, devB_after_pool2, Bsize2, cudaMemcpyDeviceToHost));
    check_success(cudaFree(devX));
    //printf("3333");
    check_success(cudaFree(devW));
    check_success(cudaFree(devY));
    check_success(cudaFree(devXdims));
    check_success(cudaFree(devYdims));
    check_success(cudaFree(devWdims));
}
static void fully_forward_kernel(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]){
  float *devX;
  float *devW;
  float *devY;
  int *devXdims;
  int *devWdims;
  int *devYdims;
  int Xsize = xdims[0]*xdims[1]*sizeof(float);
  int Wsize = wdims[0]*wdims[1]*sizeof(float);
  int Ysize = ydims[0]*ydims[1]*sizeof(float);
  check_success(cudaMalloc(&devX, Xsize));
  check_success(cudaMalloc(&devY, Ysize));
  check_success(cudaMalloc(&devW, Wsize));
  
  check_success(cudaMalloc(&devYdims, sizeof(int)*2));
  check_success(cudaMalloc(&devWdims, sizeof(int)*2));

  cudaMalloc(&devXdims, sizeof(int)*2);

  check_success(cudaMemcpy(devX, X, Xsize, cudaMemcpyHostToDevice));
  check_success(cudaMemcpy(devW, W, Wsize, cudaMemcpyHostToDevice));
  check_success(cudaMemcpy(devXdims, &xdims[0], sizeof(int)*2, cudaMemcpyHostToDevice));
  check_success(cudaMemcpy(devWdims, &wdims[0], sizeof(int)*2, cudaMemcpyHostToDevice));
  check_success(cudaMemcpy(devYdims, &ydims[0], sizeof(int)*2, cudaMemcpyHostToDevice));
for(int i =0;i< 100;i++){
  printf("%f  ", X[i]);
}
  dim3 fullyforwardGrid(ceil(ydims[0]*ydims[1]/1024.0),1,1); //i =(threadIdx.x + blockIdx.x * blockDim.x)/wdims[1];
  dim3 fullyforwardBlock(1024,1,1); //j = (threadIdx.x + blockIdx.x * blockDim.x)%wdims[1];
  fullyforward<<<fullyforwardGrid, fullyforwardBlock>>>(devX, devXdims, devW, devWdims, devY, devYdims);

  check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));
  for(int i = 0; i< ydims[0];i++){
    for ( int j =0; j< ydims[1];j++){
      printf("%f(%d) ", Y[i*ydims[1]+j], i*ydims[1]+j);
    }
    printf("\n");
  }
  check_success(cudaFree(devX));
  check_success(cudaFree(devW));
  check_success(cudaFree(devY));
  check_success(cudaFree(devXdims));
  check_success(cudaFree(devYdims));
  check_success(cudaFree(devWdims));


}

static void conv_forward_valid_no_use(const float *X, const int xdims[4],
                               const float *W, int wdims[4], float *Y,
                               const int ydims[4]) {
    const auto filter_h   = KERNEL_WIDTH; //wdims[0]; //Height of convolution kernel = 5
    const auto filter_w   = KERNEL_WIDTH; //wdims[1]; //Width = 5
    //const auto in_channel = wdims[2]; //Channel
    
    //Putting wdims into constant memory. It can have only 2 possible values
    float *devX;
  float *devXremapped;
    float *devX_unrol;
    float *devW;
  float *devW_unroll;
    float *devY;
    int *devXdims;
  int *devWdims;
    int *devYdims;
    int Wsize = sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH * wdims[2] * wdims[3];
    int Xsize = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
    int Xunrol_size = sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH *xdims[3] * xdims[0] * (xdims[1]-5+1)*(xdims[1]-5+1) ;
    int Ysize = sizeof(float) * ydims[1] * ydims[2] * ydims[3] * ydims[0];
    printf("ydims[0]=%d, ydims[1]=%d, ydims[2]=%d, ydims[3]=%d\n",ydims[0],ydims[1],ydims[2],ydims[3]);
    printf("xdims[0]=%d, xdims[1]=%d, xdims[2]=%d, xdims[3]=%d\n",xdims[0],xdims[1],xdims[2],xdims[3]);
    printf("wdims[0]=%d, wdims[1]=%d, wdims[2]=%d, wdims[3]=%d\n",wdims[0],wdims[1],wdims[2],wdims[3]);
    printf("Wsize = %d\n", Wsize);
    printf("Xsize = %d\n", Xsize);
    printf("Xsize_unrol = %d\n", Xunrol_size);
    printf("Ysize = %d\n", Ysize);
    check_success(cudaMalloc(&devX, Xsize));
  check_success(cudaMalloc(&devXremapped, Xsize));
    check_success(cudaMalloc(&devX_unrol, Xunrol_size));
    check_success(cudaMalloc(&devW, Wsize));
  check_success(cudaMalloc(&devW_unroll, Wsize));     // new added to change the way filters store in memory
    check_success(cudaMalloc(&devWdims, sizeof(int)*4));
    check_success(cudaMalloc(&devY, Ysize));
    check_success(cudaMalloc(&devXdims, sizeof(int)*4));
    //check_success(cudaMalloc(&devWdims, sizeof(int)*4));
    check_success(cudaMalloc(&devYdims, sizeof(int)*4));
    check_success(cudaMemcpy(devX, X, Xsize, cudaMemcpyHostToDevice));
  
    //if(conv1d)
    //    check_success(cudaMemcpyToSymbol(constantW, W, Wsize,0, cudaMemcpyHostToDevice));
    //else
        check_success(cudaMemcpy(devW, W, Wsize, cudaMemcpyHostToDevice));
    //printf("2222");
    check_success(cudaMemcpy(devXdims, &xdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    //check_success(cudaMemcpyToSymbol(devWdims, &wdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devYdims, &ydims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(devWdims, &wdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));

  
  
  
/*
  for(int w = 0; w < xdims[0]; w++)
    for(int x = 0; x < xdims[1]; x++)
      for(int y =0; y < xdims[2]; y++)
        for(int z =0; z <xdims[3]; z++){
          int index = w*xdims[1]*xdims[2]*xdims[3] + x*xdims[2]*xdims[3] + y*xdims[3] + z;
            X[index] = index;
          }
  check_success(cudaMemcpy(devX, X, Xsize, cudaMemcpyHostToDevice));
  dim3 z_unrolGrid(xdims[3], xdims[0], 1);    // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX<<<z_unrolGrid,z_unrolBlock>>>(devX, devXdims, devX_unrol);
    cudaDeviceSynchronize();
  float* zoe_X_unrolled = (float*) malloc(Xunrol_size);
  check_success(cudaMemcpy(zoe_X_unrolled, devX_unrol, Xunrol_size, cudaMemcpyDeviceToHost));
  
*/  


  
  
  dim3 ReMapBlock(32, 32, 1);
  int Remap_x_coal = ceil( xdims[3]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is channel           //1
  int Remap_y_coal = ceil( xdims[2]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is column number     //1
  int Remap_z_coal = ceil( xdims[1]/1.0f);  // BlockIDx.z*blockDim.x+threadIdx.z is row number      //28
  dim3 ReMapGrid(Remap_x_coal, Remap_y_coal, Remap_z_coal);
  n_RemapX_memory_coalesced<<<ReMapGrid, ReMapBlock>>>(devX, devXdims, devXremapped);
  float* Xremapped = (float*) malloc(Xsize);
/*  check_success(cudaMemcpy(Xremapped, devXremapped, Xsize, cudaMemcpyDeviceToHost));
  for(int w = 0; w < xdims[0]; w++)
    for(int x = 0; x < xdims[1]; x++)
      for(int y =0; y < xdims[2]; y++)
        for(int z =0; z <xdims[3]; z++){
          int index = w*xdims[1]*xdims[2]*xdims[3] + x*xdims[2]*xdims[3] + y*xdims[3] + z;
          if( (int) Xremapped[index] != 9)
            printf("w,x,y,z = %d,%d,%d,%d index = %d    value is %d\n", w,x,y,z, index, (int) Xremapped[index] );
          }*/
  /*for(int w = 0; w < xdims[0]; w++)
    for(int z =0; z <xdims[3]; z++)
      for(int x = 0; x < xdims[1]; x++)
        for(int y =0; y < xdims[2]; y++){
          int outputindex = w*xdims[1]*xdims[2]*xdims[3] + z*xdims[1]*xdims[2] + x*xdims[2]+ y;
          int inputindex = w*xdims[1]*xdims[2]*xdims[3] + x*xdims[2]*xdims[3] + y*xdims[3] + z;
          if( (int) Xremapped[outputindex] != inputindex)
            printf("w,x,y,z = %d,%d,%d,%d expectedindex = %d    value is %d\n", w,x,y,z, inputindex, (int) Xremapped[outputindex] );
          }*/
  
  
    //Unrolled table height is kernel^2*xdims[3] = 25 * no of channels
  //There will be batch_size (num_of_input_maps) number of tables 
    //Unrolled table Width is number of possible kernel locations = (xdims[1] - kernel+1)^2
    //Each Thread Block should process 1 feature map ie 1 channel
    //No of blocks in X is (1*24*24)/1024
    //No of blocks in Y is 1


/*  dim3 unrolBlock(1024, 1, 1);   //1024 threads in X direction
  int  unrol_x = ceil( xdims[3] * ydims[2]*ydims[1]/1024.0f ); //BlockIDx.x*blockDim.x+threadIdx.x is column number
  dim3 unrolGrid(unrol_x, 1, 1); //One block for each channel in each batch
    for(int batch = 0; batch < xdims[0]; batch++){
    n_unrolX_prof<<<unrolGrid,unrolBlock>>>(devXremapped, devXdims, batch, devX_unrol);
  }
  cudaDeviceSynchronize();*/


  /*float* X_unrolled = (float*) malloc(Xunrol_size);
  check_success(cudaMemcpy(X_unrolled, devX_unrol, Xunrol_size, cudaMemcpyDeviceToHost));
  for(int w = 0; w < xdims[0]; w++)
    for(int c = 0; c < xdims[3]; c++)
      for(int y = 0; y < 5*5; y++){
        for(int x = 0; x < ydims[1]*ydims[1]; x++){
          int inputindex = w*25*ydims[1]*ydims[1]*xdims[3] + c*25*ydims[1]*ydims[1] + y*ydims[1]*ydims[1] + x;
          //printf("%4d,", (int) X_unrolled[inputindex] );
          //if( X_unrolled[inputindex] != zoe_X_unrolled[inputindex])
          //  printf("%4d vs %4d ", (int) X_unrolled[inputindex], (int) zoe_X_unrolled[inputindex] );
    }
  }*/
    

/* //Z-unroll matrix kernel 
   //when we call this unroll kernel, we input all the x, including batch and channel, and output a whole unroll matrix, including batch, and channel
   //later when we do the matrix multiplication, we only have one big unroll matrix as matrix B.
    dim3 z_unrolGrid(xdims[3],xdims[0],1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX<<<z_unrolGrid,z_unrolBlock>>>(devX, devXdims, devX_unrol);
    cudaDeviceSynchronize();
  float* zoe_X_unrolled = (float*) malloc(Xunrol_size);
  check_success(cudaMemcpy(zoe_X_unrolled, devX_unrol, Xunrol_size, cudaMemcpyDeviceToHost));
  for(int w = 0; w < xdims[0]; w++)
    for(int y = 0; y < 5*5; y++){
      for(int x = 0; x < ydims[1]*ydims[2]; x++){
        int inputindex = w*25*ydims[1]*ydims[1] + y*ydims[1]*ydims[1] + x;
        if( X_unrolled[inputindex] != zoe_X_unrolled[inputindex])
          printf("%4d vs %4d ", (int) X_unrolled[inputindex], (int) zoe_X_unrolled[inputindex] );
    }
    printf("\n");
  }*/
    //Z-unroll remapped kernel,same grid dims, only use devXremapped
  dim3 z_unrolGrid(xdims[3],xdims[0],1);  // one block for only an input map, xdims[3]= blockIdx.x means input map no. for one batch, blockIdx.y is batch size
    dim3 z_unrolBlock(xdims[1], xdims[2], 1);   // 2d block, x & y is input width and height, 
    z_unrolX_remapped<<<z_unrolGrid,z_unrolBlock>>>(devXremapped, devXdims, devX_unrol);
    cudaDeviceSynchronize();
  
  
    //total thread number is 25 * wdims[2]*wdims[3], one thead copy one element from W into unrollw
    dim3 z_unrollwGrid(wdims[3],1,1);
    dim3 z_unrollwBlock(5,5,wdims[2]); // at most 25*32<1024, legal
    z_unrollw<<<z_unrollwGrid, z_unrollwBlock>>>(devW, devW_unroll, devWdims);
    



    //Use Tiled multiplication
    int numARows = ydims[3];
    int numAColumns = (KERNEL_WIDTH*KERNEL_WIDTH*xdims[3]); //Not copying weights again
    int numBRows = (KERNEL_WIDTH*KERNEL_WIDTH*xdims[3]);  //Took out xdims[0]);   
    int numBColumns = ydims[1]*ydims[1];          //(xdims[1] - KERNEL_WIDTH+1)*(xdims[1] - KERNEL_WIDTH+1);
    int numCRows = ydims[3];
    int numCColumns = (xdims[1]-KERNEL_WIDTH+1)*(xdims[1]-KERNEL_WIDTH+1);
    int num_of_X_tiles = ceil(numBRows/32.0);
    //1 Thread processes 1 element in output table Y
    //Output columns = kernel^2 = 5*5
    //Output rows = number of output feature maps made for each digit = ydims[3]
    int x_blocks = ceil(numCColumns/32.0f);
    int y_blocks = ceil(numCRows/32.0f);
    dim3 z_DimGrid(x_blocks, y_blocks, 1);
    dim3 z_DimBlock(32, 32, 1);
    

    //copy devW_unroll back to cpu, then copy to constant memory.
    //when use this constant memory, only change multiply kernel
     if(Wsize<=(64*25*sizeof(float))){   //first convolution, and put all w into constant memory. 
    float* cons_wunroll = (float*)malloc(Wsize);
    check_success(cudaMemcpy(cons_wunroll, devW_unroll, Wsize, cudaMemcpyDeviceToHost));
    check_success(cudaMemcpyToSymbol(dev_cons_Wunroll,cons_wunroll, Wsize));
    
    for(int batch =0; batch < ydims[0]; batch++){
      z_MatrixMultiply_perbatch_constant<<<z_DimGrid,z_DimBlock>>>(devX_unrol, devY,
                                             numARows, numAColumns,
                                             numBRows, numBColumns,
                                             numCRows, numCColumns,
                                             num_of_X_tiles, devYdims, devXdims,batch);
    }
  }
  else{//second convolution, only fisrt 29 rows of w canbe put into constant memory, the rest are still in the global memory 
    for(int batch =0; batch < ydims[0]; batch++){
      z_MatrixMultiply_perbatch<<<z_DimGrid,z_DimBlock>>>(devW_unroll, devX_unrol, devY,
                                             numARows, numAColumns,
                                             numBRows, numBColumns,
                                             numCRows, numCColumns,
                                             num_of_X_tiles, devYdims, devXdims,batch);
    }

  }

    
  

  
  /*dim3 DimGrid(x_blocks, y_blocks, 1);
    dim3 DimBlock(32, 32, 1);
    if(conv1d)
        MatrixMultiply<<<DimGrid,DimBlock>>>(constantW, devX_unrol, devY,
                                             numARows, numAColumns,
                                             numBRows, numBColumns,
                                             numCRows, numCColumns,
                                             num_of_X_tiles);
    else
        //    A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns
        MatrixMultiply<<<DimGrid,DimBlock>>>(devW, devX_unrol, devY,
                                             numARows, numAColumns,
                                             numBRows, numBColumns,
                                             numCRows, numCColumns,
                                             num_of_X_tiles);
                       */


//    device_query();
//    conv_forward_kernel<<<DimGrid,DimBlock>>>(devX,devXdims,devW,conv1d, devY,devYdims);

    
/* Original Serial Code
 
 //Total number of W filters is total input feature maps * total output feature maps
  for (int i =0; i < ydims[0]; i++ ) { //Number of input feature maps X
    for (int m =0; m < ydims[3]; m++ ) { //Number of output feature maps Y made for each digit
      for (int h =0; h < ydims[1]; h++ ) { //Height of output Y (=Input height - conv kernel height +1)
        for (int w =0; w < ydims[2]; w++ ) { //Width of ouptut Y (=Input width - conv kernel width +1)
          for (int p =0; p < filter_h; p++) { //Height of weight convolution kernel W
            for (int q =0; q < filter_w; q++) { //Width of weigth convolution kernel W
              for (int c =0; c < in_channel; c++) { //Each channel of W
                const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                     (h + p) * xdims[2] * xdims[3] +
                                     (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                                     q * wdims[2] * wdims[3] + c * wdims[3] + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
*/
    cudaDeviceSynchronize();
/*
    check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));

   
    for(int c = 0; c < 1; c++){
        for(int y = 0; y < ydims[2]; y++){
            for(int x = 0; x < ydims[1]; x++){
                int index =  c * ydims[1] * ydims[2] + y * ydims[1] + x;
                printf("%f at location x,y,c = %d,%d,%d\n", Y[index],x,y,c );
            }
        }
    }*/

    const int pool_size = 2;
    //Updating old xdims and ydims, so they can be sent to average pool
    const int adims[] = {ydims[0], ydims[1], ydims[2], ydims[3] };
    const int bdims[] = {ydims[0], ydims[1]/pool_size, ydims[2]/pool_size, ydims[3] };
    //xdims[0] = ydims[0]; xdims[1] = ydims[1]; xdims[2] = ydims[2]; xdims[3] = ydims[3];
    //ydims[1] = ydims[1]/pool_size; ydims[2] = ydims[2]/pool_size; //No change to ydims[0] and ydims[3]
    const int Asize = adims[0]*adims[1]*adims[2]*adims[3]*sizeof(float);
    const int Bsize = bdims[0]*bdims[1]*bdims[2]*bdims[3]*sizeof(float);
    float* devB_after_pool;
    int* devBdims;
    check_success(cudaMalloc(&devB_after_pool,  Bsize));  //allocate space for afterpoolY
    check_success(cudaMalloc(&devBdims, sizeof(int)*4));
    check_success(cudaMemcpy(devBdims, &bdims[0], sizeof(int)*4, cudaMemcpyHostToDevice));
    //Printinf adims and bdims to ensure they are correct before avg pool
    printf("**************************VALUES just before AVG POOL *****************\n");
    printf("adims[0]=%d, adims[1]=%d, adims[2]=%d, adims[3]=%d\n",adims[0],adims[1],adims[2],adims[3]);
    printf("bdims[0]=%d, bdims[1]=%d, bdims[2]=%d, bdims[3]=%d\n",bdims[0],bdims[1],bdims[2],bdims[3]);
    dim3 avgPoolBlock(32,32,1);
    int avgpool_x_coal = ceil( bdims[2]/32.0f); // BlockIDx.x*blockDim.x+threadIdx.x is column number 	//1
    int avgpool_y_coal = ceil( bdims[1]/32.0f); // BlockIDx.y*blockDim.y+threadIdx.y is row number		//1
    int avgpool_z_coal = ceil( bdims[3]/1.0f);  // BlockIDx.z*blockDim.z+threadIdx.z is channel   //32 then 64
    dim3 avgPoolGrid(avgpool_x_coal, avgpool_y_coal, avgpool_z_coal);
    for(int batch =0; batch < ydims[0]; batch++){
        n_averagepool_coalesced<<<avgPoolGrid,avgPoolBlock>>>(devY, devYdims, pool_size,
                                                        devB_after_pool, devBdims, batch);
    }
    cudaDeviceSynchronize();
    //Removed to test averagepool check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));
   
    check_success(cudaMemcpy(Y, devB_after_pool, Bsize, cudaMemcpyDeviceToHost));
   /* for(int c = 0; c < 1; c++){
        for(int y = 0; y < bdims[1]; y++){
            for(int x = 0; x < bdims[2]; x++){
                int index =  c * bdims[1] * bdims[2] + y * bdims[1] + x;
                printf("%f at location x,y,c = %d,%d,%d\n", Y[index],x,y,c );
            }
        }
    }
*/




    //cudaDeviceReset();
    //printf("5555");
  /*  check_success(cudaMemcpy(Y, devY, Ysize, cudaMemcpyDeviceToHost));

   
    for(int c = 0; c < 1; c++){
        for(int y = 0; y < ydims[2]; y++){
            for(int x = 0; x < ydims[1]; x++){
                int index =  c * ydims[1] * ydims[2] + y * ydims[1] + x;
                printf("%f at location x,y,c = %d,%d,%d\n", Y[index],x,y,c );
            }
        }
    }  */

    check_success(cudaFree(devX));
    //printf("3333");
    check_success(cudaFree(devW));
    check_success(cudaFree(devY));
    check_success(cudaFree(devXdims));
    check_success(cudaFree(devYdims));
    //check_success(cudaFree(devWdmins));
}

/*
// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}*/

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
/*static void average_pool(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]) {
  for (const auto i : range(0, ydims[0])) { //Number of batches of output feature maps Y
    for (const auto m : range(0, ydims[3])) { //Each channel of Y
      for (const auto w : range(0, ydims[2])) { //Width of output Y
        for (const auto h : range(0, ydims[1])) { //Height of output Y
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                   (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}*/

static void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  for (const auto i : range(0, xdims[0])) {
    for (const auto j : range(0, wdims[1])) {
      float sum = 0;
      for (const auto k : range(0, xdims[1])) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  check_success(cudaProfilerStart());
  // conv layer
  //                      10        28-5+1
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
    
    {   const auto tic = now();
        //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
        const auto toc = now();
        const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();
        std::cout << "Calling conv_forward_valid1 took " << elapsed << "milliseconds\n";
    }

  /// relu layer
  //relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
    
    {   const auto tic = now();
        //average_pool(a, adims, pool_size, b, bdims);
        const auto toc = now();
        const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();
        std::cout << "Calling average_pool took " << elapsed << "milliseconds\n";
    }

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
        {   const auto tic = now();
               conv_forward_valid (x,xdims,conv1,conv1dims,conv2, conv2dims,c,adims, cdims );
           // conv_forward_valid(a, bdims, conv2, conv2dims, c, cdims);
            const auto toc = now();
            const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();
            std::cout << "Calling conv_forward_valid2 took " << elapsed << "milliseconds\n";
        }

  // relu
  //relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  //average_pool(c, cdims, pool_size, d, ddims);

  // conv_forward_valid_2 (x,xdims,conv1,conv1dims,conv2, conv2dims,c,adims, cdims );
  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward_kernel(c, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward_kernel(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  check_success(cudaProfilerStop()); // To flush profiling data out
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);


  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";
    
  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
