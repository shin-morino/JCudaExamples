
extern "C" __global__
void histgramMakerKernel_SharedMemAtomics(int *d_histgram, 
							 			  const uchar4* d_text4, int textLength4) {
	__shared__ int sh_histgram[256];
	for (int shPos = threadIdx.x; shPos < 256; shPos += blockDim.x)
		sh_histgram[shPos] = 0;
	__syncthreads();

	int stride = gridDim.x * blockDim.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int pos = gid; pos < textLength4; pos += stride) {
    	uchar4 ch4 = d_text4[pos];
    	atomicAdd(&sh_histgram[ch4.x], 1);
    	atomicAdd(&sh_histgram[ch4.y], 1);
    	atomicAdd(&sh_histgram[ch4.z], 1);
    	atomicAdd(&sh_histgram[ch4.w], 1);
    }
    __syncthreads();
	for (int histPos = threadIdx.x; histPos < 256; histPos += blockDim.x)
		atomicAdd(&d_histgram[histPos], sh_histgram[histPos]);
}
