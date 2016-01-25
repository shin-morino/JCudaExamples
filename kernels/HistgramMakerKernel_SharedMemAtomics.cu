
extern "C" __global__
void histgramMakerKernel_SharedMemAtomics(int *d_histgram,
							 	          const unsigned char *d_text, int textLength) {
	
	__shared__ int sh_histgram[256];
	for (int histPos = threadIdx.x; histPos < 256; histPos += blockDim.x)
		sh_histgram[histPos] = 0;
	__syncthreads();
	
	int stride = gridDim.x * blockDim.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int pos = gid; pos < textLength; pos += stride) {
    	int ch = d_text[pos];
    	atomicAdd(&sh_histgram[ch], 1);
    }
    __syncthreads();
	for (int histPos = threadIdx.x; histPos < 256; histPos += blockDim.x)
		atomicAdd(&d_histgram[histPos], sh_histgram[histPos]);
}
