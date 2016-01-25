
extern "C" __global__
void histgramMakerKernel_GlobalMemAtomics(int *d_histgram,int *d_partialHistgrams,
							 	          const unsigned char *d_text, int textLength) {
	
	int *d_myHistgram = &d_partialHistgrams[blockIdx.x * 256];

	int stride = gridDim.x * blockDim.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int pos = gid; pos < textLength; pos += stride) {
    	int ch = d_text[pos];
    	atomicAdd(&d_myHistgram[ch], 1);
    }
    __syncthreads();
	for (int histPos = threadIdx.x; histPos < 256; histPos += blockDim.x)
		atomicAdd(&d_histgram[histPos], d_myHistgram[histPos]);
}
