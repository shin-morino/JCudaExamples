
extern "C" __global__
void histgramMakerKernel_GlobalMemAtomics2(int *d_histgram,int *d_partialHistgrams,
							 	           const uchar4 *d_text4, int textLength4) {
	
	int *d_myHistgram = &d_partialHistgrams[blockIdx.x * 256];

	int stride = gridDim.x * blockDim.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int pos = gid; pos < textLength4; pos += stride) {
    	uchar4 ch4 = d_text4[pos];
    	atomicAdd(&d_myHistgram[ch4.x], 1);
    	atomicAdd(&d_myHistgram[ch4.y], 1);
    	atomicAdd(&d_myHistgram[ch4.z], 1);
    	atomicAdd(&d_myHistgram[ch4.w], 1);
    }
    __syncthreads();
	for (int histPos = threadIdx.x; histPos < 256; histPos += blockDim.x)
		atomicAdd(&d_histgram[histPos], d_myHistgram[histPos]);
}
