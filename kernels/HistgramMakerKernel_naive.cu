
extern "C"
__global__
void histgramMakerKernel_naive(int *d_histgram, 
							   const unsigned char* d_text, int textLength) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < textLength) {
    	unsigned char ch = d_text[gid];
    	atomicAdd(&d_histgram[(int)ch], 1);
    }
}
