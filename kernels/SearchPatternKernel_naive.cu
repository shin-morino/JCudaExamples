

extern "C"
__global__
void SearchPatternKernel_naive(int *d_nFound, int *d_offsets, int nMaxMatched,
							   const unsigned char *d_pattern, int patternLength,
                               const unsigned char *d_text, int searchLength) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < searchLength) {
        const unsigned char *d_myPos = &d_text[gid];
        int idx = 0;
        for (; idx < patternLength; ++idx) {
            if (d_pattern[idx] != d_myPos[idx])
                break;
        }

        if (idx == patternLength) {
            int offsetPos = atomicAdd(d_nFound, 1);
            if (offsetPos < nMaxMatched)
            	d_offsets[offsetPos] = gid;
        }
    }
}
