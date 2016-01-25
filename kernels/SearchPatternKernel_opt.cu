

extern "C"
__constant__ unsigned int pattern[256];


extern "C"
__global__
void searchPatternKernel_opt(int *d_nFound, int *d_offsets, int patternLength,
							 int nMaxMatched,
                             const unsigned int* __restrict__ d_text4, int searchLength4) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < searchLength4) {
        bool matched[4] = { true, true, true, true };

        const unsigned int *d_myPos4 = &d_text4[gid];
        unsigned int chars4_1 = d_myPos4[0];
		bool noMatch = false;
		int patternLength4 = patternLength / 4;
        int idx = 0;
		for (;idx < patternLength4; ++idx) {
        	unsigned int chars4_0 = chars4_1;
        	chars4_1 = d_myPos4[idx + 1];
			unsigned int pattern4 = pattern[idx];
        	matched[0] &= (pattern4 == chars4_0);
        	matched[1] &= (pattern4 == __byte_perm(chars4_0, chars4_1, 0x4321));
        	matched[2] &= (pattern4 == __byte_perm(chars4_0, chars4_1, 0x5432));
        	matched[3] &= (pattern4 == __byte_perm(chars4_0, chars4_1, 0x6543));
			noMatch = (!matched[0] && !matched[1]) && (!matched[2] && !matched[3]);
			if (noMatch)
				return;
		}				
        int nToBeCompared = patternLength - patternLength4 * 4;
        if (nToBeCompared != 0) {
	    	unsigned int chars4_0 = chars4_1;
	    	chars4_1 = d_myPos4[idx + 1];
			unsigned int patternMask = (0xffffffff >> ((4 - nToBeCompared) * 8));
			unsigned int pattern4 = pattern[patternLength4] & patternMask;
	    	matched[0] &= (pattern4 == (chars4_0 & patternMask));
	    	matched[1] &= (pattern4 == (__byte_perm(chars4_0, chars4_1, 0x4321) & patternMask));
	    	matched[2] &= (pattern4 == (__byte_perm(chars4_0, chars4_1, 0x5432) & patternMask));
	    	matched[3] &= (pattern4 == (__byte_perm(chars4_0, chars4_1, 0x6543) & patternMask));
			noMatch = (!matched[0] && !matched[1]) && (!matched[2] && !matched[3]);
			if (noMatch)
				return;
		}

#pragma unroll
		for (int idx = 0; idx < 4; ++idx) {
        	if (matched[idx]) {
    	        int offsetPos = atomicAdd(d_nFound, 1);
        	    if (offsetPos < nMaxMatched)
            		d_offsets[offsetPos] = gid * 4 + idx;
            }
        }
    }
}
