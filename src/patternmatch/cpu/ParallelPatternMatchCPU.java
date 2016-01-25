package patternmatch.cpu;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import patternmatch.PatternMatchCommon;
import utils.BenchmarkTimer;

public class ParallelPatternMatchCPU extends PatternMatchCommon {
    private int[] right;     // the bad-character skip array
    private byte[] pattern;  // store the pattern as a character array
    private byte[] text;
    /**
     * Preprocesses the pattern string.
     *
     * @param pattern the pattern string
     * @param R the alphabet size
     */
    public ParallelPatternMatchCPU(byte[] text, byte[] pattern, int R) {
    	this.text = text;
    	this.pattern = pattern;
        // position of rightmost occurrence of c in the pattern
        right = new int[R];
        for (int c = 0; c < R; c++)
            right[c] = -1;
        
        for (int j = 0; j < pattern.length; j++)
            right[pattern[j]] = j;
    }

    class Worker extends Thread {
    	ArrayList<Integer> offsets = new ArrayList<Integer>(100);
    	int begin, end;
    	
    	Worker(int begin, int end) {
    		this.begin = begin;
    		this.end = end;
    	}
    	
	    /**
	     * Returns the index of the first occurrence of the pattern string
	     * in the text string.
	     *
	     * @param  text the text string
		 * @param  offset offset of the text string to be searched.
	     * @return the index of the first occurrence of the pattern string
	     *         in the text string; N if no such match
	     */

		@Override
		public void run() {
	        final byte[] pattern = ParallelPatternMatchCPU.this.pattern;
	        final byte[] text = ParallelPatternMatchCPU.this.text;
	        final int[] right = ParallelPatternMatchCPU.this.right;

	        final int M = pattern.length;
	        int skip;
	        for (int i = begin; i < end; i += skip) {
	            skip = 0;
	            for (int j = M-1; j >= 0; j--) {
	                if (pattern[j] != text[i+j]) {
	                    skip = Math.max(1, j - right[text[i+j]]);
	                    break;
	                }
	            }
	            if (skip == 0) {
	            	offsets.add(i);    // found
	            	skip = pattern.length;
	            }
	        }
		}
    }

	ArrayList<Integer> find(int nThreads) {
		Worker[] workers = new Worker[nThreads];
		int elmsPerThread = this.text.length / nThreads;
		for (int idx = 0; idx < nThreads; ++idx) {
			int begin = elmsPerThread * idx;
			int end = Math.min(this.text.length - this.pattern.length + 1, elmsPerThread * (idx + 1));
			workers[idx] = new Worker(begin, end);
			workers[idx].start();
		}
		for (int idx = 0; idx < nThreads; ++idx) {
			try {
				workers[idx].join();
			} catch (InterruptedException e) { }
		}
		int nOffsets = 0;
		for (int threadIdx = 0; threadIdx < nThreads; ++threadIdx)
			nOffsets += workers[threadIdx].offsets.size();
		ArrayList<Integer> offsets = new ArrayList<Integer>(nOffsets);
		for (int threadIdx = 0; threadIdx < nThreads; ++threadIdx) {
			Worker worker = workers[threadIdx];
			int nOffsetInWorker = worker.offsets.size();
			for (int offsetIdx = 0; offsetIdx < nOffsetInWorker; ++offsetIdx)
				offsets.addAll(worker.offsets);
		}
		return offsets;
	}
    
    public static
	void main(String[] args) {
		String filename = "pi/pi.txt";

		BenchmarkTimer timer = new BenchmarkTimer();
		timer.start("Pattern Match (parallel)");
		
		byte[] pi = null;
		try {
			pi = Files.readAllBytes(Paths.get(filename));
		} catch (IOException e) { 
			System.err.println("Failed to read pi.txt");
			e.printStackTrace();
			return;
		}

		timer.record("load");

		final byte[] pattern = PatternMatchCommon.pattern.getBytes();
		ParallelPatternMatchCPU pm = new ParallelPatternMatchCPU(pi, pattern, 256);
		ArrayList<Integer> offsets = pm.find(Runtime.getRuntime().availableProcessors());
		timer.record("match");
		
		if (offsets.isEmpty()) {
			System.out.println("Not found");
		}
		else {
			System.out.println("Offset : " + offsets.toString());
		}
		timer.output(System.out);
    }
}
