package histgram.cpu;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import histgram.HistgramCommon;
import utils.BenchmarkTimer;

public class ParallelHigramMaker {
	static class Worker extends Thread {
		int begin, end;
		byte[] sequence;
		int[] histgram = new int[256];
		
		Worker(int begin, int end, byte[] sequence) {
			this.begin = begin;
			this.end = end;
			this.sequence = sequence;
		}
		
		@Override
		public void run() {
			for (int idx = this.begin; idx < this.end; ++idx)
				++this.histgram[this.sequence[idx]];
		}
		
	}

	public static int[] make(byte[] sequence, int nThreads) {
		Worker[] workers = new Worker[nThreads];
		int nElmsPerThread = sequence.length / nThreads;
		for (int idx = 0; idx < nThreads; ++idx) {
			int begin = nElmsPerThread * idx;
			int end = Math.min(nElmsPerThread * (idx + 1), sequence.length);
			workers[idx] = new Worker(begin, end, sequence);
			workers[idx].start();
		}
		try {
			for (int idx = 0; idx < nThreads; ++idx)
				workers[idx].join();
		} catch (InterruptedException e) {
		}
		
		int[] histgram = new int[256];
		for (int threadIdx = 0; threadIdx < nThreads; ++threadIdx) {
			Worker worker = workers[threadIdx];
			for (int binIdx = 0; binIdx < 256; ++binIdx)
				histgram[binIdx] += worker.histgram[binIdx];
		}
		return histgram;
	}


	public static
	void main(String[] args) {
		String filename = "pi/pi.txt";
		byte[] pi = null;
		try {
			pi = Files.readAllBytes(Paths.get(filename));
		} catch (IOException e) { }
		if (pi == null)
			return;

		BenchmarkTimer timer = new BenchmarkTimer();

		/* CPU */
		int nCores = Runtime.getRuntime().availableProcessors();
		System.out.println("Creating histgram with " + nCores + " threads.");
		
		timer.start("Histgram(CPU) Parallel");
		int[] histgram= ParallelHigramMaker.make(pi, nCores);
		timer.record("make");

		HistgramCommon.outputHistgram("CPU", histgram);
		timer.output(System.out);
	}
}
