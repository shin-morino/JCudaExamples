package histgram.cpu;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import histgram.HistgramCommon;
import utils.BenchmarkTimer;

public class HisgramMaker {

	public static int[] make(byte[] sequence) {
		int[] histgram = new int[256];
		for (int idx = 0; idx < sequence.length; ++idx)
			++histgram[sequence[idx]];
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

		timer.start("Histgram(CPU) 1 core");
		int[] histgram = HisgramMaker.make(pi);
		timer.record("make");

		HistgramCommon.outputHistgram("CPU", histgram);
		timer.output(System.out);
	}
	
}
