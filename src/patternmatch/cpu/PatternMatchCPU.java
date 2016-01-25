package patternmatch.cpu;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import patternmatch.PatternMatchCommon;
import utils.BenchmarkTimer;

public class PatternMatchCPU extends PatternMatchCommon {

    private int[] right;     // the bad-character skip array
    private byte[] pattern;  // store the pattern as a character array

    /**
     * Preprocesses the pattern string.
     *
     * @param pattern the pattern string
     * @param R the alphabet size
     */
    public PatternMatchCPU(byte[] pattern, int R) {
        this.pattern = pattern;
        // position of rightmost occurrence of c in the pattern
        right = new int[R];
        for (int c = 0; c < R; c++)
            right[c] = -1;
        for (int j = 0; j < pattern.length; j++)
            right[pattern[j]] = j;
    }

    /**
     * Returns the index of the first occurrrence of the pattern string
     * in the text string.
     *
     * @param  text the text string
	 * @param  offset offset of the text string to be searched.
     * @return the index of the first occurrence of the pattern string
     *         in the text string; N if no such match
     */
    public int search(byte[] text, int offset) {
        int M = pattern.length;
        int N = text.length;
        int skip;
        for (int i = offset; i <= N - M; i += skip) {
            skip = 0;
            for (int j = M-1; j >= 0; j--) {
                if (pattern[j] != text[i+j]) {
                    skip = Math.max(1, j - right[text[i+j]]);
                    break;
                }
            }
            if (skip == 0) return i;    // found
        }
        return N;                       // not found
    }

	
	public static
	void main(String[] args) {
		String filename = "pi/pi.txt";

		BenchmarkTimer timer = new BenchmarkTimer();
		timer.start("Pattern Match (1 core)");
		
		byte[] pi = null;
		try {
			pi = Files.readAllBytes(Paths.get(filename));
		} catch (IOException e) { 
			System.err.println("Failed to read pi.txt");
			e.printStackTrace();
			return;
		}
		timer.record("load");

		byte[] pattern = PatternMatchCommon.pattern.getBytes();
		
		PatternMatchCPU bm = new PatternMatchCPU(pattern, 256);
		ArrayList<Integer> found = new ArrayList<Integer>(100);

		/* CPU */
		int offset = 0;
		while (true) {
			offset = bm.search(pi, offset);
			if (offset == pi.length)
				break;
			found.add(offset);
			offset += pattern.length;
		}
		timer.record("match");
		
		if (found.isEmpty()) {
			System.out.println("Not found");
		}
		else {
			System.out.println("Offset : " + found.toString());
		}

		timer.output(System.out);
	}
}
