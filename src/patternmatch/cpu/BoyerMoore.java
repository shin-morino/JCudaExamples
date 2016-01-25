package patternmatch.cpu;
import java.io.PrintStream;

/**
 *  The <tt>BoyerMoore</tt> class finds the first occurrence of a pattern string
 *  in a text string`.
 *  <p>
 *  This implementation uses the Boyer-Moore algorithm (with the bad-character
 *  rule, but not the strong good suffix rule).
 *  <p>
 *  For additional documentation,
 *  see <a href="http://algs4.cs.princeton.edu/53substring">Section 5.3</a> of
 *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 */
public class BoyerMoore {
    private int[] right;     // the bad-character skip array
    private byte[] pattern;  // store the pattern as a character array

    /**
     * Preprocesses the pattern string.
     *
     * @param pattern the pattern string
     * @param R the alphabet size
     */
    public BoyerMoore(byte[] pattern, int R) {
        this.pattern = new byte[pattern.length];
        for (int j = 0; j < pattern.length; j++)
            this.pattern[j] = pattern[j];

        // position of rightmost occurrence of c in the pattern
        this.right = new int[R];
        for (int c = 0; c < R; c++)
        	this.right[c] = -1;
        for (int j = 0; j < pattern.length; j++)
        	this.right[pattern[j]] = j;
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
    public int search(byte[] text, int offset) {
        int M = pattern.length;
        int N = text.length;
        int skip;
        for (int i = offset; i <= N - M; i += skip) {
            skip = 0;
            for (int j = M-1; j >= 0; j--) {
                if (pattern[j] != text[i+j]) {
                    skip = Math.max(1, j - this.right[text[i+j]]);
                    break;
                }
            }
            if (skip == 0) return i;    // found
        }
        return N;                       // not found
    }


    /**
     * Takes a pattern string and an input string as command-line arguments;
     * searches for the pattern string in the text string; and prints
     * the first occurrence of the pattern string in the text string.
     */
    public static void main(String[] args) {
    	PrintStream StdOut = System.out;
    	
        String pat = args[0];
        String txt = args[1];
        byte[] pattern = pat.getBytes();
        byte[] text    = txt.getBytes();

        BoyerMoore boyermoore2 = new BoyerMoore(pattern, 256);
        int offset2 = boyermoore2.search(text, 0);

        // print results
        StdOut.println("pattern:    " + pat);
        StdOut.println("Found : " + offset2);
    }
}
