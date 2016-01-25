package histgram;

public class HistgramCommon {
	public final static boolean generateDebugInfo = false;
	public final static int deviceNo = 0;

	public static void outputHistgram(String caption, int[] histgram) {
		System.out.println(caption);
		for (int idx = 0; idx < 256; ++idx) {
			if (histgram[idx] == 0)
				continue;
			System.out.println((char)idx + " : " + histgram[idx]);
		}
		System.out.println();
	}
}
