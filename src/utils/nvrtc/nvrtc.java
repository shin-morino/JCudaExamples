package utils.nvrtc;

import java.util.List;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;


public class nvrtc {
	
	interface NativeLib extends Library {
		Pointer nvrtcGetErrorString(int result);
	
		int nvrtcVersion(IntByReference major, IntByReference minor);

		int nvrtcCompileProgram (long prog, int numOptions, String[] options);

		int nvrtcCreateProgram(LongByReference prog,
							   byte[] src, String name, 
							   int numHeaders,
							   Object[] headers, String[] includeNames);

		int nvrtcDestroyProgram(long[] progRef);

		int nvrtcGetProgramLog(long prog, byte[] log);

		int nvrtcGetProgramLogSize (long prog, LongByReference logSizeRet);

		int nvrtcGetPTX(long prog, byte[] ptx);
	
		int nvrtcGetPTXSize(long prog, LongByReference ptxSizeRet);
	}

	static NativeLib lib = 
			(NativeLib)Native.loadLibrary((Platform.isWindows() ? "nvrtc64_75" : "nvrtc"), NativeLib.class);
	
	public static
	String getErrorString(int result) {
		Pointer pointer = lib.nvrtcGetErrorString(result);
		int length;
		for (length = 0; pointer.getByte(length) != 0; ++length);
		byte[] errBytes = pointer.getByteArray(0, length);
		return new String(errBytes);
	}

	public static
	String version() throws NvrtcException {
		IntByReference major = new IntByReference();
		IntByReference minor = new IntByReference();
		checkError(lib.nvrtcVersion(major, minor));
		return major.getValue() + "." + minor.getValue();
	}
	
	public static
	void compileProgram(long prog, String[] options) throws NvrtcException {
		int numOptions = (options != null) ? options.length : 0;
		checkError(lib.nvrtcCompileProgram(prog, numOptions, options));
	}

	public static
	long createProgram(NamedText source, List<NamedText> headers) throws NvrtcException {
		Object[] headerBodies = null;
		String[] includeNames = null;
		int numHeaders = 0;
		if (headers != null) {
			numHeaders = headers.size();
			headerBodies = new Pointer[numHeaders];
			includeNames = new String[numHeaders];
			for (int idx = 0; idx < numHeaders; ++idx) {
				NamedText nt = headers.get(idx);
				headerBodies[idx] = nt.text;
				includeNames[idx] = nt.name;
			}
		}
		LongByReference programRef = new LongByReference();
		int res = lib.nvrtcCreateProgram(programRef, 
				source.text, source.name, numHeaders, headerBodies, includeNames);
		checkError(res);
		return programRef.getValue();
	}

	public static
	void deestroyProgram(long prog) throws NvrtcException {
		long[] progRef = new long[]{prog};
		checkError(lib.nvrtcDestroyProgram(progRef));
	}

	public static
	String getProgramLog(long prog) throws NvrtcException {
		LongByReference logSizeRef = new LongByReference();
		checkError(lib.nvrtcGetProgramLogSize(prog, logSizeRef));
		long logSize = logSizeRef.getValue();
		byte[] log = new byte[(int)logSize];
		checkError(lib.nvrtcGetProgramLog(prog, log));
		return new String(log, 0, (int)logSize - 1);
	}

	public static
	byte[] getPTX(long prog) throws NvrtcException {
		LongByReference ptxSizeRef = new LongByReference();
		checkError(lib.nvrtcGetPTXSize(prog, ptxSizeRef));
		long ptxSize = ptxSizeRef.getValue();
		byte[] ptx = new byte[(int)ptxSize];
		checkError(lib.nvrtcGetPTX(prog, ptx));
		return ptx;
	}
	
	public static class NamedText {
		public NamedText() { }
		public NamedText(String name, byte[] text) {
			this.name = name;
			this.text = text;
		}
		
		String name;
		byte[] text;
	}
	
	private static void checkError(int id) throws NvrtcException {
		if (id != NvrtcException.NVRTC_SUCCESS)
			throw new NvrtcException(id);
	}

}
