package tests;

import utils.nvrtc.nvrtc;

public class NvrtcTestMain {

	public static void main(String[] args) {
		String code = 
				"__global__ void add(float *dc, const float *da, const float *db, int size) {\n" +
				"  int gid = blockDim.x * blockIdx.x + threadIdx.x;\n" +
				"  if (gid < size)\n" +
				"    dc[gid] = da[gid] + db[gid];\n" +
				"}\n";
		
		try {
			System.out.println("Error string : " + nvrtc.getErrorString(0));
			System.out.println("Error string : " + nvrtc.getErrorString(5));
			System.out.println("Version      : " + nvrtc.version());

			nvrtc.NamedText src = new nvrtc.NamedText("main", code.getBytes());
			long program = nvrtc.createProgram(src, null);
			String[] options = {"-arch=compute_50", "-G" };
			nvrtc.compileProgram(program, options);
			System.out.println("Log          : " + nvrtc.getProgramLog(program));
			System.out.println("PTX          : " + nvrtc.getPTX(program));
		
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e.getMessage());
		}		
	}
}
