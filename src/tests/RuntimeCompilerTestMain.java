package tests;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataJIT;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.driver.JITOptions;
import utils.CUDAEnvHelper;
import utils.RuntimeCompiler;
import utils.RuntimeLinker;
import utils.RuntimeLinkerOptions;

public class RuntimeCompilerTestMain {

	public static void main(String[] args) {
		String code = 
				"extern \"C\"\n" +
				"__global__ void add(float *dc, const float *da, const float *db, int size) {\n" +
				"  int gid = blockDim.x * blockIdx.x + threadIdx.x;\n" +
				"  if (gid < size)\n" +
				"    dc[gid] = da[gid] + db[gid];\n" +
				"}\n";
		
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
		RuntimeCompiler rc = new RuntimeCompiler();
		rc.setSourceString(code.getBytes(), "main");
		rc.setOptions(new String[]{"-arch=compute_50"});
		rc.compile();
		System.out.println("Log          : " + rc.getProgramLog());
		System.out.println("PTX          : " + rc.getPTX());
		
		CUDAEnvHelper.initAndSetDevice(0);
		
		RuntimeLinkerOptions linkerOptions = RuntimeLinkerOptions.createDefaultOptions(false, null);
		linkerOptions.setLogVerbose(true);
		RuntimeLinker linker = new RuntimeLinker(linkerOptions);
		linker.addPTXData(rc.getPTX(), "main", null);
		Pointer cubin = linker.complete(System.err);
		
        CUmodule module = new CUmodule();
        // System.out.println(new String(ptx));
        cuModuleLoadDataJIT(module, cubin, new JITOptions());
        
        /* Obtain a function pointer to the kernel function. */
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, "add"); 		
		
		linker.destroy();
	}
}
