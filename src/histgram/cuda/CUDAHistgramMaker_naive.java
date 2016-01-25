package histgram.cuda;

import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.cudaStream_t;
import utils.BenchmarkTimer;
import utils.CUDAEnvHelper;
import utils.RuntimeCompiler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import histgram.HistgramCommon;


public class CUDAHistgramMaker_naive {
	CUfunction function;
	Pointer d_text;
	long textLength;
	Pointer d_histgram;

	public void 
	compile(CUmodule module, boolean generateDebugInfo) throws IOException {
	    RuntimeCompiler rc = new RuntimeCompiler();
	    rc.setSourceFile("kernels/HistgramMakerKernel_naive.cu");
        rc.setOptions(CUDAEnvHelper.generateCompileOptions(generateDebugInfo));
        rc.compile();

	    cuModuleLoadData(module, rc.getPTX());
	    // Obtain a function pointer to the kernel function.
		this.function = new CUfunction();
		cuModuleGetFunction(this.function, module, "histgramMakerKernel_naive");
	}

	public void allocateDeviceMemory(long lengthToReserve) {
		this.d_text = new Pointer();
		cudaMalloc(this.d_text, lengthToReserve);
		this.d_histgram = new Pointer();
		cudaMalloc(this.d_histgram, Sizeof.INT * 256);
	}

	public void freeDeviceMemory() {
		if (this.d_text != null)
			cudaFree(this.d_text);
		if (this.d_histgram != null)
			cudaFree(this.d_histgram);
		this.d_text = this.d_histgram = null;
	}
	
	public
	void copyTextToDevice(byte[] text) {
		cudaMemcpyAsync(this.d_text, Pointer.to(text), text.length, cudaMemcpyHostToDevice, null);
		this.textLength = text.length;
	}
	
	int[] make() {
		cudaStream_t stream = null;
		cudaMemsetAsync(this.d_histgram, 0, Sizeof.INT * 256, stream);

		/* Set up the kernel parameters */
		Pointer kernelParameters = Pointer.to(
		    Pointer.to(this.d_histgram),
		    Pointer.to(this.d_text),
		    Pointer.to(new int[]{(int)this.textLength})
		);
		
		int blockSize = 128;
		int gridSize = (int)((textLength + blockSize - 1) / blockSize);
		// Call the kernel function.
		cuLaunchKernel(function, 
		    gridSize,  1, 1,      // Grid dimension 
		    blockSize, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    kernelParameters, null // Kernel- and extra parameters
		);  		
		int[] histgram = new int[256];
		cudaMemcpy(Pointer.to(histgram), this.d_histgram, Sizeof.INT * 256, cudaMemcpyDeviceToHost);
		return histgram;
	}
	
	public static
	void main(String[] args) {

	    JCudaDriver.setExceptionsEnabled(true);
		
		BenchmarkTimer timer = new BenchmarkTimer();

		
		timer.start("CUDA Histgram Maker (naive)");
		CUcontext context = CUDAEnvHelper.initAndSetDevice(HistgramCommon.deviceNo);
		timer.record("initialize");
		
		/* GPU */
		CUDAHistgramMaker_naive cudaHistgramMaker = new CUDAHistgramMaker_naive();
		byte[] pi = null;
		CUmodule module = new CUmodule();

		try {
			cudaHistgramMaker.compile(module, HistgramCommon.generateDebugInfo);
			timer.record("compile");

			Path path = Paths.get("pi/pi.txt");
			pi = Files.readAllBytes(path);
			timer.record("load file");

		} catch (IOException e) {
			e.printStackTrace();
			return;
		}

		cudaHistgramMaker.allocateDeviceMemory(pi.length);
		timer.record("allocate");
		cudaHistgramMaker.copyTextToDevice(pi);
		cudaDeviceSynchronize();
		timer.record("transfer");
		int[] histgram = cudaHistgramMaker.make();
		timer.record("make");
		cudaHistgramMaker.freeDeviceMemory();
		timer.record("free");
		CUDAEnvHelper.terminate(context, module);
		timer.record("terminate");

		HistgramCommon.outputHistgram("GPU(naive)", histgram);
		timer.output(System.out);
	}

}
