package histgram.cuda;

import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaDeviceProp;
import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import jcuda.Pointer;
import jcuda.Sizeof;
import utils.BenchmarkTimer;
import utils.CUDAEnvHelper;
import utils.RuntimeCompiler;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import histgram.HistgramCommon;


public class CUDAHistgramMaker_GlobalMemAtomics {
	CUfunction function;
	Pointer d_text;
	long textLength;
	Pointer d_partialHistgram;
	Pointer d_histgram;

	int gridDim;
	final int blockDim = 64;
	
	public void 
	compile(CUmodule module, boolean generateDebugInfo) throws IOException {
	    RuntimeCompiler rc = new RuntimeCompiler();
	    rc.setSourceFile("kernels/HistgramMakerKernel_GlobalMemAtomics.cu");
        rc.setOptions(CUDAEnvHelper.generateCompileOptions(generateDebugInfo));
        rc.compile();

	    cuModuleLoadData(module, rc.getPTX());
	    // Obtain a function pointer to the kernel function.
		this.function = new CUfunction();
		cuModuleGetFunction(this.function, module, "histgramMakerKernel_GlobalMemAtomics");
	}

	public void allocateDeviceMemory(long lengthToReserve) {
		this.d_text = new Pointer();
		cudaMalloc(this.d_text, lengthToReserve);

		/* get SM count */
		int [] device = new int[1];
		cudaGetDevice(device);
		cudaDeviceProp prop = new cudaDeviceProp();
		cudaGetDeviceProperties(prop, device[0]);
		this.gridDim = (prop.multiProcessorCount * 2048) / 64;
		this.d_partialHistgram = new Pointer();
		cudaMalloc(this.d_partialHistgram, Sizeof.INT * 256 * this.gridDim);
		this.d_histgram = new Pointer();
		cudaMalloc(this.d_histgram, Sizeof.INT * 256);
	}

	public void freeDeviceMemory() {
		cudaFree(this.d_text);
		cudaFree(this.d_partialHistgram);
		cudaFree(this.d_histgram);
	}
	
	public
	void copyTextToDevice(Pointer text, long length) {
		cudaMemcpyAsync(this.d_text, text, length, cudaMemcpyHostToDevice, null);
		this.textLength = length;
	}

	public
	int[] make() {
		cudaMemsetAsync(this.d_partialHistgram, 0, Sizeof.INT * 256 * this.gridDim, null);
		cudaMemsetAsync(this.d_histgram, 0, Sizeof.INT * 256, null);

		/* Set up the kernel parameters */
		Pointer kernelParameters = Pointer.to(
			Pointer.to(this.d_histgram),
		    Pointer.to(this.d_partialHistgram),
		    Pointer.to(this.d_text),
		    Pointer.to(new int[]{(int)this.textLength})
		);
		
		// Call the kernel function.
		cuLaunchKernel(this.function, 
		    this.gridDim,  1, 1,      // Grid dimension 
		    this.blockDim, 1, 1,      // Block dimension
		    0, null, // Shared memory size and stream 
		    kernelParameters, null // Kernel- and extra parameters
		);  		
		// cudaDeviceSynchronize();
		int[] histgram = new int[256];
		cudaMemcpy(Pointer.to(histgram), this.d_histgram, Sizeof.INT * 256, cudaMemcpyDeviceToHost);
		return histgram;
	}
	
	public static
	void main(String[] args) {

	    JCudaDriver.setExceptionsEnabled(true);
		
		BenchmarkTimer timer = new BenchmarkTimer();
		
		timer.start("CUDA Histgram Maker");
		CUcontext context = CUDAEnvHelper.initAndSetDevice(HistgramCommon.deviceNo);
		timer.record("initialize");
		
		/* GPU */
		CUDAHistgramMaker_GlobalMemAtomics cudaHistgramMaker = new CUDAHistgramMaker_GlobalMemAtomics();
		File file = null;
		Pointer pi = null;
		CUmodule module = null;

		try {
			module = new CUmodule();
			cudaHistgramMaker.compile(module, HistgramCommon.generateDebugInfo);
			timer.record("compile");

			pi = new Pointer();
			file = new File("pi/pi.txt");
			cudaHostAlloc(pi, file.length(), cudaHostAllocDefault);
			ByteBuffer buf = pi.getByteBuffer(0, file.length());
			FileInputStream istm = new FileInputStream(file);
			FileChannel channel = istm.getChannel();
			channel.read(buf);
			istm.close();

			timer.record("load file");

		} catch (IOException e) {
			e.printStackTrace();
			return;
		}

		cudaHistgramMaker.allocateDeviceMemory(file.length());
		timer.record("allocate");
		cudaHistgramMaker.copyTextToDevice(pi, file.length());
		cudaDeviceSynchronize();
		timer.record("transfer");
		int[] histgram = cudaHistgramMaker.make();
		timer.record("make");
		cudaHistgramMaker.freeDeviceMemory();
		timer.record("free");
		CUDAEnvHelper.terminate(context, module);
		timer.record("terminate");

		HistgramCommon.outputHistgram("GPU(GlobalAtomics)", histgram);
		timer.output(System.out);
	}
	
}
