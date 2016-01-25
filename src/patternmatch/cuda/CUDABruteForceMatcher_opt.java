package patternmatch.cuda;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.driver.*;
import jcuda.runtime.cudaStream_t;
import patternmatch.PatternMatchCommon;
import utils.BenchmarkTimer;
import utils.CUDAEnvHelper;
import utils.RuntimeCompiler;
import jcuda.Pointer;
import jcuda.Sizeof;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CUDABruteForceMatcher_opt {
	Pointer d_text;
	int textLength;
	int patternLength;
	Pointer d_offsets;
	Pointer d_nMatched;
	CUfunction function;
	
	
	public void compile(CUmodule module, boolean generateDebugInfo) throws IOException {
	    RuntimeCompiler rc = new RuntimeCompiler();

	    rc.setSourceFile("kernels/SearchPatternKernel_opt.cu");
        rc.setOptions(CUDAEnvHelper.generateCompileOptions(generateDebugInfo));
        rc.compile();

	    cuModuleLoadData(module, rc.getPTX());
	    // Obtain a function pointer to the kernel function.
		this.function = new CUfunction();
		cuModuleGetFunction(this.function, module, "searchPatternKernel_opt");
	}

	public void allocateDeviceMemory(long lengthToReserve) {
		this.d_offsets = new Pointer();
		this.d_nMatched = new Pointer();
		cudaMalloc(d_nMatched, Sizeof.INT);
		cudaMalloc(d_offsets, Sizeof.INT * PatternMatchCommon.nMaxMatched);
		
		this.d_text = new Pointer();
		long txtLength4 = ((lengthToReserve + 3) / 4) * 4;
		cudaMalloc(this.d_text, txtLength4);
	}

	public void freeDeviceMemory() {
		cudaFree(this.d_text);
		cudaFree(this.d_offsets);
		cudaFree(this.d_nMatched);
	}
	
	public
	void copyTextToDevice(Pointer text, int length) {
		cudaMemcpyAsync(this.d_text, text, length, cudaMemcpyHostToDevice, null);
		this.textLength = length;

		Pointer d_textWithOffset = this.d_text.withByteOffset(length);
		long txtLength4 = ((length + 3) / 4) * 4;
		cudaMemset(d_textWithOffset, 0, txtLength4 - length);
	}
	
	void setPattern(CUmodule module, byte[] pattern) {
		CUdeviceptr d_pattern = new CUdeviceptr();
		long [] bytes = new long[1];
		cuModuleGetGlobal(d_pattern, bytes, module, "pattern");
		cudaMemcpyAsync(d_pattern, Pointer.to(pattern), pattern.length, cudaMemcpyHostToDevice, null);
		this.patternLength = pattern.length;
	}
	
	int[] search() {
		cudaStream_t stream = null;
		
		cudaMemsetAsync(d_nMatched, 0, Sizeof.INT, stream);
		int nSearchThreads = (int)this.textLength - (patternLength - 1);
		int nSearchThreads4 = (nSearchThreads + 3) / 4;
		
		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(
			Pointer.to(d_nMatched),
			Pointer.to(d_offsets),
			Pointer.to(new int[]{this.patternLength}),
		    Pointer.to(new int[]{PatternMatchCommon.nMaxMatched}),
		    Pointer.to(d_text),
		    Pointer.to(new long[]{nSearchThreads4})
		);
		
		final int blockDim = 64;
		final int gridDim = (nSearchThreads4 + blockDim - 1) / blockDim;

		// Call the kernel function.
		cuLaunchKernel(this.function, 
			gridDim,  1, 1,      // Grid dimension 
		    blockDim, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    kernelParameters, null // Kernel- and extra parameters
		);  		
		cudaDeviceSynchronize();

		int[] nOffsets = new int[1];
		cudaMemcpy(Pointer.to(nOffsets), d_nMatched, Sizeof.INT, cudaMemcpyDeviceToHost);
		int[] offsets = new int[nOffsets[0]];
		cudaMemcpy(Pointer.to(offsets), d_offsets, Sizeof.INT * nOffsets[0], cudaMemcpyDeviceToHost);
		Arrays.sort(offsets);
		return offsets;
		
	}

	public static
	void main(String[] args) {
		JCudaDriver.setExceptionsEnabled(true);

		BenchmarkTimer timer = new BenchmarkTimer();
		
	    String filename = "pi/pi.txt";
		byte[] pattern = PatternMatchCommon.pattern.getBytes();

		timer.start("Brute force matcher 2");
		CUcontext context = CUDAEnvHelper.initAndSetDevice(0);
		timer.record("initialize");
		CUmodule module = new CUmodule();
		
		Pointer piHost = null;
		
		/* GPU */
		CUDABruteForceMatcher_opt cudaMatcher = new CUDABruteForceMatcher_opt();
		long textLength = 0;
		try {
			cudaMatcher.compile(module, PatternMatchCommon.generateDebugInfo);
			timer.record("compile");

			/* allocate */
			Path path = Paths.get(filename);
			FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
			textLength = Files.size(path);

			piHost = new Pointer();
			jcuda.runtime.JCuda.cudaHostAlloc(piHost, textLength, jcuda.runtime.JCuda.cudaHostAllocPortable);
			cudaMatcher.allocateDeviceMemory(textLength);
			timer.record("allocate");

			/* load data from file to pinned memory */
			ByteBuffer bb = piHost.getByteBuffer(0, textLength);
			channel.read(bb);
			timer.record("load pi");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}

		cudaMatcher.copyTextToDevice(piHost, (int)textLength);
		cudaDeviceSynchronize();
		timer.record("copy pi to GPU");
		cudaMatcher.setPattern(module, pattern);
		cudaDeviceSynchronize();
		timer.record("set pattern");
		int[] nOffsets = cudaMatcher.search();
		cudaDeviceSynchronize();
		timer.record("search");
		List<Object> list = IntStream.of(nOffsets).boxed().collect(Collectors.toList());
		System.out.println("Offsets: " + list.toString());

		CUDAEnvHelper.terminate(context, module);
		timer.record("terminate");

		timer.output(System.out);
	}
	
}
