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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CUDABruteForceMatcher_naive {

	Pointer d_text;
	long textLength;
	Pointer d_pattern;
	int patternLength;
	Pointer d_offsets;
	Pointer d_nMatched;
	CUfunction function;
	
	
	public void compile(CUmodule module, boolean generateDebugInfo) throws IOException {
	    RuntimeCompiler rc = new RuntimeCompiler();

	    rc.setSourceFile("kernels/SearchPatternKernel_naive.cu");
        rc.setOptions(CUDAEnvHelper.generateCompileOptions(generateDebugInfo));
        rc.compile();

	    cuModuleLoadData(module, rc.getPTX());
	    // Obtain a function pointer to the kernel function.
		this.function = new CUfunction();
		cuModuleGetFunction(this.function, module, "SearchPatternKernel_naive");
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

	void setPattern(byte[] pattern) {
		this.d_pattern = new Pointer();
		cudaMalloc(this.d_pattern, pattern.length);
		cudaMemcpyAsync(d_pattern, Pointer.to(pattern), pattern.length, cudaMemcpyHostToDevice, null);
		this.patternLength = pattern.length;
	}
	
	public
	void copyTextToDevice(byte[] text) {
		cudaMemcpyAsync(this.d_text, Pointer.to(text), text.length, cudaMemcpyHostToDevice, null);
		this.textLength = text.length;
	}
	
	int[] search() {
		cudaStream_t stream = null;
		
		cudaMemsetAsync(d_nMatched, 0, Sizeof.INT, stream);
		final int nSearchThreads = (int)this.textLength - (this.patternLength - 1);
		
		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(
			Pointer.to(this.d_nMatched),
			Pointer.to(this.d_offsets),
		    Pointer.to(new int[]{PatternMatchCommon.nMaxMatched}),
		    Pointer.to(this.d_pattern),
		    Pointer.to(new int[]{this.patternLength}),
		    Pointer.to(this.d_text),
		    Pointer.to(new int[]{nSearchThreads})
		);
		
		final int blockSize = 64;
		final int gridSize = (nSearchThreads + blockSize - 1) / blockSize;

		// Call the kernel function.
		cuLaunchKernel(this.function, 
		    gridSize,  1, 1,      // Grid dimension 
		    blockSize, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    kernelParameters, null // Kernel- and extra parameters
		);  		
		cudaDeviceSynchronize();

		int[] nOffsets = new int[1];
		cudaMemcpy(Pointer.to(nOffsets), d_nMatched, Sizeof.INT, cudaMemcpyDeviceToHost);
		int[] offsets = new int[nOffsets[0]];
		cudaMemcpy(Pointer.to(offsets), d_offsets, Sizeof.INT * nOffsets[0], cudaMemcpyDeviceToHost);
		return offsets;
	}

	public static
	void main(String[] args) {
		JCudaDriver.setExceptionsEnabled(true);

		BenchmarkTimer timer = new BenchmarkTimer();
		
	    String filename = "pi/pi.txt";
		byte[] pattern = PatternMatchCommon.pattern.getBytes();

		timer.start("Brute force matcher 3");
		CUcontext context = CUDAEnvHelper.initAndSetDevice(0);
		timer.record("initialize");
		CUmodule module = new CUmodule();
		
		byte[] pi = null;
		
		/* GPU */
		CUDABruteForceMatcher_naive cudaMatcher = new CUDABruteForceMatcher_naive();
		long textLength = 0;
		try {
			cudaMatcher.compile(module, PatternMatchCommon.generateDebugInfo);
			timer.record("compile");

			/* allocate */
			Path path = Paths.get(filename);
			textLength = Files.size(path);
			cudaMatcher.allocateDeviceMemory(textLength);
			timer.record("allocate");

			/* load data from file */
			pi = Files.readAllBytes(path);
			timer.record("load pi");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		cudaMatcher.copyTextToDevice(pi);
		cudaDeviceSynchronize();
		timer.record("copy pi to GPU");
		cudaMatcher.setPattern(pattern);
		cudaDeviceSynchronize();
		timer.record("set pattern");
		int[] nOffsets = cudaMatcher.search();
		cudaDeviceSynchronize();
		timer.record("search");
		List<Integer> list = IntStream.of(nOffsets).boxed().collect(Collectors.toList());
		System.out.println("Offsets: " + list.toString());

		CUDAEnvHelper.terminate(context, module);
		timer.record("terminate");

		timer.output(System.out);
	}
}
