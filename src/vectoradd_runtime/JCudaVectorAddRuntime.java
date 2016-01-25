package vectoradd_runtime;

import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.cudaDeviceProp;
import utils.RuntimeCompiler;

import java.io.IOException;
import java.util.ArrayList;


public class JCudaVectorAddRuntime {
		
	public static void main(String[] args) {

        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        /* initialization by using CUDA Driver API, 
         * equivalent to CUDA driver API programming */
        cuInit(0);
        /* setting device */
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        /* generate compile options */
        cudaDeviceProp prop = new cudaDeviceProp();
        cudaGetDeviceProperties(prop, 0);
        ArrayList<String> options = new ArrayList<String>();
        /* -arch option, e.g. "-arch=compute_20" */
        String ccOption = "-arch=compute_" + prop.major + prop.minor;
        options.add(ccOption);
        /* adding -lineinfo for profiling */
        options.add("-lineinfo");
        /* debug option. */
        // options.add("-G");
        
        /* compile kernel */
        CUmodule module = new CUmodule();
        RuntimeCompiler rc = new RuntimeCompiler();
		try {
			rc.setSourceFile("kernels/JCudaVectorAddKernel.cu");
	        rc.setOptions(options.toArray(new String[1]));
	        rc.compile();
		} catch (IOException e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
			return;
		}

		/* module initialization */
		byte[] ptx = rc.getPTX();
        System.out.println(new String(ptx));
        cuModuleLoadData(module, ptx);
        
        /* Obtain a function pointer to the kernel function. */
		CUfunction function = new CUfunction();
		cuModuleGetFunction(function, module, "add"); 		
	
		/* input preparation */

		int numElements = 1024;
		float inputA[] = new float[numElements];
		float inputB[] = new float[numElements];
		float output[] = new float[numElements];

		for (int idx = 0; idx < numElements; ++idx) {
			inputA[idx] = idx;
			inputB[idx] = numElements - idx ;
		}

		/* allocating device memory */
		Pointer dInputA = new Pointer();
		Pointer dInputB = new Pointer();
		Pointer dOutput = new Pointer();
		int bufSize = Sizeof.FLOAT * numElements;
		cudaMalloc(dInputA, bufSize);
		cudaMalloc(dInputB, bufSize);
		cudaMalloc(dOutput, bufSize);

		cudaMemcpy(dInputA, Pointer.to(inputA), bufSize, cudaMemcpyHostToDevice);
		cudaMemcpy(dInputB, Pointer.to(inputB), bufSize, cudaMemcpyHostToDevice);
		
		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(
		    Pointer.to(new int[]{numElements}),
		    Pointer.to(dInputA),
		    Pointer.to(dInputB),
		    Pointer.to(dOutput)
		);

		int blockSize = 128;
		int gridSize = (numElements + blockSize - 1) / blockSize;

		// Call the kernel function.
		cuLaunchKernel(function, 
		    gridSize,  1, 1,      // Grid dimension 
		    blockSize, 1, 1,      // Block dimension
		    0, null,               // Shared memory size and stream 
		    kernelParameters, null // Kernel- and extra parameters
		);  		
		// cudaDeviceSynchronize();

		cudaMemcpy(Pointer.to(output), dOutput, bufSize, cudaMemcpyDeviceToHost);

		/* validate */
		boolean passed = true;
		for (int idx = 0; idx < numElements; ++idx) {
			float c = inputA[idx] + inputB[idx];
			if (c != output[idx]) {
				System.out.println(idx + " : " + output[idx]);
				passed = false;
			}
		}
		System.out.println(passed ? "PASSED" : "FAIL");
	
		/* termination */
		cudaFree(dInputA);
		cudaFree(dInputB);
		cudaFree(dOutput);
		cuModuleUnload(module);
		cuCtxDestroy(context);
		cudaDeviceReset();
	}
}
