package utils;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;

import java.util.ArrayList;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUmodule;

import static jcuda.driver.CUdevice_attribute.*;


public class CUDAEnvHelper {

	public static CUcontext initAndSetDevice(int devOrd) {
	    cuInit(0);
	    CUdevice device = new CUdevice();
	    cuDeviceGet(device, devOrd);
	    CUcontext context = new CUcontext();
	    cuCtxCreate(context, 0, device);
	    return context;
	}
	
	public static void terminate(CUcontext context, CUmodule module) {
		cuModuleUnload(module);
		cuCtxDestroy(context);
		cudaDeviceReset();
	}


	public static String[] generateCompileOptions(boolean debugEnabled) {
		ArrayList<String> options = new ArrayList<String>();

		CUdevice device = new CUdevice();
		cuCtxGetDevice(device);
		int cc = getComputeCapability(device);
		options.add("-arch=compute_" + cc);
		options.add("-lineinfo");
		if (debugEnabled)
			options.add("-G");

		return options.toArray(new String[1]);
	}

	public static int getComputeCapability(CUdevice device) {
		int[] major = new int[1];
		int[] minor = new int[1];
		cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
		cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
		return major[0] * 10 + minor[0];
	}

}
