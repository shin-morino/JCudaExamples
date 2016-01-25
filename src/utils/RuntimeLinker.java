package utils;

import jcuda.driver.CUlinkState;
import jcuda.driver.JITOptions;

import static jcuda.driver.JCudaDriver.*;

import java.io.PrintStream;
import java.util.ArrayList;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUjitInputType;


public class RuntimeLinker {
	CUlinkState state = new CUlinkState();
	RuntimeLinkerOptions defaultOptions;
	ArrayList<Object> inputs = new ArrayList<Object>();
	
	public RuntimeLinker(RuntimeLinkerOptions defaultOptions) {
		/* set default options */
		this.defaultOptions = defaultOptions;
		cuLinkCreate(this.defaultOptions.jitOptions, this.state);
	}

	public void addFile(int inputType, String path, RuntimeLinkerOptions options) {
		JITOptions jitOptions = (options != null) ? options.jitOptions : new JITOptions();
		cuLinkAddFile(this.state, inputType, path, jitOptions);
		this.inputs.add(path);
	}

	public void addData(int inputType, byte[] data, String name, RuntimeLinkerOptions options) {
		JITOptions jitOptions = (options != null) ? options.jitOptions : new JITOptions();
		cuLinkAddData(this.state, inputType, Pointer.to(data), data.length, name, jitOptions);
		this.inputs.add(data);
	}
	
	public void addPTXFile(String path, RuntimeLinkerOptions options) {
		addFile(CUjitInputType.CU_JIT_INPUT_PTX, path, options);
	}

	public void addPTXData(byte[] data, String name, RuntimeLinkerOptions options) {
		addData(CUjitInputType.CU_JIT_INPUT_PTX, data, name, options);
	}

	public Pointer complete(PrintStream stream) {
		Pointer cubinOut = new Pointer();
		long[] cubinOutSize = new long[]{0};
		try {
			cuLinkComplete(this.state, cubinOut, cubinOutSize);
		}
		catch (CudaException ex) {
			stream.print(this.defaultOptions.getInfoLog());
			stream.print(this.defaultOptions.getErrorLog());
			throw ex;
		}
		stream.print(this.defaultOptions.getInfoLog());
		return cubinOut;
	}

	public void destroy() {
		cuLinkDestroy(this.state);
	}
	
}
