package utils;

import jcuda.driver.JITOptions;
import static jcuda.driver.CUjit_option.*;
import jcuda.driver.CUdevice;

public class RuntimeLinkerOptions {
	JITOptions jitOptions = new JITOptions();
	
    /**
     * Max number of registers that a thread may use.<br />
     * Applies to: compiler only
     */
	public void setMaxRegisters(int maxRegisters) {
		this.jitOptions.putInt(CU_JIT_MAX_REGISTERS, maxRegisters);
	}

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for<br />
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.<br />
     * Cannot be combined with ::CU_JIT_TARGET.<br />
     * Option type: unsigned int<br />
     * Applies to: compiler only
     */
	public void setThreadsPerBlock(int threadsPerBlock) {
		this.jitOptions.putInt(CU_JIT_THREADS_PER_BLOCK, threadsPerBlock);
	}

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker<br />
     * Applies to: compiler and linker
     */
	public 	void setWallTime(float wallTime) {
		this.jitOptions.putFloat(CU_JIT_WALL_TIME, wallTime);
	}
	
	public void prepareLogBuffers(int logBufferSizeBytes, int errLogBufferSizeBytes) {
		byte[] logBuffer = new byte[logBufferSizeBytes];
	    /**
	     * Pointer to a buffer in which to print any log messages
	     * that are informational in nature (the buffer size is specified via
	     * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)<br />
	     * Applies to: compiler and linker
	     */
		this.jitOptions.putBytes(CU_JIT_INFO_LOG_BUFFER, logBuffer);
	    /**
	     * IN: Log buffer size in bytes.  Log messages will be capped at this size
	     * (including null terminator)<br />
	     * OUT: Amount of log buffer filled with messages<br />
	     * Applies to: compiler and linker
	     */
		this.jitOptions.putInt(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, logBufferSizeBytes);

		byte[] errLogBuffer = new byte[errLogBufferSizeBytes];
	    /**
	     * Pointer to a buffer in which to print any log messages that
	     * reflect errors (the buffer size is specified via option
	     * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)<br />
	     * Applies to: compiler and linker
	     */
		this.jitOptions.putBytes(CU_JIT_ERROR_LOG_BUFFER, errLogBuffer);
	
	    /**
	     * IN: Log buffer size in bytes.  Log messages will be capped at this size
	     * (including null terminator)<br />
	     * OUT: Amount of log buffer filled with messages<br />
	     * Applies to: compiler and linker
	     */
		this.jitOptions.putInt(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, errLogBufferSizeBytes);
	}

	public void setOptimizationLevel(int optimizationLevel) {
	    /**
	     * Level of optimizations to apply to generated code (0 - 4), with 4
	     * being the default and highest level of optimizations.<br />
	     * Applies to: compiler only
	     */
		this.jitOptions.putInt(CU_JIT_OPTIMIZATION_LEVEL, optimizationLevel);
	}

	public void setTargetFromContext() {
		/**
	     * No option value required. Determines the target based on the current
	     * attached context (default)<br />
	     * Applies to: compiler and linker
	     */
		this.jitOptions.put(CU_JIT_TARGET_FROM_CUCONTEXT);
	}

	public void setTarget(CUdevice device) {
	    /**
	     * Target is chosen based on supplied ::CUjit_target.  Cannot be
	     * combined with ::CU_JIT_THREADS_PER_BLOCK.<br />
	     * Option type: unsigned int for enumerated type ::CUjit_target<br />
	     * Applies to: compiler and linker
	     */
		int cc = CUDAEnvHelper.getComputeCapability(device);
		this.jitOptions.putInt(CU_JIT_TARGET, cc);
	}

	public void setFallbackStrategy(int fallbackStrategy) {
		/**
	     * Specifies choice of fallback strategy if matching cubin is not found.
	     * Choice is based on supplied ::CUjit_fallback.<br />
	     * Option type: unsigned int for enumerated type ::CUjit_fallback<br />
	     * Applies to: compiler only
	     */
		this.jitOptions.putInt(CU_JIT_FALLBACK_STRATEGY, fallbackStrategy);
	}

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler and linker
     */
	public void setGenerateDebugInfo(boolean generateDebugInfo) {
		this.jitOptions.putInt(CU_JIT_GENERATE_DEBUG_INFO, generateDebugInfo ? 1 : 0);
	}

    /**
     * Generate verbose log messages (0: false, default)<br />
     * Option type: int<br />
     * Applies to: compiler and linker
     */
	public void setLogVerbose(boolean logVerbose) {
		this.jitOptions.putInt(CU_JIT_LOG_VERBOSE, logVerbose ? 1 : 0);
	}

	/**
     * Generate line number information (-lineinfo) (0: false, default)<br />
     * Applies to: compiler only
     */
	public void setGenerateLineInfo(boolean generateLineInfo) {
		this.jitOptions.putInt(CU_JIT_GENERATE_LINE_INFO, generateLineInfo ? 1 : 0);
	}

	/**
     * Specifies whether to enable caching explicitly (-dlcm) <br />
     * Choice is based on supplied ::CUjit_cacheMode_enum.<br />
     * Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum<br />
     * Applies to: compiler only
     */
	public void setCacheMode(int cacheMode) {
		this.jitOptions.putInt(CU_JIT_CACHE_MODE, cacheMode);
	}

	public String getInfoLog() {
		byte[] infoLog = this.jitOptions.getBytes(CU_JIT_INFO_LOG_BUFFER);
		int infoLogSize = this.jitOptions.getInt(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
		if (infoLogSize <= 1)
			return "";
		return new String(infoLog, 0, infoLogSize - 1);
	}

	public String getErrorLog() {
		byte[] infoLog = this.jitOptions.getBytes(CU_JIT_ERROR_LOG_BUFFER);
		int infoLogSize = this.jitOptions.getInt(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
		return new String(infoLog, 0, infoLogSize - 1);
	}


	public static RuntimeLinkerOptions createDefaultOptions(boolean generateDebugInfo, CUdevice device) {
		RuntimeLinkerOptions options = new RuntimeLinkerOptions();
		if (device == null)
			options.setTargetFromContext();
		else
			options.setTarget(device);
		options.setGenerateLineInfo(true);
		options.prepareLogBuffers(2048, 2048);
		options.setGenerateDebugInfo(generateDebugInfo);
		return options;
	}

}
