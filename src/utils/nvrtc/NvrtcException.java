package utils.nvrtc;


/**
 * \ingroup error
 * \brief   The enumerated type nvrtcResult defines API call result codes.
 *          NVRTC API functions return nvrtcResult to indicate the call
 *          result.
 */

public class NvrtcException extends RuntimeException {
	private static final long serialVersionUID = 1L;

	public static final int NVRTC_SUCCESS = 0;
	public static final int NVRTC_ERROR_OUT_OF_MEMORY = 1;
	public static final int NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2;
	public static final int NVRTC_ERROR_INVALID_INPUT = 3;
	public static final int NVRTC_ERROR_INVALID_PROGRAM = 4;
	public static final int NVRTC_ERROR_INVALID_OPTION = 5;
	public static final int NVRTC_ERROR_COMPILATION = 6;
	public static final int NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7;

	@Override
	public String getMessage() {
		return nvrtc.getErrorString(id);
	}
	NvrtcException(final int id) {
		this.id = id;
	}
	final int id;
}
