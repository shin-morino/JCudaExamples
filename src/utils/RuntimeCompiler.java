package utils;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import utils.nvrtc.NvrtcException;
import utils.nvrtc.nvrtc;
import utils.nvrtc.nvrtc.NamedText;

public class RuntimeCompiler {
	NamedText source = null;
	ArrayList<NamedText> includes = new ArrayList<NamedText>();
	String[] options = null;
	
	byte[] ptx = null;
	String log = "";

	public void setSourceFile(String filePath) throws IOException {
		this.source = new NamedText(filePath, readFile(filePath));
	}

	public void setSourceString(byte[] code, String name) {
		this.source = new NamedText(name, nullTerminate(code));
	}

	public void setIncludeFile(String filePath) throws IOException {
		NamedText include = new NamedText(filePath, readFile(filePath));
		this.includes.add(include);
	}

	public void setIncludeString(byte[] code, String name){
		NamedText include = new NamedText(name, nullTerminate(code));
		this.includes.add(include);
	}

	public void setIncludeFile(String filePath, String name) throws IOException{
		NamedText include = new NamedText(name, readFile(filePath));
		this.includes.add(include);
	}
	
	public void setOptions(String[] options){
		this.options = options;
	}
	
	public void compile() {
		this.ptx = null;
		this.log = null;
		long prog = 0;

		try {
			prog = nvrtc.createProgram(this.source, this.includes);
		}
		catch (NvrtcException e) {
			showError(e, prog);
			throw e;
		}
		
		try {
			nvrtc.compileProgram(prog, this.options);
			this.ptx = nvrtc.getPTX(prog);
		}
		catch (NvrtcException e) {
			showError(e, prog);
			throw e;
		}
		finally {
			try {
				this.log = nvrtc.getProgramLog(prog);
			} catch (NvrtcException e) { }
			try {
				if (prog != 0)
					nvrtc.deestroyProgram(prog);
			} catch (NvrtcException e) { }
		}
	}

	public byte[] getPTX() {
		return this.ptx;
	}

	public String getProgramLog() {
		return this.log;
	}

	
	void showError(NvrtcException e, long prog) {
		e.printStackTrace();
		System.err.println(e.getMessage());
		try {
			String programLog = nvrtc.getProgramLog(prog);
			System.err.println(programLog);
		}
		catch (NvrtcException e1) {}
	}

	byte[] nullTerminate(byte[] source) {
		if (source[source.length - 1] == 0)
			return source;
		byte[] terminated = new byte[source.length + 1];
		System.arraycopy(source, 0, terminated, 0, source.length);
		terminated[source.length] = 0;
		return terminated;
	}

	byte[] readFile(String filePath) throws IOException {
		
		Path path = Paths.get(filePath);
		RandomAccessFile file = null;
		long size;
		try {
			size = Files.size(path);
			byte[] fileBody = new byte[(int)(size + 1)];
			file = new RandomAccessFile(path.toFile(), "r");
			file.read(fileBody);
			fileBody[(int)size] = 0;
			return fileBody;
		} catch (IOException e) {
			throw e;
		}
		finally {
			try {
				if (file != null)
					file.close();
			} catch (IOException e) {
			}
		}
	}
}
