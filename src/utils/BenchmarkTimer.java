package utils;

import java.io.PrintStream;
import java.util.ArrayList;

public class BenchmarkTimer {
	class TimeMarker {
		String label;
		long time;
	}
	ArrayList<TimeMarker> timeMarkers = new ArrayList<TimeMarker>(20);

	void addTimeMarker(String label) {
		TimeMarker marker = new TimeMarker();
		marker.time = System.currentTimeMillis();
		marker.label = label;
		this.timeMarkers.add(marker);
	}
	
	public void start(String caption) {
		this.timeMarkers.clear();
		addTimeMarker(caption);
	}

	public void record(String label) {
		addTimeMarker(label);
	}

	public void output(PrintStream stream) {
		stream.println(this.timeMarkers.get(0).label);
		long startTime = this.timeMarkers.get(0).time;
		stream.format("%-24s : %-8s  %-8s\n", "Label", "elapsed", "lap");
		stream.format("%-24s : %8d,\n", "Start", 0);
		for (int idx = 1; idx < timeMarkers.size(); ++idx) {
			TimeMarker prevMarker = this.timeMarkers.get(idx - 1);
			TimeMarker thisMarker = this.timeMarkers.get(idx);
			stream.format("%-24s : %8s, %8s\n",
					thisMarker.label, 
					thisMarker.time - startTime,
					thisMarker.time - prevMarker.time);
		}
		stream.println();
	}
	
}
