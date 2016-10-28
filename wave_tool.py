from linescan import WaveFromFile

file_path = '/home/sebsikora/Documents/linescan_tool/test_images/good/wave_test/04082016 CONFLEC 05a_raw_G_event_4.png'
lines_per_second = 188.0
line_width = 0.0000002

wave = WaveFromFile(file_path = file_path, lines_per_second = lines_per_second, line_width = line_width)
status, top_results, bottom_results = wave.Analyse(display_plots = True, output_directory = '', background_span = 25, distance_threshold = 10, fit_window_span = 10, peak_threshold_multiplier = 15.0)
	
