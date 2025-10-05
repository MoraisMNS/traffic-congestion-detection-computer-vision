# frontend/batch_processing_mode.py
from batch_processor import process_video_batch, generate_comparative_analysis

class BatchProcessingMode:
    def __init__(self):
        pass

    def run(self, input_folder, output_folder, analyze=False):
        if analyze:
            generate_comparative_analysis(output_folder)
        else:
            process_video_batch(input_folder, output_folder)
