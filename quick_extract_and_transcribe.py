import os
import sys
import time
import logging
import tempfile
import subprocess
import base64
from speech_recognition_fallback import fallback_transcribe, convert_to_wav

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio_segment(input_file, output_file, start_time_seconds, duration_seconds):
    """
    Extract a specific segment from an audio file using ffmpeg.
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to save the output audio segment
        start_time_seconds (int): Start time in seconds
        duration_seconds (int): Duration to extract in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-ss", str(start_time_seconds),
            "-t", str(duration_seconds),
            "-vn", output_file
        ]
        
        # Run ffmpeg command
        logger.info(f"Extracting segment from {start_time_seconds}s to {start_time_seconds + duration_seconds}s")
        subprocess.run(command, check=True, capture_output=True)
        
        # Verify the output file exists and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Segment extracted successfully to {output_file}")
            return True
        else:
            logger.error(f"Extraction failed: Output file is empty or does not exist")
            return False
            
    except Exception as e:
        logger.error(f"Error extracting audio segment: {str(e)}")
        return False

def main():
    """
    Extract a very short segment and transcribe with ChatGPT
    """
    # Path to the attached audio file
    audio_file_path = "attached_assets/Voice 104.mp3"
    
    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Extract a very short segment (5 seconds from the 15 minute mark)
    start_time = 900  # 15 minutes in seconds
    duration = 5    # 5 seconds for a very quick test
    
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_path = os.path.join(temp_dir, "audio_segment_15min_30sec.mp3")
        
        # Extract the segment
        if extract_audio_segment(audio_file_path, segment_path, start_time, duration):
            # Start transcription
            logger.info(f"Starting transcription with fallback method (google)...")
            
            # Measure the transcription time
            start_process_time = time.time()
            
            try:
                # Use fallback transcription (Google Speech or Sphinx)
                transcription = fallback_transcribe(segment_path, preferred_method="google")
                
                if transcription:
                    # Calculate processing time
                    processing_time = time.time() - start_process_time
                    logger.info(f"Transcription completed in {processing_time:.2f} seconds")
                                        
                    # Print results
                    print("\n=== SEGMENT TRANSCRIPTION (5sec from 15min mark) ===\n")
                    print(transcription)
                    print("\n=========================================\n")
                    
                    # Save to file
                    output_file = "transcription_15min_5sec.txt"
                    with open(output_file, "w") as f:
                        f.write("=== SEGMENT TRANSCRIPTION (5sec from 15min mark) ===\n\n")
                        f.write(transcription)
                    
                    logger.info(f"Results saved to {output_file}")
                else:
                    logger.error("Transcription failed: Empty result returned")
                    
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
        else:
            logger.error("Failed to extract the requested segment")

if __name__ == "__main__":
    main()