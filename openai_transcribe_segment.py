import os
import sys
import time
import logging
import tempfile
import subprocess
from text_analysis import generate_summary, extract_key_points

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

def transcribe_with_openai(audio_file_path, language="ar-eg"):
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        audio_file_path (str): Path to the audio file
        language (str): Preferred language code (default: ar-eg for Egyptian Arabic)
        
    Returns:
        str: Transcribed text
    """
    try:
        import openai
        
        # Check if OPENAI_API_KEY is set
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return None
        
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # First, check if file exists and is not too large
        if not os.path.exists(audio_file_path):
            logger.error(f"File not found: {audio_file_path}")
            return None
            
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        if file_size_mb > 25:
            logger.error(f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed by the API (25 MB)")
            return None
            
        logger.info(f"Starting transcription with Whisper API for {audio_file_path}")
        
        start_time = time.time()
        
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language=language,
                response_format="text"
            )
            
        duration = time.time() - start_time
        logger.info(f"Transcription completed in {duration:.2f} seconds")
        
        # Return transcribed text from the response
        return response.text
        
    except Exception as e:
        logger.error(f"Error during transcription with OpenAI: {str(e)}")
        return None
        
def convert_time_to_seconds(time_str):
    """
    Convert time string in format MM:SS or HH:MM:SS to seconds.
    
    Args:
        time_str (str): Time string in format MM:SS or HH:MM:SS
        
    Returns:
        int: Time in seconds
    """
    parts = time_str.split(":")
    if len(parts) == 2:  # MM:SS format
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS format
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS or HH:MM:SS")

def parse_time_argument(arg):
    """
    Parse time argument which can be in seconds or MM:SS format.
    
    Args:
        arg (str): Time argument as string
        
    Returns:
        int: Time in seconds
    """
    try:
        if ":" in arg:  # Format is MM:SS or HH:MM:SS
            return convert_time_to_seconds(arg)
        else:  # Format is seconds
            return int(arg)
    except Exception as e:
        logger.error(f"Error parsing time argument '{arg}': {str(e)}")
        return None

def main():
    """
    Extract and transcribe a segment from an audio file with OpenAI's Whisper model.
    """
    # Path to the attached audio file
    audio_file_path = "attached_assets/Voice 104.mp3"
    
    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file not found at {audio_file_path}")
        return
    
    if len(sys.argv) < 3:
        print("Usage: python openai_transcribe_segment.py start_time duration [language]")
        print("Example 1: python openai_transcribe_segment.py 900 60")
        print("Example 2: python openai_transcribe_segment.py 15:00 1:00")
        print("Example 3: python openai_transcribe_segment.py 15:00 1:00 en")
        return
        
    # Parse arguments
    start_time_arg = sys.argv[1]
    duration_arg = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "ar-eg"
    
    # Convert to seconds
    start_time = parse_time_argument(start_time_arg)
    duration = parse_time_argument(duration_arg)
    
    if start_time is None or duration is None:
        return
    
    # Generate descriptive file names
    start_time_str = start_time_arg.replace(":", "_")
    duration_str = duration_arg.replace(":", "_")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a segment filename based on the start time and duration
        segment_path = os.path.join(temp_dir, f"audio_segment_{start_time_str}_to_{duration_str}.mp3")
        
        # Extract the segment
        if extract_audio_segment(audio_file_path, segment_path, start_time, duration):
            # Start transcription
            logger.info(f"Starting transcription with OpenAI Whisper for language: {language}...")
            
            # Measure the transcription time
            start_process_time = time.time()
            
            try:
                # Use OpenAI for transcription
                transcription = transcribe_with_openai(segment_path, language=language)
                
                if transcription:
                    # Calculate processing time
                    processing_time = time.time() - start_process_time
                    logger.info(f"Transcription completed in {processing_time:.2f} seconds")
                    
                    # Generate summary
                    logger.info("Generating summary and key points...")
                    summary = generate_summary(transcription)
                    key_points = extract_key_points(transcription)
                    
                    # Create output filenames with timestamp information
                    segment_description = f"{start_time_str}_for_{duration_str}"
                    output_file = f"transcription_{segment_description}.txt"
                    
                    # Print results
                    print(f"\n=== SEGMENT TRANSCRIPTION ({start_time_arg} for {duration_arg}) ===\n")
                    print(transcription)
                    print("\n=========================================\n")
                    
                    print("\n=== SUMMARY ===\n")
                    print(summary)
                    print("\n===============\n")
                    
                    print("\n=== KEY POINTS ===\n")
                    for point in key_points:
                        print(f"- {point}")
                    print("\n=================\n")
                    
                    # Save to file
                    with open(output_file, "w") as f:
                        f.write(f"=== SEGMENT TRANSCRIPTION ({start_time_arg} for {duration_arg}) ===\n\n")
                        f.write(transcription)
                        f.write("\n\n=== SUMMARY ===\n\n")
                        f.write(summary)
                        f.write("\n\n=== KEY POINTS ===\n\n")
                        for point in key_points:
                            f.write(f"- {point}\n")
                    
                    logger.info(f"Results saved to {output_file}")
                else:
                    logger.error("Transcription failed: Empty result returned")
                    
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
        else:
            logger.error("Failed to extract the requested segment")

if __name__ == "__main__":
    main()