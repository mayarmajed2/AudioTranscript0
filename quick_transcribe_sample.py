import os
import sys
import logging
import tempfile
import subprocess
from speech_recognition_fallback import fallback_transcribe, convert_to_wav

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_audio_segment(input_file, output_file, start_time=0, duration=60):
    """
    Extract a segment from an audio file using ffmpeg.
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to save the output audio segment
        start_time (int): Start time in seconds
        duration (int): Duration to extract in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-ss", str(start_time),
            "-t", str(duration),
            "-vn", output_file
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        return True
    except Exception as e:
        logger.error(f"Error extracting audio segment: {str(e)}")
        return False

def main():
    """
    Quick transcription script for a sample of the attached audio file.
    Accepts command line arguments for start time and duration.
    """
    # Path to the attached audio file
    audio_file_path = "attached_assets/Voice 104.mp3"
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Parse command line arguments
    start_time = 120  # Default: 2 minutes
    duration = 60     # Default: 60 seconds
    
    if len(sys.argv) > 1:
        try:
            start_time = int(sys.argv[1])
            print(f"Using provided start time: {start_time} seconds")
        except ValueError:
            print(f"Invalid start time: {sys.argv[1]}. Using default: {start_time} seconds")
    
    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
            print(f"Using provided duration: {duration} seconds")
        except ValueError:
            print(f"Invalid duration: {sys.argv[2]}. Using default: {duration} seconds")
    
    # Create a temporary directory for the audio sample
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a descriptive filename
        sample_path = os.path.join(temp_dir, f"audio_sample_{start_time}to{start_time+duration}.mp3")
        
        print(f"Extracting a {duration}-second sample starting at {start_time} seconds from {audio_file_path}...")
        if extract_audio_segment(audio_file_path, sample_path, start_time=start_time, duration=duration):
            print(f"Sample extracted to {sample_path}")
            
            print("Starting transcription of the sample...")
            print("Using fallback method (Google Speech Recognition or CMU Sphinx)")
            
            try:
                # Convert to WAV format for better compatibility
                wav_path = convert_to_wav(sample_path)
                
                # Use fallback transcription directly
                transcription = fallback_transcribe(wav_path, preferred_method="google")
                
                if transcription:
                    print(f"\n--- TRANSCRIPTION RESULT ({start_time}s to {start_time+duration}s) ---\n")
                    print(transcription)
                    print("\n---------------------------\n")
                    
                    # Save to file
                    output_file = f"sample_transcription_{start_time}to{start_time+duration}.txt"
                    with open(output_file, "w") as f:
                        f.write(transcription)
                    
                    print(f"Sample transcription saved to {output_file}")
                else:
                    print("Transcription failed: Empty result returned")
                    
            except Exception as e:
                print(f"Error during transcription: {str(e)}")
        else:
            print("Failed to extract audio sample")
        
if __name__ == "__main__":
    main()