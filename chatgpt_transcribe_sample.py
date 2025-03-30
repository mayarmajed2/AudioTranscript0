import os
import base64
import json
import logging
import tempfile
import subprocess
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = None

try:
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")

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

def transcribe_with_openai(audio_file_path, language="ar-eg"):
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        audio_file_path (str): Path to the audio file
        language (str): Preferred language code (default: ar-eg for Egyptian Arabic)
        
    Returns:
        str: Transcribed text
    """
    if not client:
        raise ValueError("OpenAI client not initialized. Please check your API key.")
    
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    try:
        # Directly use the file for transcription with Whisper model
        with open(audio_file_path, "rb") as audio_file:
            logger.info(f"Starting transcription with Whisper API for {audio_file_path}")
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )
            return response
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {str(e)}")
        raise

def main():
    """Extract and transcribe a short sample from the audio file."""
    audio_file_path = "attached_assets/Voice 104.mp3"
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Create a temporary directory for the audio sample
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract a 30-second segment starting at 2 minutes
        sample_path = os.path.join(temp_dir, "audio_sample.mp3")
        
        print(f"Extracting a 30-second sample from {audio_file_path}...")
        if extract_audio_segment(audio_file_path, sample_path, start_time=120, duration=30):
            print(f"Sample extracted to {sample_path}")
            
            print("Starting transcription of the sample using OpenAI Whisper API...")
            
            try:
                transcription = transcribe_with_openai(sample_path)
                
                if transcription:
                    print("\n--- TRANSCRIPTION RESULT (30-SECOND SAMPLE) ---\n")
                    print(transcription)
                    print("\n----------------------------------------------\n")
                    
                    # Save to file
                    output_file = "sample_openai_transcription.txt"
                    with open(output_file, "w") as f:
                        f.write(transcription)
                    
                    print(f"Sample transcription saved to {output_file}")
                else:
                    print("Transcription failed: Empty result returned")
            except ValueError as ve:
                print(f"Configuration error: {str(ve)}")
                print("Please ensure you have set the OPENAI_API_KEY environment variable.")
            except Exception as e:
                print(f"Error during transcription: {str(e)}")
        else:
            print("Failed to extract audio sample")

if __name__ == "__main__":
    main()