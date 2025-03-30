import os
import logging
from chatgpt_transcribe import transcribe_with_chatgpt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test transcription of the attached audio file."""
    audio_file_path = "attached_assets/Voice 104.mp3"
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    print(f"Starting direct transcription of {audio_file_path} using OpenAI...")
    
    try:
        transcription = transcribe_with_chatgpt(audio_file_path)
        
        if transcription:
            print("\n--- TRANSCRIPTION RESULT ---\n")
            print(transcription)
            print("\n---------------------------\n")
            
            # Save to file
            output_file = "voice_104_transcription.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            print(f"Transcription saved to {output_file}")
        else:
            print("Transcription failed: Empty result returned")
    except ValueError as ve:
        print(f"Configuration error: {str(ve)}")
        print("Please ensure you have set the OPENAI_API_KEY environment variable.")
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()