import os
import sys
import time
import logging
import argparse
from speech_recognition_fallback import fallback_transcribe
from text_analysis import generate_summary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Quick transcription script for audio files with optional summarization.
    Uses the fallback transcription method directly, with optimized performance.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quickly transcribe an audio file using fallback methods.')
    parser.add_argument('--file', '-f', default="attached_assets/Voice 104.mp3", 
                        help='Path to the audio file (default: attached_assets/Voice 104.mp3)')
    parser.add_argument('--method', '-m', choices=['google', 'sphinx'], default='google',
                        help='Preferred transcription method (default: google)')
    parser.add_argument('--summarize', '-s', action='store_true',
                        help='Generate a summary of the transcription')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (default: transcription_result.txt)')
    
    args = parser.parse_args()
    audio_file_path = args.file
    preferred_method = args.method
    summarize = args.summarize
    output_file = args.output or "transcription_result.txt"
    
    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Show a progress message
    logger.info(f"Starting transcription of {audio_file_path}...")
    logger.info(f"Using fallback method: {preferred_method.upper()}")
    
    start_time = time.time()
    
    try:
        # Use fallback transcription directly
        transcription = fallback_transcribe(audio_file_path, preferred_method=preferred_method)
        
        if transcription:
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            # Generate summary if requested
            summary = None
            if summarize:
                logger.info("Generating summary...")
                summary_start_time = time.time()
                summary = generate_summary(transcription)
                summary_time = time.time() - summary_start_time
                logger.info(f"Summary generated in {summary_time:.2f} seconds")
            
            # Print results
            print("\n=== TRANSCRIPTION RESULT ===\n")
            print(transcription)
            print("\n===========================\n")
            
            if summary:
                print("\n=== SUMMARY ===\n")
                print(summary)
                print("\n===============\n")
            
            # Save to file
            with open(output_file, "w") as f:
                f.write("=== TRANSCRIPTION ===\n\n")
                f.write(transcription)
                if summary:
                    f.write("\n\n=== SUMMARY ===\n\n")
                    f.write(summary)
            
            logger.info(f"Results saved to {output_file}")
        else:
            logger.error("Transcription failed: Empty result returned")
            
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
if __name__ == "__main__":
    main()