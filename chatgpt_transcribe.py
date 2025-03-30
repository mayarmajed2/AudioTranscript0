import os
import base64
import json
import logging
import time
import random
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Rate limit handling constants
RATE_LIMIT_MAX_RETRIES = 12  # Higher number of retries specifically for rate limit errors
RATE_LIMIT_INITIAL_DELAY = 5  # Longer initial delay for rate limit errors (in seconds)
RATE_LIMIT_BACKOFF_FACTOR = 4  # More aggressive backoff for rate limit errors
MAX_DELAY = 120  # Maximum delay between retries (2 minutes)

# Initialize OpenAI client with API key and longer timeout
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = None

try:
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=180.0)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")

def _make_api_call_with_retry(api_call_func, max_retries=8, initial_retry_delay=5):
    """
    Helper function to make an API call with advanced retry logic for rate limits.
    
    Args:
        api_call_func (function): Function that makes the actual API call
        max_retries (int): Maximum number of retry attempts
        initial_retry_delay (int): Initial delay in seconds before retrying
        
    Returns:
        Any: Response from the API call
        
    Raises:
        Exception: If all retry attempts fail
    """
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            return api_call_func()
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle rate limit errors (429)
            if "429" in error_msg or "too many requests" in error_msg or "rate limit" in error_msg:
                if attempt < max_retries - 1:
                    # Add jitter to retry delay for rate limit errors
                    jitter = random.uniform(-0.2, 0.2)
                    actual_delay = retry_delay * (1 + jitter)
                    # Cap the maximum delay
                    actual_delay = min(actual_delay, MAX_DELAY)
                    logger.warning(f"Rate limit exceeded. Retrying in {actual_delay:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(actual_delay)
                    # Increase delay more aggressively for rate limits
                    retry_delay *= RATE_LIMIT_BACKOFF_FACTOR
                else:
                    logger.error(f"Failed after {max_retries} attempts due to rate limiting.")
                    raise Exception("OpenAI API rate limit exceeded. Please try again later.")
            
            # Handle quota exceeded errors - no point in retrying
            elif "insufficient_quota" in error_msg or "quota" in error_msg:
                logger.error(f"OpenAI API quota exceeded: {error_msg}")
                raise Exception("Your OpenAI API quota has been exceeded. Please check your billing details.")
            
            # Handle other errors with shorter backoff
            else:
                if attempt < max_retries - 1:
                    jitter = random.uniform(-0.5, 0.5)
                    actual_delay = retry_delay * (1 + jitter)
                    logger.warning(f"API error: {error_msg}. Retrying in {actual_delay:.2f} seconds...")
                    time.sleep(actual_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed after {max_retries} attempts: {error_msg}")
                    raise
    
    # This should never be reached
    raise Exception(f"Failed after {max_retries} attempts with unknown error")

def audio_to_base64(file_path):
    """Convert audio file to base64 encoding."""
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding audio file to base64: {str(e)}")
        raise

def transcribe_with_chatgpt(audio_file_path, language="ar-eg"):
    """
    Transcribe audio file using ChatGPT with audio input capability.
    Uses advanced retry logic to handle rate limits and API errors.
    For large files, it attempts chunk-based transcription with more robust error handling.
    
    Args:
        audio_file_path (str): Path to the audio file
        language (str or None): Preferred language code (default: ar-eg for Egyptian Arabic).
                              Set to None for automatic language detection.
        
    Returns:
        str: Transcribed text
    """
    if not client:
        raise ValueError("OpenAI client not initialized. Please check your API key.")
    
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Check file size to determine approach
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    
    # For large files, we might want to use a different approach
    is_large_file = file_size_mb > 20  # Consider files > 20MB as "large"
    
    if is_large_file:
        logger.info(f"Large file detected ({file_size_mb:.2f} MB). Using chunked approach with increased resilience.")
        try:
            # Import here to avoid circular imports
            from audio_splitter import split_audio_file
            from speech_recognition_fallback import fallback_transcribe
            
            # Try to split the file
            try:
                audio_chunks = split_audio_file(audio_file_path, max_duration_seconds=300)  # 5 minute chunks
                logger.info(f"Split audio into {len(audio_chunks)} chunks for more reliable processing")
                
                # Process each chunk with retries
                transcriptions = []
                for i, chunk_path in enumerate(audio_chunks):
                    logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}: {chunk_path}")
                    
                    # Try OpenAI first
                    for attempt in range(3):  # 3 attempts per chunk
                        try:
                            # Define inner transcription function for this chunk
                            def make_chunk_transcription_call():
                                with open(chunk_path, "rb") as audio_file:
                                    kwargs = {
                                        "model": "whisper-1",
                                        "file": audio_file,
                                        "response_format": "text"
                                    }
                                    
                                    # Only add language parameter if it's specified
                                    if language is not None:
                                        kwargs["language"] = language
                                        
                                    return client.audio.transcriptions.create(**kwargs)
                            
                            # Try OpenAI transcription with our retry logic
                            chunk_text = _make_api_call_with_retry(
                                make_chunk_transcription_call,
                                max_retries=RATE_LIMIT_MAX_RETRIES,
                                initial_retry_delay=RATE_LIMIT_INITIAL_DELAY
                            )
                            transcriptions.append(chunk_text)
                            break  # Success, break the attempt loop
                            
                        except Exception as e:
                            error_msg = str(e).lower()
                            if attempt < 2:  # Last attempt will fallback
                                if "429" in error_msg or "rate limit" in error_msg:
                                    logger.warning(f"Rate limit hit on chunk {i+1}. Waiting before retry...")
                                    time.sleep(10 * (attempt + 1))  # Progressive delay
                                else:
                                    logger.warning(f"Failed to transcribe chunk {i+1} (attempt {attempt+1}): {error_msg}")
                            else:
                                # On final attempt, try fallback method
                                logger.warning(f"OpenAI transcription failed for chunk {i+1}, trying fallback method")
                                try:
                                    fallback_text = fallback_transcribe(chunk_path, preferred_method="google")
                                    transcriptions.append(f"[Fallback transcription for segment {i+1}]: {fallback_text}")
                                except Exception as fallback_error:
                                    logger.error(f"Fallback also failed for chunk {i+1}: {str(fallback_error)}")
                                    # Add placeholder for failed chunk
                                    transcriptions.append(f"[Failed to transcribe segment {i+1}]")
                
                # Combine all transcriptions
                combined_text = "\n\n".join(transcriptions)
                
                # Clean up chunks
                for chunk_path in audio_chunks:
                    try:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                    except Exception as e:
                        logger.warning(f"Could not clean up chunk file {chunk_path}: {str(e)}")
                
                return combined_text
                
            except Exception as split_error:
                logger.error(f"Error splitting audio file: {str(split_error)}")
                # Fall back to standard approach if splitting fails
                logger.info("Falling back to standard approach after split failure")
            
        except ImportError as ie:
            logger.error(f"Could not import needed modules for chunked processing: {str(ie)}")
            logger.info("Continuing with standard approach")
    
    # Standard approach for smaller files or if chunking failed
    logger.info(f"Using standard approach for transcribing: {audio_file_path}")
    
    # Define the API call function to be used with retry logic
    def make_transcription_call():
        with open(audio_file_path, "rb") as audio_file:
            logger.info(f"Starting transcription with Whisper API for {audio_file_path}")
            # Create transcription request with proper parameters
            kwargs = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "text"
            }
            
            # Only add language parameter if it's specified (not None)
            if language is not None:
                kwargs["language"] = language
                
            # Make the API call
            return client.audio.transcriptions.create(**kwargs)
    
    try:
        # Use the retry function to handle potential API errors
        logger.info(f"Transcribing audio file: {audio_file_path} with language: {language}")
        return _make_api_call_with_retry(
            make_transcription_call,
            max_retries=RATE_LIMIT_MAX_RETRIES,
            initial_retry_delay=RATE_LIMIT_INITIAL_DELAY
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during Whisper transcription: {error_msg}")
        
        # Try fallback method before giving up
        if "429" in error_msg or "too many requests" in error_msg.lower() or "rate limit" in error_msg.lower():
            # Before giving up on rate limit, try one last fallback
            try:
                logger.warning("Rate limit exceeded, trying fallback transcription method")
                from speech_recognition_fallback import fallback_transcribe
                fallback_result = fallback_transcribe(audio_file_path, preferred_method="google")
                return f"[Transcribed using fallback method due to API rate limits]\n\n{fallback_result}"
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {str(fallback_error)}")
                # Re-raise with more user-friendly message
                raise Exception(f"OpenAI API rate limit exceeded and fallback transcription failed.")
                
        # Standard error handling
        elif "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            raise Exception(f"OpenAI API quota exceeded. Please check your billing information.")
        else:
            # For other errors
            try:
                # Try fallback for other errors as well
                logger.warning(f"OpenAI transcription failed with error: {error_msg}. Trying fallback transcription method")
                from speech_recognition_fallback import fallback_transcribe
                fallback_result = fallback_transcribe(audio_file_path, preferred_method="google")
                return f"[Transcribed using fallback method due to API error]\n\n{fallback_result}"
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {str(fallback_error)}")
                # If all else fails
                raise Exception(f"Transcription failed and fallback transcription also failed. Original error: {error_msg}")

def main():
    """Main function to transcribe the audio."""
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
            output_file = "openai_transcription.txt"
            with open(output_file, "w") as f:
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