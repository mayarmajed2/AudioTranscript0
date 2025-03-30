import os
import logging
import time
import random
from openai import OpenAI
from audio_splitter import get_audio_duration, split_audio_file
from speech_recognition_fallback import fallback_transcribe

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with longer timeout (180 seconds instead of default 10)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
try:
    openai = OpenAI(api_key=OPENAI_API_KEY, timeout=180.0) if OPENAI_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    openai = None

# Constants for fallback settings
MAX_WHISPER_RETRIES = 8  # Increased maximum number of retries with Whisper before falling back
USE_FALLBACK_ON_ERROR = True  # Whether to use fallback when Whisper fails
DEFAULT_FALLBACK_METHOD = "google"  # Default fallback method (google or sphinx)

# Rate limit handling constants
RATE_LIMIT_MAX_RETRIES = 12  # Higher number of retries specifically for rate limit errors
RATE_LIMIT_INITIAL_DELAY = 5  # Longer initial delay for rate limit errors (in seconds)
RATE_LIMIT_BACKOFF_FACTOR = 4  # More aggressive backoff for rate limit errors

def _make_api_call_with_advanced_retry(api_call_func, max_retries=5, initial_retry_delay=2):
    """
    Helper function to make an API call with advanced retry logic.
    
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
    jitter_range = 0.5  # Add random jitter of ±50% to delay times
    
    for attempt in range(max_retries):
        try:
            return api_call_func()
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            retry_delay_with_jitter = retry_delay * (1 + random.uniform(-jitter_range, jitter_range))
            
            # Handle different types of errors based on error message content
            if "insufficient_quota" in error_msg:
                # Quota exceeded - no point in retrying
                logger.error(f"OpenAI API quota exceeded: {error_msg}")
                raise Exception("Your OpenAI API quota has been exceeded. Please check your billing details or try again later.")
                
            elif "429" in error_msg or "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                # Rate limit error (429) - apply longer backoff with more aggressive parameters
                # For rate limit errors, we use different max_retries
                if attempt < RATE_LIMIT_MAX_RETRIES - 1:
                    # Apply longer delay based on RATE_LIMIT constants
                    extended_delay = retry_delay_with_jitter * 2
                    # Cap the delay at 120 seconds
                    extended_delay = min(extended_delay, 120)
                    logger.warning(f"Rate limit exceeded: {error_msg}. Retrying in {extended_delay:.2f} seconds (attempt {attempt+1}/{RATE_LIMIT_MAX_RETRIES})...")
                    time.sleep(extended_delay)
                    # Increase delay more aggressively for rate limits using RATE_LIMIT_BACKOFF_FACTOR
                    retry_delay *= RATE_LIMIT_BACKOFF_FACTOR
                else:
                    logger.error(f"Too many retries due to rate limiting. Considering using fallback method.")
                    raise
            
            elif any(term in error_msg.lower() for term in ["connection", "timeout", "server", "unavailable", "500", "503"]):
                # Transient API errors - apply standard backoff
                if attempt < max_retries - 1:
                    logger.warning(f"API error: {error_msg}. Retrying in {retry_delay_with_jitter:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay_with_jitter)
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed due to API errors.")
                    raise
            
            else:
                # Unexpected errors - apply shorter backoff and fewer retries
                if attempt < min(2, max_retries - 1):  # Max 3 retries for unexpected errors
                    logger.warning(f"Unexpected error ({error_type}): {error_msg}. Retrying in {retry_delay_with_jitter:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay_with_jitter)
                    retry_delay *= 1.5
                else:
                    logger.error(f"Failed after {attempt+1} attempts due to unexpected error.")
                    raise
    
    # This should never be reached due to the raise statements above
    raise Exception(f"Failed after {max_retries} attempts with unknown error")

def transcribe_audio(audio_file_path, use_fallback_mode='auto'):
    """
    Transcribe an audio file using OpenAI's Whisper API with focus on Egyptian Arabic and English.
    For audio files longer than 15 minutes (900 seconds), the file will be automatically split 
    into smaller chunks, transcribed separately, and then combined.
    
    Args:
        audio_file_path (str): Path to the audio file
        use_fallback_mode (str): Fallback mode: 'auto' (default), 'always', or 'never'
        
    Returns:
        str: Transcribed text
    
    Raises:
        Exception: If transcription fails
    """
    logger.debug(f"Starting transcription process for file: {audio_file_path} with fallback mode: {use_fallback_mode}")
    
    # Use fallback right away if requested
    if use_fallback_mode == 'always':
        logger.info(f"Fallback mode set to 'always', using {DEFAULT_FALLBACK_METHOD} method directly")
        try:
            transcription = fallback_transcribe(audio_file_path, preferred_method=DEFAULT_FALLBACK_METHOD)
            if transcription:
                return f"[Transcribed using {DEFAULT_FALLBACK_METHOD} method as requested]\n\n{transcription}"
            else:
                raise Exception("Fallback transcription returned empty result")
        except Exception as e:
            logger.error(f"Direct fallback transcription failed: {str(e)}")
            raise Exception(f"Failed to transcribe audio with {DEFAULT_FALLBACK_METHOD}: {str(e)}")
    
    # Try using OpenAI's Whisper API
    if not OPENAI_API_KEY or openai is None:
        if use_fallback_mode == 'never':
            raise ValueError("OpenAI API key is missing and fallback is disabled. Please set the OPENAI_API_KEY environment variable.")
        else:
            logger.warning("OpenAI API key is missing or client failed to initialize, using fallback method instead")
            try:
                transcription = fallback_transcribe(audio_file_path, preferred_method=DEFAULT_FALLBACK_METHOD)
                if transcription:
                    return f"[Transcribed using fallback method due to missing API key]\n\n{transcription}"
                else:
                    raise Exception("Fallback transcription returned empty result")
            except Exception as e:
                logger.error(f"Fallback transcription failed: {str(e)}")
                raise Exception(f"Failed to transcribe audio with fallback method: {str(e)}")
    
    try:
        # Check if the audio file needs to be split (longer than 15 minutes)
        duration = get_audio_duration(audio_file_path)
        
        if duration > 900:  # 900 seconds = 15 minutes
            logger.info(f"Audio file is {duration:.2f} seconds long. Splitting into chunks...")
            
            # Split the audio file into 15-minute chunks
            audio_chunks = split_audio_file(audio_file_path)
            logger.info(f"Split audio into {len(audio_chunks)} chunks")
            
            # Transcribe each chunk and combine the results
            combined_transcription = ""
            
            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path}")
                
                # Define function to make the API call
                def transcribe_chunk():
                    if openai is None:
                        raise Exception("OpenAI client not available. Please check API key or use fallback method.")
                        
                    with open(chunk_path, "rb") as audio_file:
                        # Using Whisper-1 model which has good support for Egyptian Arabic
                        return openai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="ar-eg",  # Specify Egyptian Arabic as preferred language
                            response_format="text"
                        )
                
                # Use the advanced retry function
                try:
                    chunk_transcription = _make_api_call_with_advanced_retry(
                        transcribe_chunk, 
                        max_retries=MAX_WHISPER_RETRIES,
                        initial_retry_delay=RATE_LIMIT_INITIAL_DELAY
                    )
                    
                    # Add a delimiter between chunks for clarity if needed
                    if combined_transcription:
                        combined_transcription += "\n\n"
                    
                    combined_transcription += chunk_transcription
                    
                except Exception as chunk_error:
                    logger.error(f"Error transcribing chunk {i+1}: {str(chunk_error)}")
                    
                    # Try fallback for this chunk if fallback is not disabled
                    if use_fallback_mode != 'never' and USE_FALLBACK_ON_ERROR:
                        try:
                            logger.info(f"Attempting fallback for chunk {i+1}")
                            chunk_fallback = fallback_transcribe(chunk_path, preferred_method=DEFAULT_FALLBACK_METHOD)
                            
                            if combined_transcription:
                                combined_transcription += "\n\n"
                                
                            combined_transcription += f"[Chunk {i+1} transcribed with fallback method]\n{chunk_fallback}"
                            
                        except Exception as fallback_chunk_error:
                            logger.error(f"Fallback for chunk {i+1} also failed: {str(fallback_chunk_error)}")
                            
                            if combined_transcription:
                                combined_transcription += "\n\n"
                                
                            combined_transcription += f"[Transcription failed for chunk {i+1}]"
                    else:
                        # No fallback, just note the failure
                        if combined_transcription:
                            combined_transcription += "\n\n"
                            
                        combined_transcription += f"[Transcription failed for chunk {i+1}]"
            
            logger.debug("Transcription of all chunks completed")
            
            # If nothing was transcribed, raise an error
            if not combined_transcription.strip():
                raise Exception("All chunks failed to transcribe")
                
            return combined_transcription
            
        else:
            # For files shorter than 15 minutes, transcribe directly
            logger.info(f"Audio file is {duration:.2f} seconds long. Transcribing directly...")
            
            # Define function to make the API call
            def transcribe_file():
                if openai is None:
                    raise Exception("OpenAI client not available. Please check API key or use fallback method.")
                    
                with open(audio_file_path, "rb") as audio_file:
                    # Using Whisper-1 model which has good support for Egyptian Arabic
                    return openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="ar-eg",  # Specify Egyptian Arabic as preferred language
                        response_format="text"
                    )
            
            # Use the advanced retry function with improved parameters for rate limit handling
            response = _make_api_call_with_advanced_retry(
                transcribe_file,
                max_retries=MAX_WHISPER_RETRIES,
                initial_retry_delay=RATE_LIMIT_INITIAL_DELAY
            )
            
            logger.debug("Transcription completed successfully")
            return response
            
    except Exception as e:
        logger.error(f"Whisper transcription error: {str(e)}")
        error_message = str(e)
        
        # Try fallback methods if enabled
        if use_fallback_mode != 'never' and USE_FALLBACK_ON_ERROR:
            try:
                logger.info("Whisper transcription failed, using fallback method instead")
                logger.info(f"Attempting fallback transcription with {DEFAULT_FALLBACK_METHOD} method")
                
                # Use fallback transcription method
                transcription = fallback_transcribe(audio_file_path, preferred_method=DEFAULT_FALLBACK_METHOD)
                
                if transcription:
                    logger.info("Fallback transcription completed successfully")
                    return f"[Transcribed using fallback method due to Whisper API error]\n\n{transcription}"
                else:
                    logger.error("Fallback transcription returned empty result")
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {str(fallback_error)}")
                # Continue to the original error handling below
        
        # If fallback is disabled or also failed, provide detailed error messages
        if "insufficient_quota" in error_message:
            raise Exception(f"OpenAI API quota has been exceeded. Please check your billing information and try again later.")
        elif "429" in error_message or "Too Many Requests" in error_message:
            raise Exception(f"Failed to transcribe audio due to OpenAI API rate limits. Please try again in a few minutes.")
        elif "413" in error_message or "Payload Too Large" in error_message:
            raise Exception(f"Audio file is too large for the API. Please try a smaller file or split it manually.")
        elif "OpenAI API key is missing" in error_message and use_fallback_mode == 'never':
            raise Exception("OpenAI API key is missing and fallback is disabled. Please set the OPENAI_API_KEY environment variable.")
        else:
            raise Exception(f"Failed to transcribe audio with all available methods: {error_message}")
