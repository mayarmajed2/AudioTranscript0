import os
import logging
import speech_recognition as sr
import tempfile
import subprocess
from pydub import AudioSegment
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def convert_to_wav(audio_file_path, downsample=True):
    """
    Convert any audio file to WAV format for compatibility with speech_recognition.
    Ensures the WAV file is compatible with SpeechRecognition library.
    
    Args:
        audio_file_path (str): Path to the audio file
        downsample (bool): Whether to downsample the audio for faster processing
        
    Returns:
        str: Path to the converted WAV file
    """
    try:
        # Create temporary file for the WAV output
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()
        
        logger.debug(f"Converting {audio_file_path} to WAV format at {temp_wav_path}")
        
        # Always use FFmpeg for reliability across formats
        # Create a WAV file that's compatible with SpeechRecognition
        cmd = [
            "ffmpeg", 
            "-i", audio_file_path,
            "-ar", "16000",             # Sample rate: 16kHz (good for speech recognition)
            "-ac", "1",                 # Mono audio
            "-vn",                      # No video
            "-acodec", "pcm_s16le",     # PCM 16-bit little-endian format (best compatibility)
            "-f", "wav",                # Force WAV format output
            "-y",                       # Overwrite output file
            "-hide_banner",             # Reduce log clutter
            "-loglevel", "error",       # Only show errors
            temp_wav_path
        ]
        
        logger.debug(f"Running ffmpeg conversion: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Verify the file was created and has content
        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            logger.debug(f"FFmpeg conversion completed successfully. Output size: {os.path.getsize(temp_wav_path) / 1024:.2f} KB")
            return temp_wav_path
        else:
            raise Exception("FFmpeg created an empty WAV file")
            
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {str(e)}")
        
        # Try a different ffmpeg command as fallback
        try:
            logger.debug("Attempting alternative FFmpeg command...")
            alt_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            
            # Different set of parameters for maximum compatibility
            cmd = [
                "ffmpeg", 
                "-i", audio_file_path,
                "-ar", "44100",         # Standard 44.1kHz sample rate
                "-ac", "1",             # Mono
                "-acodec", "pcm_s16le", # PCM 16-bit
                "-y",                   # Overwrite
                alt_wav_path
            ]
            
            logger.debug(f"Running alternative ffmpeg conversion: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            
            if os.path.exists(alt_wav_path) and os.path.getsize(alt_wav_path) > 0:
                logger.debug(f"Alternative FFmpeg conversion successful. Output size: {os.path.getsize(alt_wav_path) / 1024:.2f} KB")
                
                # Clean up the first attempt if it exists
                if os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except:
                        pass
                        
                return alt_wav_path
            else:
                raise Exception("Alternative FFmpeg command created an empty WAV file")
                
        except Exception as e2:
            logger.error(f"Alternative conversion also failed: {str(e2)}")
            
            # Try a final approach with pydub if all else fails
            try:
                logger.debug("Attempting conversion with pydub as last resort...")
                pydub_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                
                # Force pydub to create a compatible WAV file
                sound = AudioSegment.from_file(audio_file_path)
                sound = sound.set_channels(1)  # Convert to mono
                sound = sound.set_frame_rate(16000)  # Convert to 16kHz
                sound.export(pydub_wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
                
                if os.path.exists(pydub_wav_path) and os.path.getsize(pydub_wav_path) > 0:
                    logger.debug(f"Pydub conversion successful. Output size: {os.path.getsize(pydub_wav_path) / 1024:.2f} KB")
                    return pydub_wav_path
                else:
                    raise Exception("Pydub created an empty WAV file")
                    
            except Exception as e3:
                logger.error(f"All conversion methods failed: {str(e)}, {str(e2)}, {str(e3)}")
                raise Exception(f"Failed to convert audio to WAV format after multiple attempts: {str(e)}")

def transcribe_with_sphinx(audio_file_path):
    """
    Transcribe audio using CMU Sphinx (offline, lower accuracy but works without internet).
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    recognizer = sr.Recognizer()
    
    # Convert to WAV if needed
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    wav_path = audio_file_path if file_ext == '.wav' else convert_to_wav(audio_file_path)
    
    try:
        logger.debug(f"Starting Sphinx transcription for {wav_path}")
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)  # Record the entire audio file
        
        # Use Sphinx for offline recognition
        text = recognizer.recognize_sphinx(audio_data)
        
        # Clean up temporary file if we created one
        if wav_path != audio_file_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        return text
    except Exception as e:
        logger.error(f"Sphinx transcription error: {str(e)}")
        
        # Clean up temporary file if we created one
        if wav_path != audio_file_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass
        
        raise Exception(f"Failed to transcribe with Sphinx: {str(e)}")

def transcribe_with_google(audio_file_path):
    """
    Transcribe audio using Google Speech Recognition (requires internet).
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    recognizer = sr.Recognizer()
    
    # Convert to WAV if needed
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    wav_path = audio_file_path if file_ext == '.wav' else convert_to_wav(audio_file_path)
    
    try:
        logger.debug(f"Starting Google Speech Recognition for {wav_path}")
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)  # Record the entire audio file
        
        # Use Google Speech Recognition (requires internet)
        text = recognizer.recognize_google(audio_data, language="ar-EG")  # Using Egyptian Arabic
        
        # Clean up temporary file if we created one
        if wav_path != audio_file_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        return text
    except Exception as e:
        logger.error(f"Google Speech Recognition error: {str(e)}")
        
        # Clean up temporary file if we created one
        if wav_path != audio_file_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass
        
        raise Exception(f"Failed to transcribe with Google Speech Recognition: {str(e)}")

def split_audio_for_recognition(audio_file_path, chunk_length_ms=120000):
    """
    Split audio file into smaller chunks for better processing with speech recognition.
    
    Args:
        audio_file_path (str): Path to the audio file
        chunk_length_ms (int): Length of each chunk in milliseconds (default: 120 seconds)
        
    Returns:
        list: List of paths to the chunked audio files
    """
    try:
        # Use ffmpeg directly for reliability with large files
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Get file duration using ffprobe
        try:
            cmd = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration_seconds = float(result.stdout.strip())
            logger.debug(f"Audio duration: {duration_seconds} seconds")
        except Exception as e:
            logger.warning(f"Failed to get duration with ffprobe: {e}, using default value")
            # Use a default large value if ffprobe fails
            duration_seconds = 7200  # Default to 2 hours

        # Calculate number of chunks (convert chunk_length_ms to seconds)
        chunk_length_sec = chunk_length_ms / 1000
        total_chunks = int(duration_seconds / chunk_length_sec) + 1  # Add 1 to ensure we cover the whole file
        logger.debug(f"Splitting audio into approximately {total_chunks} chunks of {chunk_length_sec} seconds each")
        
        # Use ffmpeg to create smaller chunks
        for i in range(total_chunks):
            # Calculate start time for this chunk
            start_time = i * chunk_length_sec
            
            # If we've gone beyond the file duration, stop
            if start_time >= duration_seconds:
                break
                
            # Create output path
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            chunk_paths.append(chunk_path)
            
            # Use ffmpeg to extract this chunk with parameters for speech recognition
            cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', audio_file_path,
                '-ss', str(start_time),
                '-t', str(chunk_length_sec),
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-vn',           # No video
                chunk_path
            ]
            
            logger.debug(f"Running ffmpeg to create chunk {i+1}/{total_chunks} at position {start_time}s")
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"Created chunk {i+1}/{total_chunks}: {chunk_path}")
        
        if not chunk_paths:
            raise Exception("No chunks were created from the audio file")
            
        return chunk_paths
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        raise Exception(f"Failed to split audio for recognition: {str(e)}")

def transcribe_large_audio(audio_file_path, method="google", max_chunks=25):
    """
    Transcribe large audio files by splitting into smaller chunks and combining results.
    For very large files, we limit the number of chunks to process.
    
    Args:
        audio_file_path (str): Path to the audio file
        method (str): Transcription method to use ("google" or "sphinx")
        max_chunks (int): Maximum number of chunks to process for very large files
        
    Returns:
        str: Combined transcription text
    """
    temp_dir = None
    
    try:
        # Create chunks
        chunk_paths = split_audio_for_recognition(audio_file_path)
        
        # Store the temp_dir path for cleanup
        if chunk_paths and len(chunk_paths) > 0:
            temp_dir = os.path.dirname(chunk_paths[0])
            
        # Limit number of chunks for extremely large files
        total_chunks = len(chunk_paths)
        logger.debug(f"Total chunks created: {total_chunks}")
        
        if total_chunks > max_chunks:
            logger.warning(f"Audio file generated {total_chunks} chunks, limiting to {max_chunks} for processing")
            # Take evenly spaced samples across the entire recording
            step = total_chunks // max_chunks
            # If step is 0, default to 1
            step = max(1, step)
            sampled_chunks = chunk_paths[::step]
            # Ensure we don't exceed max_chunks
            chunk_paths = sampled_chunks[:max_chunks]
            logger.debug(f"Reduced to {len(chunk_paths)} representative chunks")
        
        # Process each chunk
        transcriptions = []
        processed_chunks = len(chunk_paths)
        
        for i, chunk_path in enumerate(chunk_paths):
            logger.debug(f"Transcribing chunk {i+1}/{processed_chunks}")
            
            try:
                # Choose the transcription method
                if method.lower() == "google":
                    chunk_text = transcribe_with_google(chunk_path)
                else:  # default to sphinx
                    chunk_text = transcribe_with_sphinx(chunk_path)
                    
                transcriptions.append(chunk_text)
                
            except Exception as e:
                logger.error(f"Error transcribing chunk {i+1}: {str(e)}")
                transcriptions.append(f"[Transcription failed for segment {i+1}]")
                
            # Don't delete each chunk yet, as we want to handle cleanup in finally block
        
        # Combine all transcriptions with clear segmentation
        combined_text = ""
        for i, text in enumerate(transcriptions):
            if i > 0:
                combined_text += "\n\n"
            combined_text += text
        
        return combined_text
        
    except Exception as e:
        logger.error(f"Error in large audio transcription: {str(e)}")
        raise Exception(f"Failed to transcribe large audio: {str(e)}")
        
    finally:
        # Clean up all temporary files and directory
        try:
            if temp_dir and os.path.exists(temp_dir):
                logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")
            # Continue execution, don't raise exception for cleanup errors

def fallback_transcribe(audio_file_path, preferred_method="google"):
    """
    Main entry point for fallback transcription. Tries multiple methods if needed.
    
    Args:
        audio_file_path (str): Path to the audio file
        preferred_method (str): Preferred transcription method ("google" or "sphinx")
        
    Returns:
        str: Transcribed text
    """
    logger.info(f"Starting fallback transcription with {preferred_method} method")
    
    try:
        # For larger files (>10MB), use chunking approach
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        
        if file_size_mb > 10:
            logger.info(f"Audio file is large ({file_size_mb:.2f} MB), using chunking approach")
            return transcribe_large_audio(audio_file_path, method=preferred_method)
        
        # For smaller files, try direct transcription
        if preferred_method.lower() == "google":
            try:
                logger.info("Attempting Google Speech Recognition")
                return transcribe_with_google(audio_file_path)
            except Exception as google_error:
                logger.warning(f"Google transcription failed: {str(google_error)}")
                logger.info("Falling back to Sphinx")
                return transcribe_with_sphinx(audio_file_path)
        else:
            try:
                logger.info("Attempting Sphinx transcription")
                return transcribe_with_sphinx(audio_file_path)
            except Exception as sphinx_error:
                logger.warning(f"Sphinx transcription failed: {str(sphinx_error)}")
                logger.info("Falling back to Google Speech Recognition")
                return transcribe_with_google(audio_file_path)
    
    except Exception as e:
        logger.error(f"All fallback transcription methods failed: {str(e)}")
        raise Exception(f"Unable to transcribe audio with any available method: {str(e)}")