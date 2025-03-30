import os
import logging
import tempfile
import subprocess
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_audio_duration(audio_file_path):
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    try:
        # First try using ffprobe which is faster for large files
        try:
            logger.debug(f"Getting audio duration using ffprobe for {audio_file_path}")
            cmd = [
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.debug(f"ffprobe reported duration: {duration} seconds")
            return duration
        except Exception as ffprobe_error:
            logger.warning(f"ffprobe failed ({str(ffprobe_error)}), falling back to pydub")
            
            # Fallback to pydub
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio) / 1000  # pydub uses milliseconds
            logger.debug(f"pydub reported duration: {duration_seconds} seconds")
            return duration_seconds
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        # Return a default duration if there's an error
        return 0

def split_audio_file(audio_file_path, max_duration_seconds=900):
    """
    Split an audio file into smaller chunks if it exceeds the maximum duration.
    
    Args:
        audio_file_path (str): Path to the audio file
        max_duration_seconds (int): Maximum duration in seconds (default: 900s = 15min)
        
    Returns:
        list: List of paths to the split audio files
    """
    try:
        # Get file duration
        duration = get_audio_duration(audio_file_path)
        
        # If the file is shorter than the max duration, return the original path
        if duration <= max_duration_seconds:
            logger.debug(f"Audio file is {duration}s, shorter than max {max_duration_seconds}s, not splitting")
            return [audio_file_path]
        
        # Calculate how many chunks we need
        num_chunks = int((duration + max_duration_seconds - 0.1) // max_duration_seconds)  # Ceiling division
        logger.debug(f"Audio file is {duration}s, will be split into {num_chunks} parts")
        
        # Create a temporary directory for the chunks
        temp_dir = tempfile.mkdtemp()
        
        # For large files, use ffmpeg directly which is more efficient
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        
        if file_size_mb > 20:  # For files larger than 20MB, use ffmpeg
            logger.debug(f"File is large ({file_size_mb:.2f} MB), using ffmpeg for splitting")
            return split_with_ffmpeg(audio_file_path, temp_dir, max_duration_seconds, num_chunks)
        else:
            # For smaller files, use pydub
            logger.debug(f"File is small ({file_size_mb:.2f} MB), using pydub for splitting")
            return split_with_pydub(audio_file_path, temp_dir, max_duration_seconds)
    
    except Exception as e:
        logger.error(f"Error splitting audio file: {str(e)}")
        # Return the original file path if there's an error
        return [audio_file_path]

def split_with_ffmpeg(audio_file_path, temp_dir, max_duration_seconds, num_chunks):
    """
    Split audio file using ffmpeg, which is more efficient for large files.
    
    Args:
        audio_file_path (str): Path to the audio file
        temp_dir (str): Temporary directory to store chunks
        max_duration_seconds (int): Maximum duration of each chunk in seconds
        num_chunks (int): Number of chunks to create
        
    Returns:
        list: List of paths to the chunked audio files
    """
    try:
        # Get file extension from original file
        filename, ext = os.path.splitext(os.path.basename(audio_file_path))
        if not ext or ext.lower() not in ['.mp3', '.wav', '.m4a', '.ogg']:
            ext = ".mp3"  # Default to mp3 if unsupported extension
            
        format_name = ext.lstrip('.')
        
        # Prepare chunk paths
        chunk_paths = []
        
        # Process each chunk
        for i in range(num_chunks):
            start_time = i * max_duration_seconds
            
            # Create output path
            chunk_path = os.path.join(temp_dir, f"chunk_{i+1}{ext}")
            chunk_paths.append(chunk_path)
            
            # For the last chunk, don't specify duration to get the remainder of the file
            if i == num_chunks - 1:
                cmd = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-i', audio_file_path,
                    '-ss', str(start_time),
                    '-c', 'copy',  # Copy the codec, don't re-encode
                    chunk_path
                ]
            else:
                cmd = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-i', audio_file_path,
                    '-ss', str(start_time),
                    '-t', str(max_duration_seconds),
                    '-c', 'copy',  # Copy the codec, don't re-encode
                    chunk_path
                ]
            
            logger.debug(f"Running ffmpeg to create chunk {i+1}/{num_chunks}")
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            
        return chunk_paths
    
    except Exception as e:
        logger.error(f"Error splitting with ffmpeg: {str(e)}")
        raise

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
            "-hide_banner", "-loglevel", "error",
            "-i", input_file,
            "-ss", str(start_time_seconds),
            "-t", str(duration_seconds),
            "-vn", output_file
        ]
        
        # Run ffmpeg command
        logger.debug(f"Extracting segment from {start_time_seconds}s to {start_time_seconds + duration_seconds}s")
        result = subprocess.run(command, check=True, capture_output=True)
        
        # Verify the output file exists and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.debug(f"Segment extracted successfully to {output_file}")
            return True
        else:
            logger.error(f"Extraction failed: Output file is empty or does not exist")
            return False
            
    except Exception as e:
        logger.error(f"Error extracting audio segment: {str(e)}")
        return False

def split_with_pydub(audio_file_path, temp_dir, max_duration_seconds):
    """
    Split audio file using pydub.
    
    Args:
        audio_file_path (str): Path to the audio file
        temp_dir (str): Temporary directory to store chunks
        max_duration_seconds (int): Maximum duration of each chunk in seconds
        
    Returns:
        list: List of paths to the chunked audio files
    """
    try:
        audio = AudioSegment.from_file(audio_file_path)
        
        # Get total duration in milliseconds
        total_duration_ms = len(audio)
        max_duration_ms = max_duration_seconds * 1000
        
        # Calculate how many chunks we need
        num_chunks = (total_duration_ms + max_duration_ms - 1) // max_duration_ms  # Ceiling division
        
        # Split the audio file
        chunk_paths = []
        for i in range(num_chunks):
            start_ms = i * max_duration_ms
            end_ms = min((i + 1) * max_duration_ms, total_duration_ms)
            
            chunk = audio[start_ms:end_ms]
            
            # Get file extension from original file
            _, ext = os.path.splitext(audio_file_path)
            if not ext:
                ext = ".wav"  # Default to wav if no extension
            
            # Create a path for the chunk
            chunk_path = os.path.join(temp_dir, f"chunk_{i+1}{ext}")
            
            # Export the chunk
            chunk.export(chunk_path, format=ext.lstrip('.'))
            chunk_paths.append(chunk_path)
            
            logger.debug(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
        
        return chunk_paths
    
    except Exception as e:
        logger.error(f"Error splitting with pydub: {str(e)}")
        raise