import os
import logging
import glob
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import uuid
from werkzeug.utils import secure_filename
import tempfile
import shutil

from transcription import transcribe_audio
from text_analysis import extract_key_points, extract_text_from_pdf, generate_summary
from file_utils import allowed_audio_file, allowed_document_file
from audio_splitter import get_audio_duration, extract_audio_segment
from chatgpt_transcribe import transcribe_with_chatgpt

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-secret-key")

# Configure maximum content length for uploads (200MB)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Set up temporary file storage
# Make sure the temporary directory exists and is accessible
try:
    TEMP_FOLDER = tempfile.mkdtemp()
    # Verify we can write to this directory
    test_file = os.path.join(TEMP_FOLDER, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    app.config['TEMP_FOLDER'] = TEMP_FOLDER
    logger.debug(f"Temporary folder created at: {TEMP_FOLDER}")
except Exception as e:
    # Fallback to a local directory if /tmp is not accessible
    logger.error(f"Error setting up temp folder: {str(e)}")
    TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    app.config['TEMP_FOLDER'] = TEMP_FOLDER
    logger.debug(f"Using fallback temporary folder: {TEMP_FOLDER}")

# 200MB max upload size (already set above)

@app.route('/')
def index():
    """Render the home page with the upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and process them."""
    # Check if audio file was uploaded
    if 'audio_file' not in request.files:
        flash('No audio file provided', 'danger')
        return redirect(request.url)
    
    audio_file = request.files['audio_file']
    
    # Check if the user submitted an empty form
    if audio_file.filename == '':
        flash('No audio file selected', 'danger')
        return redirect(request.url)
    
    # Check if the file format is allowed
    if not allowed_audio_file(audio_file.filename):
        flash('Invalid audio file format. Allowed formats: MP3, WAV, M4A, OGG', 'danger')
        return redirect(request.url)
    
    # Check if a fallback method was specified
    use_fallback = request.form.get('use_fallback', 'auto')
    fallback_method = request.form.get('fallback_method', 'google')
    
    logger.debug(f"Transcription settings - Fallback Mode: {use_fallback}, Fallback Method: {fallback_method}")
    
    # Process lecture document if provided
    lecture_text = None
    lecture_path = ""  # Initialize as empty string instead of None
    if 'lecture_file' in request.files and request.files['lecture_file'].filename != '':
        lecture_file = request.files['lecture_file']
        if not allowed_document_file(lecture_file.filename):
            flash('Invalid document file format. Allowed formats: PDF, TXT', 'danger')
            return redirect(request.url)
        
        # Save lecture file
        lecture_filename = secure_filename(lecture_file.filename)
        lecture_path = os.path.join(app.config['TEMP_FOLDER'], lecture_filename)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(lecture_path), exist_ok=True)
        logger.debug(f"Saving lecture file to: {lecture_path}")
        lecture_file.save(lecture_path)
        
        # Verify the file was saved successfully
        if not os.path.exists(lecture_path):
            logger.error(f"Failed to save lecture file at {lecture_path}")
            flash('Error saving lecture file. Please try again.', 'danger')
            return redirect(url_for('index'))
        
        # Extract text from lecture file
        try:
            if lecture_filename.lower().endswith('.pdf'):
                lecture_text = extract_text_from_pdf(lecture_path)
            else:
                with open(lecture_path, 'rb') as f:
                    lecture_text = f.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting text from lecture file: {str(e)}")
            flash(f'Error processing lecture file: {str(e)}', 'danger')
            return redirect(request.url)
    
    # Generate a unique ID for this processing job
    job_id = str(uuid.uuid4())
    
    # Save the audio file
    audio_filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['TEMP_FOLDER'], f"{job_id}_{audio_filename}")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    logger.debug(f"Saving audio file to: {audio_path}")
    audio_file.save(audio_path)
    
    # Verify the file was saved successfully
    if not os.path.exists(audio_path):
        logger.error(f"Failed to save audio file at {audio_path}")
        flash('Error saving audio file. Please try again.', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Import here to modify fallback settings based on form values
        import transcription
        
        # Update fallback settings based on form input
        if use_fallback == 'always':
            transcription.USE_FALLBACK_ON_ERROR = True
        elif use_fallback == 'never':
            transcription.USE_FALLBACK_ON_ERROR = False
        # 'auto' will use the default setting
        
        # Update fallback method if specified
        if fallback_method in ['google', 'sphinx']:
            transcription.DEFAULT_FALLBACK_METHOD = fallback_method
        
        # Check if OpenAI API key is available
        if not transcription.OPENAI_API_KEY and use_fallback != 'always':
            logger.warning("OpenAI API key is not set. Suggesting fallback method.")
            flash('OpenAI API key is not available. The application will use the fallback transcription method instead.', 'warning')
            # Force to use fallback if API key is missing
            use_fallback = 'always'
        
        # Transcribe the audio file
        logger.debug(f"Starting transcription for {audio_path} with fallback mode: {use_fallback}")
        try:
            transcription_result = transcription.transcribe_audio(audio_path, use_fallback_mode=use_fallback)
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            flash(f'Error during transcription: {str(e)}', 'danger')
            return redirect(url_for('index'))
        
        if not transcription_result:
            logger.error("Transcription returned empty result")
            flash('Error processing audio file: Transcription failed. The file may be corrupted or too large.', 'danger')
            return redirect(url_for('index'))
            
        logger.debug(f"Transcription completed successfully, length: {len(transcription_result)}")
        
        # Extract key points
        logger.debug("Extracting key points")
        try:
            key_points = extract_key_points(transcription_result, lecture_text)
            
            if not key_points:
                logger.error("Key points extraction returned empty result")
                key_points = ["No key points could be extracted"]
        except Exception as e:
            logger.error(f"Key points extraction error: {str(e)}")
            key_points = ["Failed to extract key points, using basic summarization instead"]
            # Try to use the fallback method directly
            try:
                from text_analysis import _fallback_key_point_extraction
                fallback_points = _fallback_key_point_extraction(transcription_result, lecture_text)
                if fallback_points:
                    key_points = fallback_points
            except Exception as fallback_error:
                logger.error(f"Fallback extraction error: {str(fallback_error)}")
        
        # Generate summary
        logger.debug("Generating summary")
        try:
            summary = generate_summary(transcription_result)
            if not summary:
                logger.error("Summary generation returned empty result")
                summary = "Could not generate a summary for this recording."
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            summary = "Failed to generate a summary. The recording may be too short or contain unclear speech."
            try:
                from text_analysis import _fallback_summary_generation
                fallback_summary = _fallback_summary_generation(transcription_result)
                if fallback_summary:
                    summary = fallback_summary
            except Exception as fallback_error:
                logger.error(f"Fallback summary generation error: {str(fallback_error)}")
        
        # Store results in session
        session['transcription'] = transcription_result
        session['key_points'] = key_points
        session['summary'] = summary
        session['audio_filename'] = audio_filename
        
        # Store information about which method was used for transcription
        method_str = ""
        if "[Transcribed using fallback method" in transcription_result:
            method_str = "Fallback (automatic)"
            flash('Whisper API transcription failed. Used fallback method instead.', 'warning')
        elif "[Transcribed using google method as requested]" in transcription_result:
            method_str = "Google Speech (selected)"
            flash('Used Google Speech Recognition as requested.', 'info')
        elif "[Transcribed using sphinx method as requested]" in transcription_result:
            method_str = "CMU Sphinx (selected)"
            flash('Used CMU Sphinx speech recognition as requested.', 'info')
        elif "[Transcribed using fallback method due to missing API key]" in transcription_result:
            method_str = "Fallback (no API key)"
            flash('OpenAI API key not available. Used fallback method instead.', 'warning')
        else:
            method_str = "OpenAI Whisper API"
            
        session['transcription_method'] = method_str
        
        # Clean up temporary files
        try:
            # Check if the file is longer than 15 minutes, if so, it may have generated chunks in temp directories
            audio_duration = get_audio_duration(audio_path)
            if audio_duration > 900:  # 15 minutes
                logger.debug("Cleaning up temporary chunk directories...")
                # Find all temp directories that might have been created for audio chunks
                temp_dirs = glob.glob(os.path.join('/tmp', 'tmp*'))
                for temp_dir in temp_dirs:
                    if os.path.isdir(temp_dir):
                        try:
                            # Look for audio chunk files
                            chunk_files = glob.glob(os.path.join(temp_dir, 'chunk_*'))
                            if chunk_files:
                                logger.debug(f"Found {len(chunk_files)} chunk files in {temp_dir}")
                                # Remove each chunk file
                                for chunk_file in chunk_files:
                                    os.remove(chunk_file)
                                # Try to remove the directory
                                os.rmdir(temp_dir)
                        except Exception as e:
                            logger.error(f"Error cleaning up chunk directory {temp_dir}: {str(e)}")
            
            # Remove original audio file
            if os.path.exists(audio_path):
                logger.debug(f"Removing audio file: {audio_path}")
                os.remove(audio_path)
            else:
                logger.debug(f"Audio file not found during cleanup: {audio_path}")
            
            # Remove lecture file if it exists
            if lecture_path and os.path.exists(lecture_path):
                logger.debug(f"Removing lecture file: {lecture_path}")
                os.remove(lecture_path)
            elif lecture_path:
                logger.debug(f"Lecture file not found during cleanup: {lecture_path}")
        except Exception as e:
            logger.error(f"Error removing temporary files: {str(e)}")
        
        return redirect(url_for('results'))
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing error: {error_msg}")
        
        # Provide more user-friendly error messages
        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
            flash('OpenAI API quota exceeded. Please try again later or check your API key billing details.', 'danger')
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            flash('OpenAI API rate limit reached. Please try again in a few minutes.', 'danger')
        else:
            flash(f'Error processing files: {error_msg}', 'danger')
            
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display the transcription, summary and key points results."""
    if 'transcription' not in session or 'key_points' not in session:
        flash('No results found. Please upload files first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template(
        'results.html',
        transcription=session['transcription'],
        key_points=session['key_points'],
        summary=session.get('summary', 'Summary not available.'),
        audio_filename=session.get('audio_filename', 'Unknown file'),
        transcription_method=session.get('transcription_method', 'Whisper API')
    )

@app.route('/download/<content_type>')
def download(content_type):
    """Generate and return downloadable content."""
    if content_type not in ['transcription', 'key_points', 'summary', 'all']:
        flash('Invalid download type', 'danger')
        return redirect(url_for('results'))
    
    if 'transcription' not in session or 'key_points' not in session:
        flash('No results found. Please upload files first.', 'warning')
        return redirect(url_for('index'))
    
    content = ""
    filename = "audio_analysis.txt"
    
    if content_type == 'transcription':
        content = session['transcription']
        filename = "transcription.txt"
    elif content_type == 'key_points':
        content = "\n".join([f"- {point}" for point in session['key_points']])
        filename = "key_points.txt"
    elif content_type == 'summary':
        content = session.get('summary', 'Summary not available.')
        filename = "summary.txt"
    else:  # 'all'
        content = f"SUMMARY:\n\n{session.get('summary', 'Summary not available.')}\n\n"
        content += f"KEY POINTS:\n\n"
        content += "\n".join([f"- {point}" for point in session['key_points']])
        content += f"\n\nTRANSCRIPTION:\n\n{session['transcription']}"
    
    return jsonify({
        'content': content,
        'filename': filename
    })

@app.route('/extract_segment', methods=['POST'])
def extract_segment():
    """Extract and transcribe a specific segment of an audio file."""
    if 'audio_file' not in request.files:
        flash('No audio file provided', 'danger')
        return redirect(url_for('index'))
    
    audio_file = request.files['audio_file']
    
    # Check if the user submitted an empty form
    if audio_file.filename == '':
        flash('No audio file selected', 'danger')
        return redirect(url_for('index'))
    
    # Check if the file format is allowed
    if not allowed_audio_file(audio_file.filename):
        flash('Invalid audio file format. Allowed formats: MP3, WAV, M4A, OGG', 'danger')
        return redirect(url_for('index'))
    
    # Get segment parameters
    try:
        start_time = int(request.form.get('start_time', 0))
        if start_time < 0:
            start_time = 0
            flash('Start time adjusted to 0 seconds (beginning of file)', 'warning')
    except ValueError:
        start_time = 0
        flash('Invalid start time, using 0 seconds instead', 'warning')
    
    try:
        duration = int(request.form.get('duration', 60))
        if duration < 5:
            duration = 5
            flash('Duration adjusted to minimum of 5 seconds', 'warning')
        elif duration > 300:
            duration = 300
            flash('Duration adjusted to maximum of 300 seconds (5 minutes)', 'warning')
    except ValueError:
        duration = 60
        flash('Invalid duration, using 60 seconds instead', 'warning')
    
    # Check fallback options
    use_fallback = request.form.get('use_fallback', 'auto')
    fallback_method = request.form.get('fallback_method', 'sphinx')
    
    # Generate a unique ID for this extraction
    job_id = str(uuid.uuid4())
    
    # Save the audio file
    audio_filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['TEMP_FOLDER'], f"{job_id}_{audio_filename}")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    audio_file.save(audio_path)
    
    # Verify the file was saved successfully
    if not os.path.exists(audio_path):
        logger.error(f"Failed to save audio file at {audio_path}")
        flash('Error saving audio file. Please try again.', 'danger')
        return redirect(url_for('index'))
    
    # Create a temporary directory for the segment
    extract_dir = os.path.join(app.config['TEMP_FOLDER'], f"segment_{job_id}")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Create segment path
    segment_path = os.path.join(extract_dir, f"segment_{start_time}to{start_time+duration}.mp3")
    
    try:
        # Check if the file is long enough for the requested segment
        audio_duration = get_audio_duration(audio_path)
        
        if start_time >= audio_duration:
            flash(f'Start time ({start_time}s) exceeds audio duration ({int(audio_duration)}s)', 'danger')
            return redirect(url_for('index'))
        
        # Adjust duration if needed
        if start_time + duration > audio_duration:
            adjusted_duration = int(audio_duration - start_time)
            logger.info(f"Adjusting duration from {duration}s to {adjusted_duration}s to fit within audio file")
            flash(f'Duration adjusted from {duration}s to {adjusted_duration}s to fit within audio file length', 'warning')
            duration = adjusted_duration
        
        # Extract the segment
        logger.info(f"Extracting segment from {start_time}s to {start_time + duration}s")
        
        if not extract_audio_segment(audio_path, segment_path, start_time, duration):
            flash('Failed to extract the requested segment', 'danger')
            return redirect(url_for('index'))
        
        # Transcribe the segment
        import transcription
        
        # Update fallback settings based on form input
        if use_fallback == 'always':
            transcription.USE_FALLBACK_ON_ERROR = True
        elif use_fallback == 'never':
            transcription.USE_FALLBACK_ON_ERROR = False
        
        # Update fallback method if specified
        if fallback_method in ['google', 'sphinx']:
            transcription.DEFAULT_FALLBACK_METHOD = fallback_method
        
        # Check if OpenAI API key is available
        if not transcription.OPENAI_API_KEY and use_fallback != 'always':
            logger.warning("OpenAI API key is not set. Suggesting fallback method.")
            flash('OpenAI API key is not available. The application will use the fallback transcription method instead.', 'warning')
            use_fallback = 'always'
        
        try:
            logger.info(f"Starting transcription for segment with fallback mode: {use_fallback}")
            transcription_result = transcription.transcribe_audio(segment_path, use_fallback_mode=use_fallback)
            
            if not transcription_result:
                flash('Transcription failed: Empty result returned', 'danger')
                return redirect(url_for('index'))
            
            # Generate a summary
            summary = generate_summary(transcription_result)
            
            # Extract key points
            key_points = extract_key_points(transcription_result)
            
            # Format start and end times for display
            start_formatted = f"{start_time // 60}:{start_time % 60:02d}"
            end_time = start_time + duration
            end_formatted = f"{end_time // 60}:{end_time % 60:02d}"
            
            # Store results in session
            session['transcription'] = transcription_result
            session['key_points'] = key_points
            session['summary'] = summary
            session['audio_filename'] = f"{audio_filename} (Segment {start_formatted}-{end_formatted})"
            session['segment_info'] = f"Segment from {start_formatted} to {end_formatted}"
            
            # Store information about which method was used
            method_str = ""
            if "[Transcribed using fallback method" in transcription_result:
                method_str = "Fallback (automatic)"
            elif "[Transcribed using google method as requested]" in transcription_result:
                method_str = "Google Speech (selected)"
            elif "[Transcribed using sphinx method as requested]" in transcription_result:
                method_str = "CMU Sphinx (selected)"
            elif "[Transcribed using fallback method due to missing API key]" in transcription_result:
                method_str = "Fallback (no API key)"
            else:
                method_str = "OpenAI Whisper API"
                
            session['transcription_method'] = method_str
            
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(segment_path):
                os.remove(segment_path)
            if os.path.exists(extract_dir):
                try:
                    os.rmdir(extract_dir)
                except:
                    pass
            
            return redirect(url_for('results'))
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            flash(f'Error during transcription: {str(e)}', 'danger')
            return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Error extracting segment: {str(e)}")
        flash(f'Error extracting segment: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/extract')
def extract_page():
    """Render the page for extracting segments from audio files."""
    return render_template('extract.html')

@app.route('/direct', methods=['GET'])
def direct_page():
    """Render the page for direct transcription with OpenAI."""
    return render_template('direct_transcribe.html')

@app.route('/direct', methods=['POST'])
def direct_transcribe():
    """Handle direct transcription with OpenAI Whisper."""
    # Check if audio file was uploaded
    if 'audio_file' not in request.files:
        flash('No audio file provided', 'danger')
        return redirect(url_for('direct_page'))
    
    audio_file = request.files['audio_file']
    
    # Check if the user submitted an empty form
    if audio_file.filename == '':
        flash('No audio file selected', 'danger')
        return redirect(url_for('direct_page'))
    
    # Check if the file format is allowed
    if not allowed_audio_file(audio_file.filename):
        flash('Invalid audio file format. Allowed formats: MP3, WAV, M4A, OGG', 'danger')
        return redirect(url_for('direct_page'))
    
    # Get language preference
    language = request.form.get('language', 'ar-eg')
    compress_audio = 'compress_audio' in request.form
    
    # Generate a unique ID for this processing job
    job_id = str(uuid.uuid4())
    
    # Save the audio file
    audio_filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['TEMP_FOLDER'], f"{job_id}_{audio_filename}")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    logger.debug(f"Saving audio file to: {audio_path}")
    audio_file.save(audio_path)
    
    # Verify the file was saved successfully
    if not os.path.exists(audio_path):
        logger.error(f"Failed to save audio file at {audio_path}")
        flash('Error saving audio file. Please try again.', 'danger')
        return redirect(url_for('direct_page'))
    
    try:
        # Compress audio if requested and file is larger than 1MB
        processed_audio_path = audio_path
        if compress_audio and os.path.getsize(audio_path) > 1024 * 1024:  # 1MB
            try:
                from pydub import AudioSegment
                compressed_path = os.path.join(app.config['TEMP_FOLDER'], f"{job_id}_compressed_{audio_filename}")
                
                # Load audio and export with compression
                audio = AudioSegment.from_file(audio_path)
                # Export with 64kbps bitrate mono
                audio = audio.set_channels(1)
                audio.export(compressed_path, format="mp3", bitrate="64k")
                
                if os.path.exists(compressed_path):
                    # Use compressed file instead
                    processed_audio_path = compressed_path
                    logger.info(f"Compressed audio from {os.path.getsize(audio_path)/1024/1024:.2f}MB to {os.path.getsize(compressed_path)/1024/1024:.2f}MB")
                    flash(f"Audio file compressed from {os.path.getsize(audio_path)/1024/1024:.1f}MB to {os.path.getsize(compressed_path)/1024/1024:.1f}MB for faster processing", 'info')
            except Exception as e:
                logger.error(f"Audio compression error: {str(e)}")
                flash("Could not compress audio file. Proceeding with original file.", 'warning')
        
        # Direct transcription with OpenAI Whisper via ChatGPT
        try:
            logger.info(f"Starting direct transcription with OpenAI Whisper for {processed_audio_path}")
            
            # Set language parameter to None if set to "auto" for auto-detection
            lang_param = None if language == "auto" else language
            
            # Display user-friendly processing message
            flash('Processing audio file - this may take up to a minute for longer files. Please wait...', 'info')
            
            try:
                # Pass language parameter correctly
                if language == "auto":
                    transcription_result = transcribe_with_chatgpt(processed_audio_path, language=None)
                else:
                    transcription_result = transcribe_with_chatgpt(processed_audio_path, language=language)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Transcription error: {error_msg}")
                
                # Provide more user-friendly error messages and attempt fallback
                if "429" in error_msg or "too many requests" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning("Rate limit reached, switching to fallback transcription method")
                    flash('OpenAI API rate limit reached. Switching to fallback transcription method automatically...', 'warning')
                    
                    try:
                        # Attempt fallback transcription
                        from speech_recognition_fallback import fallback_transcribe
                        transcription_result = fallback_transcribe(processed_audio_path, preferred_method="google")
                        flash('Transcription completed using fallback method due to API limits.', 'info')
                        
                        # Update transcription method to show fallback was used
                        session['transcription_method'] = "Fallback Method (Google) - OpenAI API Rate Limited"
                    except Exception as fallback_err:
                        logger.error(f"Fallback transcription also failed: {str(fallback_err)}")
                        flash(f'Fallback transcription also failed. Please try again later. Error: {str(fallback_err)}', 'danger')
                        return redirect(url_for('direct_page'))
                        
                elif "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
                    flash('OpenAI API quota exceeded. Please check your API key billing details or contact support.', 'danger')
                    return redirect(url_for('direct_page'))
                else:
                    flash(f'Error during transcription: {error_msg}', 'danger')
                    return redirect(url_for('direct_page'))
            
            if not transcription_result:
                logger.error("Transcription returned empty result")
                flash('Error processing audio file: Transcription failed. The file may be corrupted or too large.', 'danger')
                return redirect(url_for('direct_page'))
                
            logger.debug(f"Direct transcription completed successfully, length: {len(transcription_result)}")
            
            # Extract key points 
            key_points = extract_key_points(transcription_result)
            
            # Generate summary
            summary = generate_summary(transcription_result)
            
            # Store results in session
            session['transcription'] = transcription_result
            session['key_points'] = key_points
            session['summary'] = summary
            session['audio_filename'] = audio_filename
            session['transcription_method'] = "OpenAI Whisper API (Direct)"
            
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            
            return redirect(url_for('results'))
            
        except Exception as e:
            logger.error(f"Direct transcription error: {str(e)}")
            flash(f'Error during transcription: {str(e)}', 'danger')
            return redirect(url_for('direct_page'))
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing error: {error_msg}")
        
        # Provide more user-friendly error messages
        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
            flash('OpenAI API quota exceeded. Please try again later or check your API key billing details.', 'danger')
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            flash('OpenAI API rate limit reached. Please try again in a few minutes.', 'danger')
        else:
            flash(f'Error processing files: {error_msg}', 'danger')
            
        return redirect(url_for('direct_page'))

@app.route('/clear')
def clear_session():
    """Clear the session data and return to the home page."""
    session.clear()
    flash('Session cleared. You can upload new files now.', 'info')
    return redirect(url_for('index'))

# Clean up temp directory when the app exits
@app.teardown_appcontext
def cleanup_temp_folder(exception):
    """Remove temporary folder on application teardown."""
    try:
        if os.path.exists(TEMP_FOLDER) and os.path.isdir(TEMP_FOLDER):
            logger.debug(f"Cleaning up temp folder: {TEMP_FOLDER}")
            shutil.rmtree(TEMP_FOLDER)
        else:
            logger.debug(f"Temp folder not found during cleanup: {TEMP_FOLDER}")
    except Exception as e:
        logger.error(f"Error cleaning up temp folder: {str(e)}")

@app.route('/attached_files')
def attached_files_page():
    """Render the page for processing attached audio files."""
    # Get list of attached audio files
    attached_files = []
    attached_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attached_assets')
    
    if os.path.exists(attached_files_dir) and os.path.isdir(attached_files_dir):
        for file in os.listdir(attached_files_dir):
            if allowed_audio_file(file):
                attached_files.append(file)
    
    return render_template('attached_files.html', attached_files=attached_files)

@app.route('/process_attached_file', methods=['POST'])
def process_attached_file():
    """Process an attached audio file."""
    # Check if file name was provided
    if 'attached_file' not in request.form or not request.form['attached_file']:
        flash('No attached file selected', 'danger')
        return redirect(url_for('attached_files_page'))
    
    attached_filename = request.form['attached_file']
    attached_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attached_assets', attached_filename)
    
    # Validate file exists
    if not os.path.exists(attached_file_path):
        flash('Selected file not found on server', 'danger')
        return redirect(url_for('attached_files_page'))
    
    # Check if the file format is allowed
    if not allowed_audio_file(attached_filename):
        flash('Invalid audio file format. Allowed formats: MP3, WAV, M4A, OGG', 'danger')
        return redirect(url_for('attached_files_page'))
    
    # Get fallback settings
    use_fallback = request.form.get('use_fallback', 'auto')
    fallback_method = request.form.get('fallback_method', 'google')
    
    logger.debug(f"Processing attached file: {attached_filename}")
    logger.debug(f"Transcription settings - Fallback Mode: {use_fallback}, Fallback Method: {fallback_method}")
    
    try:
        # Import here to modify fallback settings based on form values
        import transcription
        
        # Update fallback settings based on form input
        if use_fallback == 'always':
            transcription.USE_FALLBACK_ON_ERROR = True
        elif use_fallback == 'never':
            transcription.USE_FALLBACK_ON_ERROR = False
        # 'auto' will use the default setting
        
        # Update fallback method if specified
        if fallback_method in ['google', 'sphinx']:
            transcription.DEFAULT_FALLBACK_METHOD = fallback_method
        
        # Check if OpenAI API key is available
        if not transcription.OPENAI_API_KEY and use_fallback != 'always':
            logger.warning("OpenAI API key is not set. Suggesting fallback method.")
            flash('OpenAI API key is not available. The application will use the fallback transcription method instead.', 'warning')
            # Force to use fallback if API key is missing
            use_fallback = 'always'
        
        # Transcribe directly using the improved method that handles large files better
        logger.debug(f"Starting transcription for attached file: {attached_file_path}")
        try:
            # Improved method: try using chatgpt_transcribe which now has better chunking for large files
            from chatgpt_transcribe import transcribe_with_chatgpt
            
            try:
                logger.info(f"Using improved chatgpt_transcribe method first")
                transcription_result = transcribe_with_chatgpt(attached_file_path)
                logger.info("Successfully transcribed with chatgpt_transcribe")
            except Exception as chatgpt_err:
                logger.warning(f"chatgpt_transcribe failed, falling back to standard method: {str(chatgpt_err)}")
                transcription_result = transcription.transcribe_audio(attached_file_path, use_fallback_mode=use_fallback)
        except Exception as e:
            logger.error(f"Both transcription methods failed: {str(e)}")
            flash(f'Error during transcription: {str(e)}', 'danger')
            return redirect(url_for('attached_files_page'))
        
        if not transcription_result:
            logger.error("Transcription returned empty result")
            flash('Error processing audio file: Transcription failed. The file may be corrupted or too large.', 'danger')
            return redirect(url_for('attached_files_page'))
            
        logger.debug(f"Transcription completed successfully, length: {len(transcription_result)}")
        
        # Extract key points and generate summary
        from text_analysis import extract_key_points, generate_summary
        
        # Extract key points
        logger.debug("Extracting key points")
        try:
            key_points = extract_key_points(transcription_result)
            
            if not key_points:
                logger.error("Key points extraction returned empty result")
                key_points = ["No key points could be extracted"]
        except Exception as e:
            logger.error(f"Key points extraction error: {str(e)}")
            key_points = ["Failed to extract key points"]
            try:
                from text_analysis import _fallback_key_point_extraction
                fallback_points = _fallback_key_point_extraction(transcription_result)
                if fallback_points:
                    key_points = fallback_points
            except Exception as fallback_error:
                logger.error(f"Fallback extraction error: {str(fallback_error)}")
        
        # Generate summary
        logger.debug("Generating summary")
        try:
            summary = generate_summary(transcription_result)
            if not summary:
                logger.error("Summary generation returned empty result")
                summary = "Could not generate a summary for this recording."
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            summary = "Failed to generate a summary."
            try:
                from text_analysis import _fallback_summary_generation
                fallback_summary = _fallback_summary_generation(transcription_result)
                if fallback_summary:
                    summary = fallback_summary
            except Exception as fallback_error:
                logger.error(f"Fallback summary generation error: {str(fallback_error)}")
        
        # Store results in session
        session['transcription'] = transcription_result
        session['key_points'] = key_points
        session['summary'] = summary
        session['audio_filename'] = attached_filename
        
        # Determine which method was used
        if "[Transcribed using fallback method" in transcription_result:
            method_str = "Fallback (automatic)"
        elif "[Transcribed using google method as requested]" in transcription_result:
            method_str = "Google Speech (selected)"
        elif "[Transcribed using sphinx method as requested]" in transcription_result:
            method_str = "CMU Sphinx (selected)"
        elif "[Transcribed using fallback method due to missing API key]" in transcription_result:
            method_str = "Fallback (no API key)"
        elif "[Transcribed using fallback method due to API rate limits]" in transcription_result:
            method_str = "Fallback (rate limited)"
        elif "[Transcribed using fallback method due to API error]" in transcription_result:
            method_str = "Fallback (API error)"
        else:
            method_str = "OpenAI Whisper API"
            
        session['transcription_method'] = method_str
        
        return redirect(url_for('results'))
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Processing error: {error_msg}")
        
        # User-friendly error messages
        if "quota" in error_msg.lower() or "insufficient_quota" in error_msg:
            flash('OpenAI API quota exceeded. Please try again later or check your API key billing details.', 'danger')
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            flash('OpenAI API rate limit reached. Please try again in a few minutes.', 'danger')
        else:
            flash(f'Error processing attached file: {error_msg}', 'danger')
            
        return redirect(url_for('attached_files_page'))
