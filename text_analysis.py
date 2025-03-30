import os
import logging
import PyPDF2
import re
import time
import random
import json
from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize OpenAI client with longer timeout
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
try:
    openai = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0) if OPENAI_API_KEY else None
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    openai = None

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def clean_text(text):
    """
    Clean and normalize text for better comparison.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Arabic characters
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    # Convert to lowercase for better matching (only affects English)
    text = text.lower()
    return text.strip()

def _api_call_with_retry(api_func, max_retries=5, initial_retry_delay=2):
    """
    Helper function to make an OpenAI API call with enhanced retry logic.
    
    Args:
        api_func (function): Function that makes the API call
        max_retries (int): Maximum number of retry attempts
        initial_retry_delay (int): Initial delay in seconds before retrying
        
    Returns:
        Any: The API response
        
    Raises:
        Exception: If all retry attempts fail
    """
    retry_delay = initial_retry_delay
    jitter_range = 0.5  # Add random jitter of ±50% to delay times
    
    for attempt in range(max_retries):
        try:
            return api_func()
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            retry_delay_with_jitter = retry_delay * (1 + random.uniform(-jitter_range, jitter_range))
            
            # Handle rate limit errors
            if "429" in error_msg or "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    extended_delay = retry_delay_with_jitter * 2
                    logger.warning(f"Rate limit exceeded: {error_msg}. Retrying in {extended_delay:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(extended_delay)
                    retry_delay *= 3  # More aggressive backoff for rate limits
                else:
                    logger.error(f"All {max_retries} attempts failed due to rate limiting.")
                    raise
            
            # Handle server errors
            elif any(term in error_msg.lower() for term in ["connection", "timeout", "server", "unavailable", "500", "503"]):
                if attempt < max_retries - 1:
                    logger.warning(f"API error: {error_msg}. Retrying in {retry_delay_with_jitter:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay_with_jitter)
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed due to API errors.")
                    raise
            
            # Handle other errors
            else:
                if attempt < min(2, max_retries - 1):  # Fewer retries for unexpected errors
                    logger.warning(f"Unexpected error ({error_type}): {error_msg}. Retrying in {retry_delay_with_jitter:.2f} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay_with_jitter)
                    retry_delay *= 1.5
                else:
                    logger.error(f"Failed after {attempt+1} attempts due to unexpected error.")
                    raise
    
    raise Exception(f"Failed after {max_retries} attempts with unknown error")

def extract_key_points(transcription, lecture_text=None):
    """
    Extract key points from transcription, with focus on points not in lecture material.
    Uses an improved retry mechanism to handle API rate limits and transient errors.
    
    Args:
        transcription (str): Transcribed text from audio
        lecture_text (str, optional): Text from lecture materials
        
    Returns:
        list: List of key points
    """
    logger.debug("Starting key points extraction")
    
    if not OPENAI_API_KEY or openai is None:
        logger.warning("OpenAI API key is missing or client failed to initialize. Using fallback method for key point extraction.")
        return _fallback_key_point_extraction(transcription, lecture_text)
    
    try:
        # Prepare the prompt based on whether lecture text is available
        if lecture_text:
            prompt = f"""
            I have an audio transcription and lecture material text. I need you to:
            1. Identify the key points from the transcription.
            2. Focus specifically on points that are NOT mentioned in the lecture material.
            3. Return ONLY the points that add new information beyond what's in the lecture material.
            4. Format your response as a JSON array of strings, with each string being a key point.
            5. Be precise and concise in extracting these unique points.
            6. Pay special attention to Egyptian Arabic expressions and technical terms.
            
            Transcription:
            {transcription}
            
            Lecture Material:
            {lecture_text}
            
            Return only the JSON array of key points not in the lecture material.
            """
        else:
            prompt = f"""
            I have an audio transcription. I need you to:
            1. Identify and summarize the key points from the transcription.
            2. Format your response as a JSON array of strings, with each string being a key point.
            3. Be precise and concise in extracting these points.
            4. Pay special attention to Egyptian Arabic expressions and technical terms.
            
            Transcription:
            {transcription}
            
            Return only the JSON array of key points.
            """
        
        # Define the API call function
        def make_openai_request():
            # Check if the OpenAI client is available
            if openai is None:
                raise ValueError("OpenAI client is not available. Please check your API key.")
                
            # Use the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            return openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in extracting key information from transcriptions, with special focus on Egyptian Arabic content."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
        
        # Use the retry mechanism for the API call
        response = _api_call_with_retry(
            make_openai_request,
            max_retries=5,
            initial_retry_delay=3
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Check if the result contains a key points array
        if isinstance(result, dict) and 'key_points' in result:
            key_points = result['key_points']
        elif isinstance(result, list):
            key_points = result
        else:
            # If the structure is unexpected, try to extract array from any field
            for key, value in result.items():
                if isinstance(value, list):
                    key_points = value
                    break
            else:
                # If no list found, convert the entire result to a list of strings
                key_points = [str(item) for item in result.values()]
        
        logger.debug(f"Extracted {len(key_points)} key points")
        return key_points
    
    except Exception as e:
        logger.error(f"Error extracting key points: {str(e)}")
        error_message = str(e)
        
        # If rate limit or server error, provide a more helpful message
        if "429" in error_message or "Too Many Requests" in error_message:
            logger.warning("Using fallback extraction due to API rate limits")
        elif any(term in error_message.lower() for term in ["timeout", "server", "unavailable"]):
            logger.warning("Using fallback extraction due to API service issues")
        
        # Fallback to basic extraction if API fails
        return _fallback_key_point_extraction(transcription, lecture_text)

def _fallback_key_point_extraction(transcription, lecture_text=None):
    """
    Basic fallback method for key point extraction when API fails.
    
    Args:
        transcription (str): Transcribed text
        lecture_text (str, optional): Lecture material text
    
    Returns:
        list: Simple list of extracted sentences as key points
    """
    logger.debug("Using fallback key point extraction")
    
    if not transcription:
        return ["No transcription available to extract key points."]
    
    # Tokenize transcription into sentences
    sentences = sent_tokenize(transcription)
    
    # If no lecture text, return top sentences as key points
    if not lecture_text:
        # Return up to 5 sentences, or fewer if transcription is short
        return sentences[:min(5, len(sentences))]
    
    # Simple filtering to find sentences in transcription not in lecture text
    unique_points = []
    clean_lecture = clean_text(lecture_text)
    
    for sentence in sentences:
        clean_sentence = clean_text(sentence)
        if len(clean_sentence) > 20 and clean_sentence not in clean_lecture:
            unique_points.append(sentence)
    
    # If no unique points found, return a message
    if not unique_points:
        return ["All key points in the transcription appear to be covered in the lecture material."]
    
    # Return up to 5 unique points
    return unique_points[:min(5, len(unique_points))]

def generate_summary(transcription, max_length=500):
    """
    Generate a concise summary of the transcription.
    
    Args:
        transcription (str): The transcribed text to summarize
        max_length (int): Maximum length of the summary in characters
        
    Returns:
        str: A summary of the transcription
    """
    logger.debug("Starting summary generation")
    
    if not transcription or len(transcription.strip()) < 50:
        return "The recording is too short to generate a meaningful summary."
    
    # Clean the transcription by removing non-essential markers that might confuse the model
    cleaned_transcription = transcription
    # Remove meta markers like "[Chunk X transcribed with...]"
    cleaned_transcription = re.sub(r'\[Chunk \d+ transcribed.*?\]', '', cleaned_transcription)
    cleaned_transcription = re.sub(r'\[Transcribed using.*?\]', '', cleaned_transcription)
    
    # If the transcription is very long, truncate it for the summary to process faster
    if len(cleaned_transcription) > 10000:
        # Take first 3000, middle 4000, and last 3000 characters for a representative sample
        first_part = cleaned_transcription[:3000]
        middle_start = max(3000, len(cleaned_transcription) // 2 - 2000)
        middle_part = cleaned_transcription[middle_start:middle_start + 4000]
        last_part = cleaned_transcription[-3000:] if len(cleaned_transcription) > 6000 else ""
        
        cleaned_transcription = f"{first_part}\n...\n{middle_part}\n...\n{last_part}"
        logger.debug(f"Truncated long transcription from {len(transcription)} to {len(cleaned_transcription)} characters for summary")
    
    if not OPENAI_API_KEY or openai is None:
        logger.warning("OpenAI API key is missing or client failed to initialize. Using fallback method for summary generation.")
        return _fallback_summary_generation(cleaned_transcription, max_length)
    
    try:
        # Define prompt for summary generation with more specific instructions
        prompt = f"""
        Create a concise summary of the following transcription in about 200 words.
        Focus on:
        1. Main topics and themes
        2. Key points and ideas
        3. Important conclusions or outcomes
        4. Correctly preserve Egyptian Arabic terms and phrases
        
        Make the summary well-structured, coherent, and easy to understand.
        
        Transcription:
        {cleaned_transcription}
        """
        
        # Define the API call function
        def make_openai_request():
            # Check if the OpenAI client is available
            if openai is None:
                raise ValueError("OpenAI client is not available. Please check your API key.")
                
            # Use the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            return openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in summarizing content, especially material containing Egyptian Arabic and technical discussions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length
            )
        
        # Use the retry mechanism for the API call
        response = _api_call_with_retry(
            make_openai_request,
            max_retries=3,  # Reduced retry count to improve user experience
            initial_retry_delay=2
        )
        
        summary = response.choices[0].message.content.strip()
        logger.debug(f"Generated summary of length {len(summary)}")
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        error_message = str(e)
        
        # If rate limit or server error, provide a more helpful message
        if "429" in error_message or "Too Many Requests" in error_message:
            logger.warning("Using fallback summary generation due to API rate limits")
        elif any(term in error_message.lower() for term in ["timeout", "server", "unavailable"]):
            logger.warning("Using fallback summary generation due to API service issues")
        
        # Fallback to basic summary if API fails
        return _fallback_summary_generation(cleaned_transcription, max_length)

def _fallback_summary_generation(transcription, max_length=500):
    """
    Enhanced fallback method for summary generation when API fails.
    Uses a more sophisticated approach to extract key sentences.
    
    Args:
        transcription (str): Transcribed text
        max_length (int): Maximum length of the summary
    
    Returns:
        str: A simple summary of the transcription
    """
    logger.debug("Using enhanced fallback summary generation")
    
    if not transcription:
        return "No transcription available to generate a summary."
    
    # Split into paragraphs first (if any)
    paragraphs = transcription.split('\n\n')
    clean_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 0]
    
    # If no clear paragraphs, use the whole text
    if not clean_paragraphs:
        clean_paragraphs = [transcription]
    
    # Tokenize transcription into sentences
    all_sentences = []
    for paragraph in clean_paragraphs:
        sentences = sent_tokenize(paragraph)
        all_sentences.extend(sentences)
    
    if not all_sentences:
        return "Could not generate a summary from the transcription."
    
    # More sophisticated approach: Extract important sentences
    important_sentences = []
    
    # Always include first sentence for context
    if len(all_sentences) > 0:
        important_sentences.append(all_sentences[0])
    
    # Include sentences with important keywords/phrases that suggest significant content
    # These are common indicators of important information in Arabic and English content
    important_keywords = [
        "important", "significant", "key", "main", "primary", "critical", "crucial", "essential",
        "conclusion", "summary", "result", "therefore", "thus", "finally", "in conclusion",
        "مهم", "أساسي", "رئيسي", "ضروري", "حاسم", "نتيجة", "خلاصة", "في النهاية", "بالتالي"
    ]
    
    # Get sentences containing important keywords (up to 3 more sentences)
    keyword_sentences = []
    for sentence in all_sentences[1:-1]:  # Skip first and last which we add separately
        clean_sentence = sentence.lower()
        if any(keyword in clean_sentence for keyword in important_keywords):
            keyword_sentences.append(sentence)
            if len(keyword_sentences) >= 3:
                break
    
    important_sentences.extend(keyword_sentences)
    
    # Include last sentence for conclusion if we haven't hit our limit
    if len(all_sentences) > 1 and len(important_sentences) < 5:
        important_sentences.append(all_sentences[-1])
    
    # If we still don't have enough sentences, add some from the middle
    if len(important_sentences) < 3 and len(all_sentences) > 4:
        middle_section = all_sentences[len(all_sentences)//4:3*len(all_sentences)//4]
        # Take sentences evenly distributed from the middle section
        step = max(1, len(middle_section) // (3 - len(important_sentences)))
        for i in range(0, len(middle_section), step):
            if len(important_sentences) < 3:
                important_sentences.append(middle_section[i])
    
    # Sort sentences by their original order in the text
    sentence_positions = {sentence: i for i, sentence in enumerate(all_sentences)}
    important_sentences.sort(key=lambda s: sentence_positions.get(s, 999))
    
    # Combine sentences and ensure length limit
    summary = " ".join(important_sentences)
    
    # Truncate if necessary
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."
    
    return summary
