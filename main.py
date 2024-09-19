import os
import json
import requests
import fitz  # PyMuPDF
from google.cloud import vision
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Set up API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set up Google Custom Search API key and engine ID for image search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF pages to images using PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def detect_text(image: Image.Image) -> str:
    """Detects text in an image using Google Vision API."""
    client = vision.ImageAnnotatorClient()

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    content = img_byte_arr.getvalue()

    vision_image = vision.Image(content=content)
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations

    if texts:
        return texts[0].description
    return ""

def get_llm_response(prompt: str) -> str:
    """Get a response from the LLM."""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Using GPT-4 for better understanding of medical content
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def summarize_page(page_content: str, page_number: int) -> str:
    """Generate a summary for a single page."""
    prompt = f"""Summarize the following medical content from page {page_number} in 100-150 words:

    {page_content}

    Focus on key cardiac physiology concepts, conduction system details, and any important diagrams or charts mentioned."""

    return get_llm_response(prompt)

def generate_flashcards(page_content: str, page_number: int) -> List[Dict[str, str]]:
    """Generate flashcards for a single page."""
    prompt = f"""Create 5 flashcards (question-answer pairs) based on the following medical content from page {page_number}:

    {page_content}

    Ensure the flashcards cover key cardiac physiology concepts, conduction system details, and any important diagrams or charts mentioned.
    Format your response as a JSON array of objects, each with 'question' and 'answer' keys."""

    response = get_llm_response(prompt)
    try:
        flashcards = json.loads(response)
        if not isinstance(flashcards, list):
            raise ValueError("Response is not a list")
        for card in flashcards:
            if not isinstance(card, dict) or 'question' not in card or 'answer' not in card:
                raise ValueError("Invalid flashcard format")
        return flashcards
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for flashcards on page {page_number}: {e}")
        print(f"Raw response: {response}")
        return []
    except ValueError as e:
        print(f"Error in flashcard format on page {page_number}: {e}")
        print(f"Raw response: {response}")
        return []

def generate_search_query(page_content: str, page_number: int) -> str:
    """Generate a search query for a single page."""
    prompt = f"""Generate a short, general search query (5-7 words) to find medical images related to the cardiac physiology content on page {page_number}:

    {page_content}

    The query should be related to cardiac anatomy, physiology, or the heart's conduction system mentioned in the content.
    Provide only the search query, without any additional text or quotation marks."""

    return get_llm_response(prompt)

def get_image_urls(query: str) -> List[str]:
    """Get image URLs from Google Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "searchType": "image",
        "num": 10,
        "fileType": "jpg,png,gif",  # Specify allowed file types
        "rights": "cc_publicdomain|cc_attribute|cc_sharealike"  # Use more permissive rights
    }
    response = requests.get(url, params=params)
    results = response.json()
    
    if 'items' in results:
        return [item['link'] for item in results['items']]
    else:
        print(f"No images found for query: {query}")
        print(f"API response: {results}")
        return []

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """Process the entire PDF and generate required outputs."""
    images = pdf_to_images(pdf_path)
    results = {
        "summaries": {},
        "flashcards": {},
        "search_queries_and_images": {}
    }

    for i, image in enumerate(images, start=1):
        print(f"Processing page {i}...")
        
        # Detect text using Google Vision API
        page_content = detect_text(image)
        
        # Generate summary
        results["summaries"][f"page_{i}"] = summarize_page(page_content, i)
        
        # Generate flashcards
        results["flashcards"][f"page_{i}"] = generate_flashcards(page_content, i)
        
        # Generate search query and get image URLs
        query = generate_search_query(page_content, i)
        image_urls = get_image_urls(query)
        results["search_queries_and_images"][f"page_{i}"] = {
            "search_query": query,
            "image_urls": image_urls
        }

    return results

def save_results(results: Dict[str, Any], output_dir: str):
    """Save results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "summaries.json"), "w") as f:
        json.dump(results["summaries"], f, indent=2)
    
    with open(os.path.join(output_dir, "flashcards.json"), "w") as f:
        json.dump(results["flashcards"], f, indent=2)
    
    with open(os.path.join(output_dir, "search_queries_and_images.json"), "w") as f:
        json.dump(results["search_queries_and_images"], f, indent=2)

if __name__ == "__main__":
    pdf_path = r"E:\Task\CVS.pdf"
    output_dir = "output"
    
    results = process_pdf(pdf_path)
    save_results(results, output_dir)
    print("Processing complete. Results saved in the 'output' directory.")
