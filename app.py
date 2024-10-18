import requests
import pandas as pd
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import pytesseract
import json
import base64

def scrape_pdf_data_jina_from_url(pdf_url):
    # if not isinstance(pdf_url, str) or not pdf_url.strip():
    #     raise ValueError("Invalid PDF URL provided")

    scraping_url = 'https://r.jina.ai/' + pdf_url

    headers = {
        "Accept": "application/json",
        "X-No-Cache": "true"
    }

    print("Sending JINA Request for PDF URL: " + pdf_url)

    try:
        response = requests.get(scraping_url, headers=headers)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return 'NA'

    print("JINA Response Received")

    if response.status_code == 200:
        try:
            response_data = response.json()
        except ValueError:
            print("Invalid JSON response received from JINA")
            return 'NA'

        if 'data' not in response_data or len(str(response_data['data']).split()) < 50:
            print('Retrying Jina Request')

            headers = {
                "Accept": "application/json",
                "X-No-Cache": "true",
                "X-Timeout": "5"
            }

            try:
                response2 = requests.get(scraping_url, headers=headers)
            except requests.RequestException as e:
                print(f"Retry request failed: {e}")
                return 'NA'

            if response2.status_code != 200:
                print(f"Failed to retrieve data from JINA on retry. Status code: {response2.status_code}")
                return 'NA'

            try:
                response_data = response2.json()
            except ValueError:
                print("Invalid JSON response received from JINA on retry")
                return 'NA'

            response = response2

    else:
        print(f"Failed to retrieve data from JINA. Status code: {response.status_code}")
        return 'NA'
    
    if 'data' in response.json() and 'usage' in response.json().get('data', {}) and 'tokens' in response.json()['data'].get('usage', {}):
        print('Jina Response tokens new: ' + str(response.json()['data']['usage']['tokens']))
        return response.json().get('data', {}).get('content', 'NA')
    else:
        return 'NA'
    

def scrape_pdf_data_jina_from_bytes(pdf_bytes):
    try:
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode PDF bytes: {e}")
        return 'NA'
    
    url = 'https://r.jina.ai/'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'X-No-Cache': 'true',
        'X-With-Images-Summary': 'true'
    }
    data = {
        'url': 'https://example.com',
        'pdf': pdf_base64
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request to JINA failed: {e}")
        return 'NA'
    
    try:
        response_data = response.json()
    except ValueError:
        print("Invalid JSON response received from JINA")
        return 'NA'
    
    if 'data' in response_data and 'content' in response_data['data']:
        return response_data['data']['content']
    else:
        print("Required data not found in JINA response")
        return 'NA'


def download_pdf_from_url(url):
    # if not isinstance(url, str) or not url.startswith('http'):
    #     raise ValueError("Invalid URL provided")
    
    response = requests.get(url)
    if response.status_code == 200:
        if 'application/pdf' in response.headers.get('Content-Type', ''):
            return response.content
        else:
            raise Exception("The URL does not point to a PDF file")
    else:
        raise Exception(f"Failed to download the PDF. Status code: {response.status_code}")


def ocr_on_pdf_from_bytes(pdf_bytes):
    
    pages = convert_from_bytes(pdf_bytes)
    
    # Validate the pages
    if not pages or not isinstance(pages, list):
        raise ValueError("Failed to convert PDF bytes to pages")

    def deskew(image):
        # Check if the image is already grayscale
        if len(image.shape) == 2:  # Grayscale image
            gray = image  # Image is already in grayscale
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not

        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    
    def get_conf(image):
        # Dummy confidence value calculation
        return 0.99
    
    # Function to process each page (from extra code)
    def process_page(page):
        try:
            # Convert page image to array
            page_arr = np.array(page)
            # Convert to grayscale
            page_arr_gray = cv2.cvtColor(page_arr, cv2.COLOR_BGR2GRAY)
            # Deskew the page
            page_deskew = deskew(page_arr_gray)
            # Calculate confidence value (you can modify this function as needed)
            page_conf = get_conf(page_deskew)
            # Extract text and OCR data using Tesseract
            d = pytesseract.image_to_data(page_deskew, output_type=pytesseract.Output.DICT)
            d_df = pd.DataFrame.from_dict(d)
            # Get block number to find header and footer
            block_num = int(d_df.loc[d_df['level'] == 2, 'block_num'].max())
            # Drop header and footer by index
            header_index = d_df[d_df['block_num'] == 1].index.values
            footer_index = d_df[d_df['block_num'] == block_num].index.values
            # Combine text, excluding header and footer regions
            text = ' '.join(d_df.loc[(d_df['level'] == 5) & (~d_df.index.isin(header_index) & ~d_df.index.isin(footer_index)), 'text'].values)
            return page_conf, text
        except Exception as e:
            # Handle extraction failure
            if hasattr(e, 'message'):
                return -1, e.message
            else:
                return -1, str(e)
    
    # Create a list to store extracted text and confidence values from all pages
    output_data = []

    # Iterate over pages and process each one
    for idx, page in enumerate(pages):
        # Process the page using the process_page function
        page_conf, text = process_page(page)
        # Append results to the output data
        output_data.append({
            "page_number": idx + 1,
            "text": text
        })

    return output_data


def ocr_on_pdf_from_url(pdf_url):

    pdf_bytes = download_pdf_from_url(pdf_url)
    
    # Validate the downloaded PDF bytes
    if not pdf_bytes:
        raise ValueError("Failed to download PDF from the provided URL")
    
    pages = convert_from_bytes(pdf_bytes)
    
    # Validate the pages
    if not pages or not isinstance(pages, list):
        raise ValueError("Failed to convert PDF bytes to pages")

    def deskew(image):
        # Check if the image is already grayscale
        if len(image.shape) == 2:  # Grayscale image
            gray = image  # Image is already in grayscale
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not

        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    
    def get_conf(image):
        # Dummy confidence value calculation
        return 0.99
    
    # Function to process each page (from extra code)
    def process_page(page):
        try:
            # Convert page image to array
            page_arr = np.array(page)
            # Convert to grayscale
            page_arr_gray = cv2.cvtColor(page_arr, cv2.COLOR_BGR2GRAY)
            # Deskew the page
            page_deskew = deskew(page_arr_gray)
            # Calculate confidence value (you can modify this function as needed)
            page_conf = get_conf(page_deskew)
            # Extract text and OCR data using Tesseract
            d = pytesseract.image_to_data(page_deskew, output_type=pytesseract.Output.DICT)
            d_df = pd.DataFrame.from_dict(d)
            # Get block number to find header and footer
            block_num = int(d_df.loc[d_df['level'] == 2, 'block_num'].max())
            # Drop header and footer by index
            header_index = d_df[d_df['block_num'] == 1].index.values
            footer_index = d_df[d_df['block_num'] == block_num].index.values
            # Combine text, excluding header and footer regions
            text = ' '.join(d_df.loc[(d_df['level'] == 5) & (~d_df.index.isin(header_index) & ~d_df.index.isin(footer_index)), 'text'].values)
            return page_conf, text
        except Exception as e:
            # Handle extraction failure
            if hasattr(e, 'message'):
                return -1, e.message
            else:
                return -1, str(e)
    
    # Create a list to store extracted text and confidence values from all pages
    output_data = []

    # Iterate over pages and process each one
    for idx, page in enumerate(pages):
        # Process the page using the process_page function
        page_conf, text = process_page(page)
        # Append results to the output data
        output_data.append({
            "page_number": idx + 1,
            "text": text
        })

    return output_data

# This is my entire code in different functions right now to ask user an option to convert pdf -> text either from url or the uploaded pdf file. 
# I want the flow to be this way: 
# 1. For uploaded pdf:
#     i. Use scrape_pdf_data_jina_from_filepath
#     ii. Check if the response received is above 50 words, return the response
#     iii. If response below 50 words, use ocr_on_pdf_from_filepath, return the response
# 2. For pdf url sent:
#     i. Use scrape_pdf_data_jina_from_url
#     ii. Check if the response received is above 50 words, return the response
#     iii. If response below 50 words, use ocr_on_pdf_from_url, return the response

from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'pdf_file' in request.files:
        pdf_file = request.files['pdf_file']
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            pdf_bytes = pdf_file.read() 
            response = scrape_pdf_data_jina_from_bytes(pdf_bytes) 
            if len(response.split()) > 50:
                return response
            else:
                return ocr_on_pdf_from_bytes(pdf_bytes)
    
    app.logger.debug(f"Received PDF URL: {request.form['pdf_url']}")
    pdf_url = request.form['pdf_url']
    response = scrape_pdf_data_jina_from_url(pdf_url)
    if len(response.split()) > 50:
        return response
    else:
        return ocr_on_pdf_from_url(pdf_url)

# if __name__ == '__main__':
#     app.run(debug=True)