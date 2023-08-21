# import asyncio
# from pyppeteer import launch
# import csv
# import requests
# import json
import re
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_file_paths):
    texts = []
    for pdf_file_path in pdf_file_paths:
        try:
            # Extract text from the PDF
            text = extract_text(pdf_file_path)

            # Define the noisy patterns you want to remove
            noisy_patterns = [
                r'\s{2,}',  # extra whitespace
                r'\n{2,}',  # extra line breaks
                #r'\W+',  # non-alphanumeric characters
            ]

            # Loop over the noisy patterns and replace them with a single space
            for pattern in noisy_patterns:
                text = re.sub(pattern, ' ', text, flags=re.MULTILINE)

            # Further clean the text by removing stopwords
            text = ' '.join([word for word in text.split()])

            texts.append(text)
            
        except Exception as e:
            print(f"Error extracting text from PDF file at {pdf_file_path}: {e}")

    # Join all texts into a single string
    all_texts = ' '.join(texts)
    
    return all_texts




# # Use the function on your PDF file
# cleaned_text = extract_and_clean_text('Sova Pilot.pdf')

# if cleaned_text is not None:
#     # Print the cleaned text
#     print(cleaned_text)
# else:
#     print("No text was extracted from the PDF.")

# # GitHub Link Extraction
# def extract_github_repo_info(github_repo_url):
#     api_url = github_repo_url.replace('https://github.com/', 'https://api.github.com/repos/')
#     response = requests.get(api_url)
#     repo_info = json.loads(response.text)
#     return repo_info

# # CSV Extraction
# def extract_data_from_csv(csv_file_path):
#     with open(csv_file_path, 'r') as file:
#         reader = csv.reader(file)
#         data = [row for row in reader]
#     return data

# # Text Extraction
# def extract_text_from_file(text_file_path):
#     with open(text_file_path, 'r') as file:
#         text = file.read()
#     return text


# async def extract_web_content(url):
#     # Launch the browser in headless mode
#     browser = await launch()
#     # Create a new page
#     page = await browser.newPage()
#     # Go to the URL, wait until network is idle
#     try:
#         await page.goto(url, {'waitUntil': 'networkidle0'})
#     except Exception as e:
#         print(f"An error occured: {e}")
#         await browser.close()
#         return
#     # Wait for 1 second to allow JavaScript to run
#     await asyncio.sleep(1)

#     # Get the text content of specific elements
#     content = await page.evaluate('''() => {
#         let elements = document.querySelectorAll('p');  // replace 'p' with the specific tag you want
#         return Array.from(elements).map(element => element.textContent);
#     }''')
#     #, h1, h2, h3, h4, h5, h6, span, div, a
#     # Close the browser
#     await browser.close()
#     return content
# # Synchronous wrapper
# async def extract_all_content_from_webpage(url):
#     task = asyncio.create_task(extract_web_content(url))
#     return await task




