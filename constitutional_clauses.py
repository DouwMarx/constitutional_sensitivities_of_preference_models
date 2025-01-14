"""Retrieves Collective Constitutional AI dataset with rules or behaviour for a given AI

See:
https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input
https://github.com/saffronh/ccai/tree/main
https://pol.is/report/r3rwrinr5udrzwkvxtdkj

"""
import requests

url = 'https://raw.githubusercontent.com/saffronh/ccai/main/clean_comments.csv'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a local file in write-binary mode
    with open('data/ccai_clean_comments.csv', 'wb') as file:
        # Write the content of the response (the file) to the local file
        file.write(response.content)
    print("File downloaded and saved as 'clean_comments.csv'")
else:
    print(f"Failed to download file. Status code: {response.status_code}")


# Load the data and upload to hugging face Hub
import pandas as pd
from datasets import Dataset, DatasetInfo

ccai_data = pd.read_csv('data/ccai_clean_comments.csv')

info = DatasetInfo(description="Collective Constitutional AI dataset with rules or behaviour for a given AI from 'https://raw.githubusercontent.com/saffronh/ccai/main/clean_comments.csv', See: https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input, https://github.com/saffronh/ccai/tree/main, https://pol.is/report/r3rwrinr5udrzwkvxtdkj")
ccai_dataset = Dataset.from_pandas(ccai_data, info=info)
ccai_dataset.push_to_hub("douwmarx/ccai-dataset")

