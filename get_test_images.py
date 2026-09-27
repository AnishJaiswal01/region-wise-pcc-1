import urllib.request

test_cases = {
    "test_pathology.jpg": "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
}

for filename, url in test_cases.items():
    print(f"Downloading {filename}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print("Success!")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")