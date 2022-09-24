## Equivalent request in python
import requests
import json
import argparse
import img2dataset

import os
import torch
import clip
from PIL import Image


metadata_url = "https://knn5.laion.ai//metadata"

AESTHETIC_SCORE = "6"
NEGATIVE_FILTER_PERCENT = 0.2
ENABLE_WANDB = False
PROCESSES_COUNT = 8
RESULT_COUNT = 500
IMAGE_SIZE = 512


def get_metadata(ids):
    body = {"ids": ids, "indice_name": "laion5B"}
    method = "POST"

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "text/plain;charset=UTF-8",
    }

    response = requests.request(
        method, metadata_url, headers=headers, data=json.dumps(body)
    )
    return response.json()


knn_url = "https://knn5.laion.ai//knn-service"


def get_knn_results(text=None, image_url=None, n=None):
    if text is None and image_url is None:
        raise ValueError("Must provide either text or image_url")

    body = {
        "text": text,
        "image": None,
        "image_url": image_url,
        "embedding_input": None,
        "modality": "image",
        "num_images": n,
        "indice_name": "laion5B",
        "num_result_ids": n,
        "use_mclip": False,
        "deduplicate": True,
        "use_safety_model": False,
        "use_violence_detector": False,
        "aesthetic_score": AESTHETIC_SCORE,
        "aesthetic_weight": "0.5",
    }

    method = "POST"

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "text/plain;charset=UTF-8",
    }

    response = requests.request(method, knn_url, headers=headers, data=json.dumps(body))

    return response.json()


device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Calculates the distance between an image and text embeddings
def get_clip_distance(image_path, text):
    print(f"Calculating distance for {image_path}, {text}")

    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(image, tokens)
        return logits_per_image


# Load CLIP


## TODO(p0): Add captions to img2datset(It's in the readme)
## Try to make two sets of captions and do 80/20 split on specific vs general

SAMPLES_PER_SHARD = 10000

# Turns an i into the path into the proper shard folder
# and file path
#
# The shard folder name is zero padded to 5 digits
# The shard file name is zero padded to 9 digits
#
# This gives the key for pointing to the .json, or .json file
# for a given image index
def get_image_key(i):
    shard_folder = i // SAMPLES_PER_SHARD
    shard_file = i % SAMPLES_PER_SHARD

    shard_folder = str(shard_folder).zfill(5)
    shard_file = str(shard_file).zfill(9)

    return os.path.join(shard_folder, shard_file)


# Loads a json file with the same format as the above dictionary
# And loads it's metadata then saves it in a json file
#
# This function downloads the top matches for each text or imageUrl
# downloads them with im2dataset
# filters out the top matches to the negative text if provided
# and saves the results in a json file
def fetch_and_save_dataset(dataset_file_path, output_dir):
    with open(dataset_file_path, "r") as f:
        dataset_json = json.load(f)

    dataset = dataset_json["data"]

    all_metadata = []

    batch_metadata_per_title = {}

    # Make directory in the dataset folder
    image_dir_name = "image_files"
    image_dir = os.path.join(output_dir, image_dir_name)
    os.makedirs(image_dir, exist_ok=True)

    image_idx = 0
    for i, example in enumerate(dataset):
        title = example["title"]
        print(f"Fetching {title}, {i+1}/{len(dataset)}")

        positive = example.get("positive", [])
        negative = example.get("negative", [])
        alternate_titles = example.get("alternate_titles", [])

        all_for_title = []

        # Fetch all text matches
        for search_string in positive:
            print(f"Fetching positive results for {search_string}")
            results = get_knn_results(text=search_string, n=RESULT_COUNT)
            ids = list(map(lambda x: x["id"], results))

            metadata = get_metadata(ids)
            print(f"Found {len(metadata)} results")

            all_for_title.extend(metadata)

        # Fetch all image urls
        for image_url in example.get("positive_image_urls", []):
            print(f"Fetching positive results for {image_url}")
            results = get_knn_results(image_url=image_url, n=RESULT_COUNT)
            ids = list(map(lambda x: x["id"], results))

            metadata = get_metadata(ids)

            print(f"Found {len(metadata)} results")
            all_for_title.extend(metadata)

        # Add negative string for later lookup when
        # the image is downloaded for clip filtering
        #
        # Storing it with each item adds extra data that
        # we may want to prune later for performance
        #
        # But for now makes it easier to process
        title_metadata = {}
        keys = []
        key_to_index = {}
        for metadata in all_for_title:
            key = get_image_key(image_idx)
            keys.append(key)

            key_to_index[key] = image_idx
            image_idx += 1

        title_metadata["negative"] = negative
        title_metadata["keys"] = keys
        title_metadata["key_to_index"] = key_to_index
        # TODO: Make an alternate title for each image that reuses it's caption data
        # or CLIP data.
        #
        # We want the set of text for image to be of multiple options weighted differently.
        # The goal of this is to prevent overfitting on captions while still refining the
        # text encoder towards the new title to clean the dataset and make it more specific
        #
        # 1. Title (60%)
        # 2. Alternative title (20%)
        # 3. {Title}; {Caption } or {Title}, Caption(15%)
        # 3. Caption (5%)
        title_metadata["alternate_titles"] = alternate_titles
        title_metadata["title"] = title
        batch_metadata_per_title[title] = title_metadata

        print(f"Total results count for: {title}")
        print(f"{len(all_for_title)} results")

        all_metadata.extend(all_for_title)

    urls = list(map(lambda x: {"url": x["metadata"]["url"]}, all_metadata))
    # Save url list as a text file with each url on a new line
    url_file = os.path.join(output_dir, "urls.json")
    input_format = "json"

    with open(url_file, "w") as f:
        json.dump(urls, f)

    # with open(url_file, "w") as f:
    #     f.write("\n".join(urls))

    # Download images
    print(f"Downloading images")

    # Download images
    img2dataset.download(
        url_list=url_file,
        input_format=input_format,
        image_size=IMAGE_SIZE,
        output_folder=image_dir,
        enable_wandb=ENABLE_WANDB,
        resize_mode="no",
        processes_count=PROCESSES_COUNT,
    )

    print(f"Filtering negative results for {title}")

    for title, title_metadata in batch_metadata_per_title.items():
        negative = title_metadata["negative"]
        title_metadata["negative_results"] = {}

        for negative_text in negative:
            title_metadata["negative_results"][negative_text] = {}

            ## TODO: Batch Clip call for GPU
            for key in title_metadata["keys"]:
                image_path = key + ".jpg"
                image_path = os.path.join(image_dir, image_path)

                # Check that file exists otherwise remove key
                if not os.path.exists(image_path):
                    # remove key from keys
                    title_metadata["keys"].remove(key)
                    continue

                dist = get_clip_distance(image_path, negative_text)

                # Convert single value tensor to float
                dist = dist.item()

                title_metadata["negative_results"][negative_text][key] = dist

        # Save metadata
        print(f"Saving metadata for {title}")
        titleMetadataFolder = "title_metadata"

        # Make directory in the dataset folder
        titleMetadataDir = os.path.join(output_dir, titleMetadataFolder)
        os.makedirs(titleMetadataDir, exist_ok=True)

        metadata_file_path = os.path.join(
            output_dir, titleMetadataFolder, f"{title}.json"
        )

        with open(metadata_file_path, "w") as f:
            json.dump(title_metadata, f, indent=2)


if __name__ == "__main__":
    # Args for dataset filename and output directory
    default_output_dir = "wave_data_scrape"
    default_dataset_file_path = "example_datasets/wave_dataset.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default=default_dataset_file_path,
        help="Path to the dataset file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Path to the output directory",
    )

    args = parser.parse_args()

    fetch_and_save_dataset(args.dataset_file_path, args.output_dir)
