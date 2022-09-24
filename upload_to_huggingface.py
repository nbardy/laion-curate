import glob
import os
import json

import argparse
import datasets


title_metadata_folder_name = "title_metadata"
image_dir_name = "image_files"


# This function loads all metadata files in the dataset folder
#
# Then prunes out the closest matches to the negative text
#
# Then merges them all into a jsonl file with
# {image: image_path, text: str, alternate_titles: [str, ...]}
#
# The metadata files contain information on a batch of images for the titles
# with the fields:
#   - alternate_titles: [str, ...]
#   - negative: [str, ...]
#   - keys: [str, ...] # paths without file ending pointing to the image/json files
#   - negative_results: {str: {key: float}} # negative result for negative text and key
def consolidate_results(input_dir):
    # Find all files in the dir ending with .json
    metadata_files = glob.glob(
        os.path.join(input_dir, title_metadata_folder_name, "*.json")
    )
    all_data_points = []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        negative = metadata.get("negative", [])
        alternate_titles = metadata.get("alternate_titles", [])
        keys = metadata["keys"]
        title = metadata["title"]
        key_to_index = metadata["key_to_index"]

        print("Processing Title:", title)

        # Create a set of pruned negative results
        negative_results = {}

        for negative_text, results in metadata["negative_results"].items():
            # Calculate the lowest 20% threshold
            # We are using a distance similiarity measure
            # and we want to filter out the closest matches
            threshold = sorted(results.values())[int(len(results) * 0.2)]
            for key, score in results.items():
                if score < threshold:
                    negative_results[key] = True

        for key in keys:
            # If the key is in the negative results
            # then we skip it
            if key in negative_results:
                continue

            # Otherwise we add it to the dataset
            image_path = os.path.join(input_dir, image_dir_name, key + ".jpg")

            img_index = key_to_index[key]
            # Load url from the dataset as urls.json in the dataset folder
            with open(os.path.join(input_dir, "urls.json"), "r") as f:
                urls = json.load(f)

            url = urls[img_index]["url"]

            # check if file exists
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist")
                continue

            all_data_points.append(
                {
                    # We don't upload the image to huggingface because this
                    # is a subset of common crawl and we don't have the rights
                    # to distribute it
                    #
                    # "image": image,
                    "image_url": url,
                    "image_path_local": image_path,
                    "text": title,
                    "alternate_titles": alternate_titles,
                }
            )

    return all_data_points


# Run consolidate and upload on main
if __name__ == "__main__":
    default_repo_id = "Nbardy/filtered-laion-wave-data"
    default_datset_dir = "wave_data_scrape"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_datset_dir,
        help="Path to the directory where the dataset is located. .jsonl file will be saved here",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=default_repo_id,
        help="The repo id to upload to. Defaults to Nbardy/filtered-laion-wave-data",
    )
    args = parser.parse_args()

    all_data_points = consolidate_results(args.dataset_dir)

    # Write the data points to a jsonl file
    with open(os.path.join(args.dataset_dir, "dataset.jsonl"), "w") as f:
        for data_point in all_data_points:
            json.dump(data_point, f)
            f.write("\n")

    dataset = datasets.load_dataset(
        "json",
        data_files=[os.path.join(args.dataset_dir, "dataset.jsonl")],
    )

    dataset.push_to_hub(args.repo_id)
