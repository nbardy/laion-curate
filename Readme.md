# Dataset extractor

What if instead of building our own datasets? We just find them inside of LAION.

This project uses the awesome [clip-retrieval](https://github.com/rom1504/clip-retrieval) library to filter down
the massive [LAION-5B](https://laion.ai/blog/laion-5b/) into smaller curated sub datasets.

Provide a set of labels, and search terms(images, or text) and have a datset downloaded for you that fits the conditions. Relabled with your given titles.

## How to use

### Create your own Dataset

See the example wave dataset in the [example_datsets](https://github.com/nbardy/laion-curate/tree/main/example_datasets) folder.

Create a new dataset file mimicing the json file and update it with your given search terms. You'll want to use [clip retireval](https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn5.laion.ai&index=laion5B&useMclip=false) to check with images and text have quality data.

### Run the tool

After creating the datset you'll want to run the tool with


```bash
# Crawl the data with your given search terms
python fetch_data.py --dataset_file_path=new_dataset.json --output_dir=new_datset_dir

# Upload the data to the huggingface hub
python --dataset_dir new_dataset_dir --repo_id user/new_dataset_huggingface_repo
```

### Fine tune stable diffusion

TODO


## Motivation

LAION is large and sprawling with lots of labels that have a weak relationship to the images. This provides a great foundation for a backbone, but can make the text->image portion particularly uncontrollable.

Finding a class specification dataset and cleaning up the text labels for each can be a great way to make models more controllable.
