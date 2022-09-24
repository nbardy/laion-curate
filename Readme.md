# Dataset extractor

What if instead of building our own datasets? We just find them inside of LAION.

This project uses the awesome [clip-retrieval](https://github.com/rom1504/clip-retrieval) library to filter down
the massive [LAION-5B](https://laion.ai/blog/laion-5b/) into smaller curated sub datasets.

Provide a set of labels, and search terms(images, or text) and have a datset downloaded for you that fits the conditions. Relabled with your given titles.

## Motivation

LAION is large and sprawling with lots of labels that have a weak relationship to the images. This provides a great foundation for a backbone, but can make the text->image portion particularly uncontrollable.

Finding a class specification dataset and cleaning up the text labels for each can be a great way to make models more controllable.
