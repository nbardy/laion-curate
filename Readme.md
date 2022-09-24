# Dataset extractor

What if instead of building our own datasets? We just find them inside of LAION.

This project uses the awesome [clip-retrieval](https://github.com/rom1504/clip-retrieval) library to filter down
the massive [LAION-5B](https://laion.ai/blog/laion-5b/) into smaller curated sub datasets.

Provide a set of labels, and search terms(images, or text) and have a datset downloaded for you that fits the conditions.

The excellent retraining of stable diffusion with a pokemon subset has shown just how powerful fine-tuning stable diffusion can be.
https://twitter.com/m1guelpf/status/1573297520498950145

Finding a class specification dataset and cleaning up the text labels for each can be a great way to make models more controllable
