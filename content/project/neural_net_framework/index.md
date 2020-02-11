---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Mathematica Neural Net Framework"
summary: "Was one of the two main creators of the Mathematica Neural Net Framework"
authors: []
tags: ["machine learning"]
categories: []
date: 2019-09-10T16:12:37+02:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

The [Mathematica Neural Net Framework](https://reference.wolfram.com/language/guide/NeuralNetworks.html) is a high-level deep learning framework that is part of the Wolfram Language/Mathematica, rather than being a standalone package. Some nice features of this framework: 

- Automatic support variable-length sequences without the need for padding, without sacrificing performance on GPUs. I co-wrote a [post for O'Reilly](https://www.oreilly.com/ideas/apache-mxnet-in-the-wolfram-language) explaining some of the tricks we used to achieve this.
- Uses MXNet as a backend (like how Keras uses TensorFlow). I added a signficant number of features to MXNet to support things like variable-length sequence handling.
- Supports multi-GPU training, so designed to be scalable.
- A belief that having access to an extensive repository of pre-trained nets is essential for a high-level deep learning framework. I lead the creation of the [Wolfram Neural Net Repository](https://resources.wolframcloud.com/NeuralNetRepository/) for this goal.



