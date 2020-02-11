---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "MongoLink"
summary: "MongoLink is a package for interacting with MongoDB inside the Wolfram Language via the high-performance MongoDB C driver."
authors: []
tags: ["database", "mongodb", "data science", "machine learning"]
categories: ["Data Science"]
date: 2018-02-01T19:52:45+02:00

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

url_code: "https://github.com/WolframResearch/MongoLink"
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

[MongoLink](https://github.com/WolframResearch/MongoLink) is a package for interacting with [MongoDB](https://www.mongodb.com/) inside the [Wolfram Language](https://en.wikipedia.org/wiki/Wolfram_Language#targetText=The%20Wolfram%20Language%20is%20a,and%20the%20Wolfram%20Programming%20Cloud.) via the high-performance [MongoDB C driver](http://mongoc.org/). This package [now ships as part of Mathematica](https://reference.wolfram.com/language/MongoLink/guide/MongoLinkOperations.html).

I wrote this package as one part of a framework for managing the Wolfram Machine Learning teams terabyte-scale datasets I lead the development on.