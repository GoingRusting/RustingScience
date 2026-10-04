# RustingScience

Benchmarks, experiments and research notes on Rust, GPUs and machine learning, from the RustingStudio projects and beyond.

This repository holds only the posts. They are published on the RustingStudio website at https://rustingstudio.github.io/science/.

## Writing a post

Create `posts/<slug>/index.md`. The slug becomes the URL: `/science/<slug>/`. Put images, scripts and data next to `index.md` and link to them with relative paths.

```markdown
---
title: "TensorFlow vs PyTorch vs RustingBrain"
date: 2026-10-04
description: "One or two sentences. Shown in the post list and in search results."
tags: [benchmark, deep learning]
project: brain   # optional: engine | brain | shader. Picks the colour and 3D scene. Omit for general topics.
---

The first paragraph says what the reader will know at the end.
```

- Do not start the body with a `# Title`. The site renders the title from the front matter.
- Every number needs the script or the hardware that produced it.
- HTML comments (`<!-- TODO -->`) are removed when the post is published.

## Publishing

1. Push the post to `main`.
2. In the website repository, run `npm run sync-content` and commit the result.
