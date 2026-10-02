# RustingScience

Benchmarks and research notes from the Rusting projects, built with [Zola](https://www.getzola.org/).

- Preview locally: `zola serve`, then open http://127.0.0.1:1111
- New post: create `content/posts/<slug>/index.md`; put images and scripts next to it
- Publish: push to `main`; `.github/workflows/deploy.yml` builds and deploys to GitHub Pages
- Design: `static/gr.css` and `static/scene.js` come from `../GoingRustingWebSite`. Run `sh sync-design.sh` after changing the design there; edit only `static/science.css` here.
- Post theme: set `[extra] project = "engine" | "brain" | "shader"` in a post's front matter to pick its colour and 3D scene.
