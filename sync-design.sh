#!/bin/sh
# Pull the GoingRusting design (CSS, 3D scenes, favicon) from ../GoingRustingWebSite.
set -e
S=../GoingRustingWebSite
{ echo "/* Copied from GoingRustingWebSite/src/styles/global.css + tokens.css. Re-sync with: sh sync-design.sh */"
  cat $S/src/styles/tokens.css; grep -v -E "^@import" $S/src/styles/global.css; } > static/gr.css
cp $S/public/favicon.svg static/
echo "import { startScene } from '$S/src/components/scene';
for (const el of document.querySelectorAll('.gr-scene')) startScene(el.querySelector('canvas'), el.dataset.kind, JSON.parse(el.dataset.opt || '{}'));" \
  | $S/node_modules/.bin/esbuild --bundle --minify --format=iife --loader=ts --outfile=static/scene.js
