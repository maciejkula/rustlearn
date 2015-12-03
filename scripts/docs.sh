#!/bin/sh

cargo doc && git fetch origin gh-pages && git checkout gh-pages && (git mv doc doc-$(git describe --always master^) || rm -rf doc) && mv target/doc/ ./doc && git add -A ./doc* && git commit -m 'Update docs.' && git push origin gh-pages
