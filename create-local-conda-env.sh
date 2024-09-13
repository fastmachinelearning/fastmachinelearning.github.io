!/bin/bash

conda create -n jekyll-env
conda activate jekyll-env
conda install -c conda-forge ruby
conda install clang_osx-arm64 clangxx_osx-arm64
gem install bundler jekyll
ln -s ~/miniconda3/envs/jekyll-env/bin/ruby ~/miniconda3/envs/jekyll-env/share/rubygems/bin/ruby
jekyll serve