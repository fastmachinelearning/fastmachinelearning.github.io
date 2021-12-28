# Fast ML Lab Website Insructions
This website is built on Github Pages using Jekyll, and thus, editing raw HTML should be unneccessary for most content updates.

## Add a News Item

Create a file `_posts/yyyy-mm-dd-title.md` and add the following at the beginning:

```yaml
---
title: HLS4ML
external_link: https://fastmachinelearning.org/hls4ml/
layout: post
description: 'hls4ml: an open-source code framework for translating machine learning algorithms directly into FPGA firmware'
image: /images/hls4ml.png
---
```
 Note that file paths for images should include a leading `/` to indicate an absolute path from the root of the site. Putting quotes around the title/description (single or double) is reccomended to avoid issues with yaml parsing. You can also add markdown content following this header and it will render as a complete web page, but as of yet there is no link to this post displayed on the front page, only content from the header. This may change in the future.


## Add a Person
In the file `_data/people/institution_name/lastf.yml` include the following:
```yaml
name: First Last
degree: Degree
field: Field
position: Current Position at Institution
interests: List of Interests
image: /images/lastf.png
external_link: http://your.url.here/
```
Note that the Jekyll site is set up to generate and alphabetize the institution list based on the folder names on the fly, interpreting an underscore in the directory name as a space. Therefore, if adding someone from a new institution, just add a new folder for their institution and place the person's info file inside. Note that file paths for images should include a leading `/` to indicate an absolute path from the root of the site.

## Add a Paper
Add papers to the yaml list in `_data/papers.yml`

Note that this will render markdown syntax. Putting quotes around the title (single or double) is reccomended to avoid issues with yaml parsing.

**Example:**
```yaml
- 'Real-time Artificial Intelligence for Accelerator Control: A Study at the Fermilab Booster, [arXiv:2011.07371](https://arxiv.org/abs/2011.07371).'
```

## Add a Talk
Add talks to the yaml list in `_data/talks.yml`

Note that this will render markdown syntax. Putting quotes around the title (single or double) is reccomended to avoid issues with yaml parsing.

**Example:**
```yaml
- "C. Herwig, An ML Control System for the Fermilab Booster, BIDS Machine Learning and Science Forum, April 2021, [abstract](https://bids.berkeley.edu/events/machine-learning-and-science-forum-2021-0405)"
```

## Add a Sponsor
Add sponsors to the yaml list in `_data/sponsors.yml`

Note that this will **not** render markdown syntax. Entries for both `name` and `image` fields are required (see example). Note that file paths for images should include a leading `/` to indicate an absolute path from the root of the site.


```yaml
- name: National Science Foundation
  image: /images/nsf.png
```