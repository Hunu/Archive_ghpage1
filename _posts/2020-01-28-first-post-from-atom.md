---
layout: "post"
title: "First Post From Atom"
date: "2020-01-28 21:03"
---

# My First Post from local PC

> Text editor:      [Atom](https://atom.io/)
>
> Markdown add-on:  [markdown-writer](https://atom.io/packages/markdown-writer)

## Trying to insert a image
(With [this](https://github.com/Hunu/hunu.github.io/blob/master/_mdwriter.cson) markdown configuration file)
![testimg](/images/2020/01/testimg.png)
Works like a charm.

## Simplify the process of insert a Screenshot
I've modified the markdown-editor package on my local machine to add a new action.
{% include screenshot url='/images/2020/01/insertscreenshot.png' caption='insertScreenshot'%}

which inserted a screenshot code block like below:

<%raw%>

  {% include screenshot url='/images/2020/01/insertscreenshot.png' caption='insertScreenshot'%}

<%endraw%>

Works like a charm!

## Basic shortcuts of Markdown-writer

### Toggle a task(switch between done/not-done)
Key Mapping: Cmd + Shift + X
- [x] List 1

## Read More

I found a good post about [Using Jekyll with Atom Editor](https://insujang.github.io/2017-04-01/using-jekyll-with-atom-editor/)
