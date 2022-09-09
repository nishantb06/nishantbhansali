---
title: "Hello World"
date: 2022-09-08T23:38:21+05:30
draft: true
---
hugo new posts/second.md will create a new post on the website
and then hugo server will reflect that in the local host
write the front matter manually as well if you want to create your own website

commands ```hugo server -D``` and ```hugo server``` are same

to deploy changes to original website use
1. hugo -D
2. go to public repo and commit and push changes from there to nishantb06.github.io repo

- to add images dont add them in static folder but use a url instead

- go to [this](https://www.remove.bg/upload) site. Upload your image there. right click and copy its image url

- hyde documentation
- https://blog.hellohuigong.com/en/posts/how-to-build-personal-blog-with-github-pages-and-hugo/ 

![test image with link](https://imgs.search.brave.com/4RmOOOTM0uWDe6eXxblKpb_CNTcUnlxl43AhhZuWkMs/rs:fit:844:225:1/g:ce/aHR0cHM6Ly90c2Ux/Lm1tLmJpbmcubmV0/L3RoP2lkPU9JUC5N/OUFzWjdTbTZRcS1M/WHBZOTJUdDJBSGFF/SyZwaWQ9QXBp)

this was made with a shell codeblock in markdown
```Shell
dsfsdfsd - with shell
```
without a shell codeblock
```
dsfsdfsd - with shell
```