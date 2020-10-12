Plain HTML website published to https://1000sharks.xyz with GitLab pages - for MUMT 618 final project, Fall 2020.

# 1000 sharks

1000 sharks is my exploration of machine learning with the intention of creating a fake music artist website.

It uses SampleRNN for music generation, and StyleGAN2 for album art generation.

### Project organization

```
$ tree -L 1
.
├── LICENSE             # MIT license
├── paper-presentation  # latex files for presentation on neural audio synthesis
├── public              # website files for https://1000sharks.xyz
├── README.md           # this readme describing the project breakdown
├── samplernn-scripts   # scripts for reproducing my SampleRNN results
└── stylegan2-scripts   # scripts for reproducing my StyleGAN2 results
```

### ML models

The machine learning models used are:
* https://github.com/sevagh/dadabots_sampleRNN for music generation
* https://github.com/sevagh/prism-samplernn for (less successful) music generation
* https://github.com/sevagh/tensorflow-wavenet (for wavenet deep dive)
* https://github.com/sevagh/stylegan2 for album art generation

N.b. that the links above are all my own forks of the original models, but I have only made some very minor quality-of-life commits (and no other significant modifications from the forks).

The reason I forked them all is to ensure they stick around for future reference from this repo.
