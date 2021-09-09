Plain HTML website published to https://1000sharks.xyz with GitLab pages - for MUMT 618 final project, Fall 2020.

# 1000 sharks

1000 sharks is my exploration of machine learning with the intention of creating a fake music artist website.

It uses SampleRNN for music generation, and StyleGAN2 for album art generation.

### Project organization

```
$ tree -L 1
.
├── ddsp-scripts          # scripts for reproducing DDSP results
├── LICENSE               # MIT license
├── paper-presentation    # latex files for presentation on neural audio synthesis
├── public                # website files for https://1000sharks.xyz
├── README.md             # this readme describing the project breakdown
├── samplernn-scripts     # scripts for reproducing SampleRNN results
├── samplernn-scripts     # scripts for reproducing SampleRNN results
├── stylegan2-scripts     # scripts for reproducing StyleGAN2 results
└── vendor                # vendored copies of third-party repos
```

### ML models

The machine learning models used are:
* https://github.com/Cortexelus/dadabots_sampleRNN for music generation
* https://github.com/rncm-prism/prism-samplernn for (less successful) music generation
* https://github.com/ibab/tensorflow-wavenet (for wavenet deep dive)
* https://github.com/NVlabs/stylegan2 for album art generation
* https://github.com/magenta/ddsp for Differentiable DSP and clarinet synthesis

I vendored all of the above locally to ensure they stick around for future reference from the 1000sharks project. I also made some minor quality-of-life changes in the vendored copies.
