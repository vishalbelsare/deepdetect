# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.30.0](https://github.com/jolibrain/deepdetect/compare/v0.29.0...v0.30.0) (2026-08-26)


### Features

* add external pytorch model import skill ([3ec64fc](https://github.com/jolibrain/deepdetect/commit/3ec64fcd7981d45c622a1cc31030bd7472197431))
* add tensor-backed pytorch worker dataset adapter ([f8c4ec9](https://github.com/jolibrain/deepdetect/commit/f8c4ec9773a4f4d4945091935c954882fcac1e09))
* added vitpose eval losses ([b27951b](https://github.com/jolibrain/deepdetect/commit/b27951b17115e88bdca1f84e2d6073779c4cf0f0))
* additions to pytorch backends to accomodate new detection models ([02e43ea](https://github.com/jolibrain/deepdetect/commit/02e43eac4e3080e7182d22a923b716c409bd0505))
* C++ data augmentation wired to input pull connnector ([0d85ac4](https://github.com/jolibrain/deepdetect/commit/0d85ac468cd421fe513adfd3167a9fc6c2fcb13f))
* **cli:** repository cleansing override ([fa0a840](https://github.com/jolibrain/deepdetect/commit/fa0a8406e725cb497917ef26ba007d9c10b941be))
* hardening of pytorch backend ([622da53](https://github.com/jolibrain/deepdetect/commit/622da53a3549a9ee67142da2d1e8704a92e5f501))
* initial keypoint pose estimator input connector ([687ac8b](https://github.com/jolibrain/deepdetect/commit/687ac8b38668229b1b1aad217d4ab8c2744841a1))
* initial pytorch backend structure and messaging layers ([c67555c](https://github.com/jolibrain/deepdetect/commit/c67555c0513b3b1feb5a04f8322a984645c9e5c5))
* input c++ connector to pytorch backend pull mode ([aa6485f](https://github.com/jolibrain/deepdetect/commit/aa6485fd7acf8b5fc09df8a3ce31f9490b8aac9c))
* map equivalence between python worker and C++ implementations ([7bc6cdb](https://github.com/jolibrain/deepdetect/commit/7bc6cdbb0ce4884387ad0f0c9fb59990da824ca7))
* model flops estimate ([3526835](https://github.com/jolibrain/deepdetect/commit/35268353f8d64096a1f4ce0f3f1058100f9bfdf7))
* model flops estimate ([d4f6148](https://github.com/jolibrain/deepdetect/commit/d4f6148d0b4723d5e8269d7be05b5dc4d3c260dd))
* **model:** initial vitpose with hungarian for slot assignment ([b703e02](https://github.com/jolibrain/deepdetect/commit/b703e0204b8d185fb8c2897e8fdf0121748657c9))
* **model:** regular vitpose with per object keypoints ([cec4791](https://github.com/jolibrain/deepdetect/commit/cec47915cd916d686c0246cfcb3225b49a3352fa))
* **model:** RT-DETRv4 for pytorch backend ([086bacc](https://github.com/jolibrain/deepdetect/commit/086baccb0f087a826e817286b357bbc1ccc51a61))
* per-class map in backend and python workers metrics ([1b44cff](https://github.com/jolibrain/deepdetect/commit/1b44cff452e76148fab4bc5418aaa4f8dd317682))
* sam2 pytorch worker, inference only ([9d340ce](https://github.com/jolibrain/deepdetect/commit/9d340ce9bddf5e63e67b50c8d96afa6753c15871))
* **sdk:** initial python backend sdk ([a4a17e6](https://github.com/jolibrain/deepdetect/commit/a4a17e674001c1765b1c01d5b4781f7dfc5ee597))
* **sdk:** initial torch/torchvision python backend sdk example ([bfd862c](https://github.com/jolibrain/deepdetect/commit/bfd862c73a0fe37bd9fbebd5dee10c74b1a05ba3))
* torch-only reference detector worker and updated documentation ([a20f583](https://github.com/jolibrain/deepdetect/commit/a20f583f5fe587cdfb9557e55470c299ce76dcf5))
* training observability mechanism for agents ([993145e](https://github.com/jolibrain/deepdetect/commit/993145e15dd7147a1baa16a0621fa35211e161b9))
* vit patch_embed resampling from timm in vitpose ([7c4eae3](https://github.com/jolibrain/deepdetect/commit/7c4eae3ce0f2f06945a254c454adc236392acb9f))
* vitpose photometric augmentations ([f2c30f0](https://github.com/jolibrain/deepdetect/commit/f2c30f00fd87ac3c0daf68120bc347dd7e5b0095))
* vitpose slot loss control and balancing ([51eecb1](https://github.com/jolibrain/deepdetect/commit/51eecb1095cfb7535e46ab3687dc5a4e40627277))


### Bug Fixes

* add frontmatter to DD skill ([ec4b612](https://github.com/jolibrain/deepdetect/commit/ec4b612927a14497968678849e7341e99d56b6a2))
* build with new libs ([a29806a](https://github.com/jolibrain/deepdetect/commit/a29806af16c487db7374d13864bc67f8b8217d7f))
* graceful exit with pytorch workers ([139a060](https://github.com/jolibrain/deepdetect/commit/139a060c2f1d015b3acf056ab82bcbd2565347ae))
* inference vitpose slot keypoint coordinates resizing ([8594c2c](https://github.com/jolibrain/deepdetect/commit/8594c2ca72b15bbefee59daaa85544afbfb7d54c))
* missing vitpose files ([12a859b](https://github.com/jolibrain/deepdetect/commit/12a859b7c1112cb675fbb294a821ea55715a74cd))
* pytorch backend map ([0be18e1](https://github.com/jolibrain/deepdetect/commit/0be18e182a6767a71e0f0ec7f0eeb89f2d7c737f))
* remove env name from Visdom plot titles so we can compare runs ([725a2ac](https://github.com/jolibrain/deepdetect/commit/725a2ac49d41e2d4b848d29ca0edf3138683282b))
* resume cli when init weights are enabled ([31042b3](https://github.com/jolibrain/deepdetect/commit/31042b33bf8781e01805f99a4b8f137781b4b4ad))
* test metric iterations with pytorch workers ([e80791e](https://github.com/jolibrain/deepdetect/commit/e80791ed7bb4e7a8fd860b7f5c137d428b3600cd))
* vitpose layer wise decay only when backbone is pretrained ([135bb4b](https://github.com/jolibrain/deepdetect/commit/135bb4b7e3f4f0b8375a8cf4a209cc7f79b27867))
* wheel building ([b903d85](https://github.com/jolibrain/deepdetect/commit/b903d8519d783ff83cfe996cfe2492fc94c7ee94))
* wheels tests ([def8918](https://github.com/jolibrain/deepdetect/commit/def8918730d5f5d013c2014af68d57566c0f9c69))
