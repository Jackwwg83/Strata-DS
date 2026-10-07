# 核查来源及使用范围

读取/核对日期：2026-10-04。主链接用于定位；真实实验必须使用specs/sources.lock.json的immutable revision/digest。未解析的项目保持null，不以main替代锁定。

S1. coolbho3k模型卡：active/Engram/draft载荷、量化范围、质量指标和定制运行时边界。
https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw

S2. 双Spark配方：版本锁及Engram无损布局/AGPL边界。本轮读取recipe-lock与Engram说明，取得head 1d8ac64af01c6fec87f39eb1dd526ff183615c73。
https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/recipe-lock.json
https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/1d8ac64af01c6fec87f39eb1dd526ff183615c73/docs/engram-io-performance.md

S3. antirez模型卡：151.77GiB Q2主干、188.83GiB Engram与量化配方；卡片保留旧branch措辞，运行时应以支持V4.1的固定commit为准。
https://huggingface.co/antirez/deepseek-v4.1-flash-gguf

S4. 上轮已读取DwarfStar main相关资料与head 0aaea5a238fb41a35106a551e73c8409dfb751ac。本包把它作为待独显qual的参考，不宣布本轮重新编译或测试。
https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/docs/MODELS.md
https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/docs/DGX_SPARK.md

S5. SAGE模型卡/上轮容量复算，扩展项。精确revision和quality仍需重新核验。
https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw

S6. Strata PLE/专家分层机制，源于本对话上轮已读取的源码。仅借鉴机制，本包未复制其源代码。
https://github.com/Niko1221/Strata/blob/main/include/strata/ngram/ple_reader.hpp
https://github.com/Niko1221/Strata/blob/main/docs/UNSLOTH_Q4.md

S7. Vast CLI生命周期、停止/销毁和存储计费边界；本轮实际读取。
https://docs.vast.ai/cli/hello-world

S8. Vast搜索字段及其单位、费用参数；本轮实际读取。
https://docs.vast.ai/api-reference/search/search-offers

S9. Linux cgroup v2内存、页缓存、swap与I/O控制；本轮实际读取。
https://docs.kernel.org/admin-guide/cgroup-v2.html

S10. 本对话v0.1压缩包与128GiB复核文件：用于迁移检查和既有数字，不能作为外部模型正确性证明。本轮解压核查checkpoint.py、model.py、weights.py、cache.py、runtime.py、provenance.json。

所有本包预算与实验排序是工程设计。没有下载完整权重、没有运行GPU或读取用户Vast账户。不要把来源中的公开硬件成绩当成本包实测。
