# A Comprehensive Survey of Video and Multimodal Generation: Foundations, Architectures, Control, and Evaluation

## Executive Summary / TL;DR

Video and multimodal generation has undergone a profound paradigm shift over the past three years, transitioning from early pixel-space models and rigid 3D U-Nets to highly scalable Diffusion Transformers (DiTs), causal tokenizers, and multimodal foundational architectures [1][2][3]. Driven by breakthroughs in space-time latent compression, dense multi-modal captioning, and efficient open-source training paradigms, modern video generators are capable of producing hyper-realistic, physically coherent video sequences spanning multiple seconds to minutes [4][5][6]. 

This survey provides a rigorous, thematic synthesis of contemporary video and multimodal generation. We categorize foundations into diffusion-based and autoregressive architectures, examine advanced multi-modal conditioning mechanisms (such as text-to-video semantic alignment and structural controls like Canny/Depth/Pose), and analyze recent open-source democratizing efforts (e.g., CogVideoX, Open-Sora 2.0) alongside rigorous automatic and human-aligned benchmarking suites (VBench++, VideoScore) [4][5][6][7][8]. Our analysis highlights critical open challenges—including temporal consistency, computational scalability, and physical world modeling—and outlines pathways toward general-purpose interactive world simulators.

## Background & Motivation

The automated synthesis of dynamic visual content is one of the most compelling frontiers in generative artificial intelligence. Early video generation models were constrained by severe computational bottlenecks, low spatial resolutions, and noticeable temporal flickering, typically relying on pixel-space 3D convolutions or restricted frame interpolation networks [1]. 

The maturation of Latent Diffusion Models (LDMs) and Large Language Models (LLMs) catalyzed a transformative evolution [2][3]. By compressing high-dimensional video streams into compact latent spaces using 3D spatial-temporal VAEs, models circumvented the exorbitant memory overhead of pixel-space operations [2][5]. Simultaneously, the integration of robust semantic text encoders (such as T5 and mT5-XXL) and dense video recaptioning strategies enabled unprecedented alignment between human prompts and synthesized dynamics [4][3]. Today, video generation is no longer viewed merely as an extension of image generation, but as a core pillar of multimodal intelligence and interactive world simulation [1][9].

## Thematic Synthesis: Architectural Paradigms in Video Generation

### Diffusion Transformers vs. 3D U-Nets

The architectural backbone of video generation has migrated decisively from hierarchical 3D U-Nets toward scalable Diffusion Transformers (DiTs) [2][3]. 
* **Early Cascaded and Pseudo-3D U-Nets**: Pioneering frameworks such as Make-A-Video and Imagen Video relied on spatial-temporal convolution layers attached to pre-trained text-to-image base models, followed by cascaded spatial and temporal super-resolution modules [1]. While effective at preserving single-frame fidelity, these architectures struggled with long-range temporal dependencies and global motion coherence.
* **Latent Diffusion and Factorized Space-Time Attention**: Stable Video Diffusion (SVD) established robust three-stage training paradigms (image pretraining, video pretraining, and high-resolution latent finetuning) [10]. To scale efficiently, architectures like Open-Sora introduced Spatial-Temporal Diffusion Transformers (STDiT) that factorize spatial attention (within individual frames) and temporal attention (across frames) [3].
* **Joint Spacetime Latent Patches**: Inspired by Sora and advanced open-source models, modern systems represent videos as collections of spacetime latent patches, enabling native handling of variable durations, resolutions, and aspect ratios without fixed-length constraints [3][9]. Furthermore, CogVideoX employs 3D full attention (text-video hybrid attention) alongside an expert transformer to jointly model spatial and temporal dynamics without sacrificing efficiency [5].

### Autoregressive Video Modeling and World Simulators

While diffusion models dominate continuous high-fidelity generation, autoregressive frameworks remain essential for interactive and sequential modeling:
* **Tokenization and Causal Modeling**: Models like VideoGPT utilize 3D VQ-VAE/VQGAN architectures to discretize video streams into spatial-temporal tokens, which are subsequently modeled using causal transformers [1]. Phenaki extends this by leveraging a C-ViViT encoder-decoder for variable-length video generation via causal attention [1].
* **Interactive World Models**: Genie exemplifies the convergence of video generation and interactive environment modeling by combining spatiotemporal transformers, novel video tokenizers, and causal action models to predict next-frame dynamics conditioned on discrete user actions [1].

## Multimodal Conditioning, Text-to-Video Alignment, and Structural Control

Precise control over generated video content requires sophisticated multimodal alignment and conditioning architectures:

### Semantic Text-to-Video Alignment
Modern video generators rely heavily on advanced text encoders such as T5-XXL and mT5-XXL to capture complex semantic instructions, spatial relationships, and temporal narratives [4][3]. To mitigate the noisy and sparse nature of raw video captions, pipelines adopt DALL-E 3-style dense recaptioning, generating highly detailed descriptions of actions, camera movements, and subject attributes [9].

### Structural and Spatial Control Signals
Beyond textual prompts, precise user control is achieved via dedicated condition controllers:
* **Image-to-Video and Continuation**: Frame-level condition controllers handle image-to-video (I2V), video transitions, and seamless sequence continuation by concatenating conditioning latents and masks into the denoiser [4].
* **Explicit Motion and Structural Guidance**: Methods like Motion-I2V incorporate explicit motion modeling to maintain consistency across frames [11]. Similarly, Open-Sora Plan utilizes Structure Condition Controllers to extract high-level representations from auxiliary guidance signals—such as Canny edges, depth maps, and sketches—injecting them directly into transformer blocks [4].
* **Score Conditioning and Audio-Visual Adaptation**: Open-Sora incorporates aesthetic scores, motion scores, and camera motion descriptors (e.g., "pan left") to allow inference-time control [3]. Furthermore, audio-visual adaptations leverage lightweight adapter networks to synchronize audio streams with corresponding video dynamics [12].

## Open-Source Ecosystems, Efficiency, and Evaluation Methodologies

### Democratizing Video Generation: Open-Source Breakthroughs
The video generation landscape has experienced a major democratization wave led by powerful open-source initiatives:
* **CogVideoX**: Features a 3D causal VAE for superior compression, mixed-duration progressive training, and temporal context parallelism to ensure smooth motion and scalable training [5].
* **Open-Sora 2.0**: Demonstrates that commercial-level video generation models can be trained for roughly $200k by leveraging an 11B parameter image pre-trained backbone (derived from FLUX) combined with T5-XXL and CLIP-Large, achieving performance metrics competitive with leading closed-source commercial systems [6].

### Evaluation Benchmarks and Automated Human Feedback
Evaluating video generation quality requires multi-dimensional and robust evaluation suites:
* **VBench++**: Dissects video generation performance into 16 fine-grained evaluation dimensions (including subject identity inconsistency, motion smoothness, temporal flickering, and spatial relationships) with dedicated support for both text-to-video and image-to-video tasks [7].
* **VideoScore**: Addresses the cost and latency of human evaluation by training an automated metric (built upon the Mantis-8B-Idefics2 backbone and trained on the VideoFeedback dataset) to simulate fine-grained human preferences via regression scoring, serving as an effective proxy for reinforcement learning with human feedback (RLHF) [8].

## Trends, Open Problems, and Future Outlook

Despite remarkable progress, several critical challenges remain at the forefront of video and multimodal research:
1. **Physical World Consistency**: While models generate visually stunning outputs, they frequently violate fundamental physical laws (e.g., object persistence, gravity, and fluid dynamics). Developing true world simulators that understand physical constraints remains an open problem [9].
2. **Long-Video Generation**: Maintaining semantic and identity consistency over extended durations (minutes to hours) without error accumulation requires advanced memory banks, hierarchical autoregressive rolling windows, and long-range attention mechanisms.
3. **Computational Efficiency**: Training and inference costs remain exceptionally high, necessitating continued innovations in hardware-aware optimizations, sequence parallelism, and compressed latent architectures [5][6].

## References
[1] From Sora What We Can See: A Survey of Text-to-Video Generation. web. https://arxiv.org/html/2405.10674v1 (2024-05-17)
[2] Video Diffusion Models: A Survey. web. https://arxiv.org/abs/2405.03150 (2024-11-17)
[3] Open-Sora: Democratizing Efficient Video Production for All. web. https://arxiv.org/html/2412.20404 (2024-12-01)
[4] Open-Sora Plan: Open-Source Large Video Generation Model. arxiv. https://arxiv.org/abs/2412.00131 (2024-11-28)
[5] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2025-03-26)
[6] Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k. hf-daily. https://huggingface.co/papers/2503.09642 (2025-03-12)
[7] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-daily. https://huggingface.co/papers/2411.13503 (2024-11-20)
[8] Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation. web. https://aclanthology.org/2024.emnlp-main.127/ (2024-11-01)
[9] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. web. https://arxiv.org/html/2402.17177v3 (2024-02-27)
[10] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. hf-search. https://huggingface.co/papers/2311.15127 (2023-11-25)
[11] Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling. hf-search. https://huggingface.co/papers/2401.15977 (2024-01-29)
[12] Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation. hf-search. https://huggingface.co/papers/2309.16429 (2023-09-28)
