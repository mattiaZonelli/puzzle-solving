# Using residual networks for Jigsaw puzzle solving

In this work, we address the jigsaw puzzle solving task, proposing an automated pipeline to assess the adjacency relationship among tiles and order them. In particular, we compare two approaches Relaxation Labeling (ReLab) and Puzzle Solving by Quadratic Programming (PSQP). We train convolutional neural networks (CNNs), trying different methods to extract compatibility between tiles of images, first by approaching the task as a super- vised learning problem and then by using self-supervised learning, a variation of the unsupervised learning theme. We build a CNN trained for a pretext task, which can later be repurposed to extract tiles compatibility. Finally, we test different combinations of CNNs – as automatic feature extractors – and puzzle solving methods on publicly available datasets, providing the feasibility of our proposed method.

Extracted image from master thesis - Performance assessment on compatibility learning
<img width="390" alt="Screenshot 2023-08-20 alle 14 32 42" src="https://github.com/mattiaZonelli/puzzle-solving/assets/22390331/31a47ddf-ff77-4f9f-8fdc-3a4b45a45d59">

ps. this project is the result of my master thesis.
