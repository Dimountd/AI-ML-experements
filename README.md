# Insurance Benchmark

This is the final version of the benchmark for the InsuranceQA dataset.

Most work is done in the notebook. To run it:

1.  Open **`benchmark_final.ipynb`**.
2.  Run the cells in order.
    *   It handles dependency installation and setup automatically.
    *   It's designed to run smoothly on Google Colab or a local machine with a good GPU (H100/A100 recommended for the Giga models).

**Dependencies (Installed Automatically):**
The notebook will automatically install the necessary libraries for you. For reference, the main ones are:
*   `qdrant-client` (Vector Database)
*   `sentence-transformers` & `FlagEmbedding` (Models)
*   `accelerate` & `bitsandbytes` (GPU Optimization)
*   `numpy` (pinned to <2.0), `pandas`, `matplotlib` (Data & Math)

**A couple of notes on the project structure:**
*   **Data**: The dataset files (`test.jsonl`, `train.jsonl`) are located in the `data/` folder. The notebook looks for them there.
*   **Source Code**: You might see a `src/` folder. That was just used for the initial development steps and prototyping. It's not really used anymore since the final logic is self-contained in the notebook