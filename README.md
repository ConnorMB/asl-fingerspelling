# ASL fingerspelling recognizer

Real time asl fingerspelling recognition through a webcam. Built as a follow up to my [rock paper scissors gesture game](https://github.com/ConnorMB/rps-gesture-game), reusing the same hand tracking approach but scaled up to a full alphabet and rebuilt around a proper PyTorch model.

## How it works

MediaPipe finds 21 points on your hand in each frame. Those points get normalized, shifted so the wrist sits at zero and scaled by hand size, so the same letter looks the same whether your hand is close to the camera or far away. That's the same normalization I used in the RPS project. It already worked, so I kept it as is. A small PyTorch network takes those 63 numbers and predicts which of the 26 letters it is.

I trained it on landmarks pulled from the public [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) on Kaggle, holding out a slice of that data to track accuracy during training. I then tested the finished model separately against samples of my own hand, recorded from my own webcam, that it never saw during training.

Two letters, J and Z, involve movement rather than a fixed handshape, and this whole system only ever looks at one still frame at a time. I didn't collect real-world samples for J. I did test Z anyway, out of curiosity, and it failed completely, which is exactly what you'd expect from a system with no concept of motion.

## Results

**Validation accuracy on held-out Kaggle data:** 97.82%
**Accuracy on my own webcam:** 59.19%

That gap is the actual interesting part of this project. A model that does well on a clean, single-source dataset and a model that holds up against a real hand in a real room are two different things, and the difference between those two numbers is the honest answer to "does this actually work."

Breaking it down by letter shows the gap isn't random, most letters that failed have a real explanation:

| Letter | Accuracy | Letter | Accuracy | Letter | Accuracy |
|---|---|---|---|---|---|
| F | 0% | U | 38% | D | 93% |
| K | 0% | G | 67% | Q | 93% |
| P | 0% | A | 70% | X | 96% |
| T | 0% | O | 89% | B, C, E, H, I, L, R, W, Y | 100% |
| Z | 0% (motion) | | | | |
| N | 5% | | | | |
| V | 11% | | | | |
| M | 18% | | | | |
| S | 36% | | | | |

Most of the weak letters land in two well-documented confusion groups, not random noise. M, N, S, and T all involve subtle differences in thumb position on a closed fist, and are commonly confused even by human ASL learners for exactly that reason. K, P, U, and H share the same finger configuration (index and middle finger extended) and differ mainly by which way the hand is rotated, another well documented source of confusion. checking what the model actually guessed instead of the correct letter backs this up: N was mostly read as M, T was mostly read as A, and U was mostly read as R, all inside those same documented confusion groups.

F is the one result that doesn't fit a known confusion pair, and when I looked through the raw Kaggle training images myself, F's photos (along with T, P, and K) looked visibly off compared to standard ASL references. So the real world gap looks like it comes from two separate causes stacked together: genuinely similar handshapes that are hard to tell apart even for people, plus a smaller number of letters where this specific dataset's training images may not be fully standard.

## Setup

\`\`\`bash
git clone <this repo>
cd asl-fingerspelling
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Usage

Download the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and unzip it into the project root, then:

\`\`\`bash
python prepare_dataset.py
python train_model.py
python collect_validation_data.py   # optional: build your own held-out test set
python evaluate.py                   # optional: see real-world accuracy, letter by letter
python main.py                       # run it live
\`\`\`

## Tests

\`\`\`bash
pytest
\`\`\`