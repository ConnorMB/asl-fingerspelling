# ASL fingerspelling recognizer

Real-time American Sign Language fingerspelling recognition through a webcam. Built as a follow-up to my [rock-paper-scissors gesture game](https://github.com/ConnorMB/rps-gesture-game), reusing the same hand-tracking approach but scaled up to a full alphabet and rebuilt around a proper PyTorch model.

## How it works

MediaPipe finds 21 points on your hand in each frame. Those points get normalized, shifted so the wrist sits at zero and scaled by hand size, so the same letter looks the same whether your hand is close to the camera or far away. That's the same normalization I used in the RPS project. It already worked, so I kept it as is. A small PyTorch network takes those 63 numbers and predicts which of the 26 letters it is.

I trained it on landmarks pulled from the public [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) on Kaggle, then tested it separately against samples I collected from my own webcam. Those are hand shapes it never saw during training. That gap between the two numbers is the interesting part. Doing well on a clean dataset someone else built is one thing. Holding up against a real hand in a real room with different lighting is another test entirely.

**Accuracy on held-out Kaggle data:** <fill in from training run>
**Accuracy on my own webcam:** <fill in after evaluate.py>

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
python evaluate.py                   # optional: see real-world accuracy
python main.py                       # run it live
\`\`\`

## Tests

\`\`\`bash
pytest
\`\`\`