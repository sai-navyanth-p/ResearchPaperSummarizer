from flask import Flask, jsonify
import subprocess
import threading
import torch
import os

app = Flask(__name__)

training_lock = threading.Lock()

# Path to training script
TRAINING_SCRIPT = "/mnt/block/train_bert_mlm.py"


def run_training(command):
    with training_lock:
        try:
            print(f"Starting training with command: {command}")
            subprocess.run(command, shell=True, check=True)
            print("Training complete.")
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")


@app.route("/trigger-training-single", methods=["POST"])
def trigger_training_single():
    if not torch.cuda.is_available():
        return jsonify({"error": "CUDA is not available"}), 400

    thread = threading.Thread(target=run_training, args=(f"python3 {TRAINING_SCRIPT}",))
    thread.start()
    return jsonify({"status": "Training started on single GPU"})


@app.route("/trigger-training-multiple", methods=["POST"])
def trigger_training_multiple():
    if torch.cuda.device_count() < 2:
        return jsonify({"error": "Multiple GPUs not available"}), 400


    num_gpus = torch.cuda.device_count()
    command = f"torchrun --nproc_per_node={num_gpus} {TRAINING_SCRIPT}"
    thread = threading.Thread(target=run_training, args=(command,))
    thread.start()
    return jsonify({"status": f"Training started on {num_gpus} GPUs"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)