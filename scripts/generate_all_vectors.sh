# Create venv and install requirements if not already done
if [ ! -d "venv" ]; then
    echo "Creating venv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b"
python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" --use_base_model
python generate_vectors.py --layers $(seq 0 39) --save_activations --model_size "13b"