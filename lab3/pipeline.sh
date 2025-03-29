pip install -r requirements.txt
echo "Creating datasets..."
python data_creation.py
echo "Preprocessing datasets..."
python data_preprocessing.py
echo "Training the model..."
python model_preparation.py
echo "Testing the model..."
metric=$(python model_testing.py | head -n 1)

echo "$metric"
