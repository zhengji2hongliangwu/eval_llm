
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FlexibleDataLoader

data_loader = FlexibleDataLoader(
    dataset_name="general_qa",
    dataset_path="/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output/gesture_1013.jsonl",
)
data = data_loader.load_data()
assert len(data) > 0
print(data[0])
