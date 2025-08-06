import os
import json


def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_directory, "generated_events")
    os.makedirs(output_folder, exist_ok=True)

    batch_sizes = [1, 8, 16, 32, 64, 128, 256]
    models = ["resnet-18", "resnet-34", "resnet-50", "resnet-101", "resnet-152"]

    count = 0
    for model in models:
        for batch_size in batch_sizes:
            event_payload = {
                "images": [],
                "batch_size": batch_size,
                "use_random_image": True,
                "num_random_images": 256,
                "model_name": model,
            }

            event_file_path = os.path.join(output_folder, f"event_random{count}.json")

            with open(event_file_path, "w") as event_file:
                json.dump(event_payload, event_file, indent=4)

            count += 1

            print(f"Event file generated at: {event_file_path}")


if __name__ == "__main__":
    main()
