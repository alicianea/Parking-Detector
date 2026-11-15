from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ParkingDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
         # Convert the dataset directory into a Path object.
        self.data_dir = Path(data_dir)
        # Store the optional image preprocessing pipeline.
        self.transform = transform

        # Store image paths and labels without loading all images.
        # Images are loaded individually when requested.
        self.samples = []

        # Construct the path to the requested label file.
        label_path = self.data_dir / "LABELS" / label_file

        # Ensure the label file exists before attempting to read it.
        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_path}"
            )

        with open(label_path, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                # rsplit(maxsplit=1) separates the label from
                # the image path using the final whitespace.
                image_path, label = line.rsplit(maxsplit=1)

                # Construct the full path to the image.
                full_path = self.data_dir / "PATCHES" / image_path

                # Store the image path and integer class label.
                self.samples.append(
                    (full_path, int(label))
                )

         # Display the number of samples loaded from the label file.
        print(
            f"Loaded {len(self.samples)} images "
            f"from {label_file}"
        )

    # Return the total number of samples in the dataset.
    def __len__(self):
        return len(self.samples)

    # Retrieve a parking-space image and its occupancy label.
    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
