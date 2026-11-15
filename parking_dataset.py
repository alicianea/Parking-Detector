
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ParkingDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        label_path = self.data_dir / "LABELS" / label_file

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_path}"
            )

        with open(label_path, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                image_path, label = line.rsplit(maxsplit=1)

                full_path = self.data_dir / "PATCHES" / image_path

                self.samples.append(
                    (full_path, int(label))
                )

        print(
            f"Loaded {len(self.samples)} images "
            f"from {label_file}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
