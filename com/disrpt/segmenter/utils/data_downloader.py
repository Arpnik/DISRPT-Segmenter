from pathlib import Path

import requests, os
import warnings
warnings.filterwarnings("ignore")

class DatasetDownloader:
    """Download eng.erst.gum dataset from DISRPT 2025"""

    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.base_url = "https://raw.githubusercontent.com/disrpt/sharedtask2025/refs/heads/master"
        self.corpus_name = ["eng.erst.gum", "eng.dep.scidtb", "eng.rst.oll", "eng.rst.sts" ,"eng.rst.umuc" ,
                            "eng.sdrt.msdc" , "eng.sdrt.stac"]
        self.dataset_dir.mkdir(exist_ok=True)

    def download_file(self, file_path, local_path):
        """Download a single file from GitHub"""
        url = f"{self.base_url}/{file_path}"
        print(url)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            local_file = self.dataset_dir / local_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_bytes(response.content)
            print(f"✓ Downloaded: {local_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to download {file_path}: {e}")
            return False

    def count_files_in_dataset(self):
        total_files = 0

        for corpus in self.corpus_name:
            folder_name = corpus.split(".")[-1]
            folder_path = Path(self.dataset_dir) / folder_name

            if folder_path.exists() and folder_path.is_dir():
                # Count files recursively inside that folder
                file_count = sum(len(files) for _, _, files in os.walk(folder_path))
                print(f"{folder_name}: {file_count} files")
                total_files += file_count
            else:
                print(f"⚠️ Skipping {folder_name} (folder not found)")

        print(f"\nTotal files across selected corpora: {total_files}")
        return total_files

    def download_corpus(self):
        """Download train, dev, test files for eng.erst.gum"""
        print("=" * 70)
        print("Downloading eng.erst.gum dataset from DISRPT 2025")
        print("=" * 70)

        files_downloaded = []
        splits = ["train", "dev", "test"]

        for corpus in self.corpus_name:
            #check if corpus is already downloaded
            folder_name = corpus.split(".")[-1]
            if Path(self.dataset_dir.joinpath(folder_name)).exists():
                continue

            for split in splits:
                print(f"\n📥 Downloading {split} split...")

                # Download .conllu files (with full annotations including Seg labels)
                conllu_file = f"data/{corpus}/{corpus}_{split}.conllu"
                conllu_success = self.download_file(conllu_file, f"{folder_name}/{split}.conllu")

                if conllu_success:
                    files_downloaded.append(split)
                    print(f"✓ {folder_name}: {split} split complete")

            if len(files_downloaded)%3 == 0:
                print(f"\n✅ {folder_name}: Download complete! All splits ready for training.")
            else:
                print(f"\n⚠️ Only {len(files_downloaded)} splits downloaded successfully.")

        count = self.count_files_in_dataset()
        print()
        return count == 3 * len(self.corpus_name)
