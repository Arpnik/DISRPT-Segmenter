from pathlib import Path

import requests
import warnings
warnings.filterwarnings("ignore")

class GUMDatasetDownloader:
    """Download eng.erst.gum dataset from DISRPT 2025"""

    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.base_url = "https://raw.githubusercontent.com/disrpt/sharedtask2025/refs/heads/master"
        self.corpus_name = "eng.erst.gum"
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

    def download_gum_corpus(self):
        """Download train, dev, test files for eng.erst.gum"""
        print("=" * 70)
        print("Downloading eng.erst.gum dataset from DISRPT 2025")
        print("=" * 70)

        files_downloaded = []
        splits = ["train", "dev", "test"]

        for split in splits:
            print(f"\n📥 Downloading {split} split...")

            # Download .tok files (tokenized text)
            tok_file = f"data/{self.corpus_name}/{self.corpus_name}_{split}.tok"
            tok_success = self.download_file(tok_file, f"gum/{split}.tok")

            # Download .conllu files (with full annotations including Seg labels)
            conllu_file = f"data/{self.corpus_name}/{self.corpus_name}_{split}.conllu"
            conllu_success = self.download_file(conllu_file, f"gum/{split}.conllu")

            if tok_success and conllu_success:
                files_downloaded.append(split)
                print(f"✓ {split} split complete")



        if len(files_downloaded) == 3:
            print("\n✅ Download complete! All splits ready for training.")
        else:
            print(f"\n⚠️  Only {len(files_downloaded)} splits downloaded successfully.")

        return len(files_downloaded) == 3
