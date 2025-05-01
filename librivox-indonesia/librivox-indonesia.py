# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LibriVox-Indonesia Dataset"""

import csv
import os

import datasets
from datasets.utils.py_utils import size_str

from .languages import LANGUAGES
from .release_stats import STATS

_CITATION = """\
"""

_HOMEPAGE = "https://huggingface.co/indonesian-nlp/librivox-indonesia"

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"

_DATA_URL = "https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia/resolve/main/data"


class LibriVoxIndonesiaConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriVoxIndonesia."""

    def __init__(self, name, version, **kwargs):
        self.language = kwargs.pop("language", None)
        self.release_date = kwargs.pop("release_date", None)
        self.num_clips = kwargs.pop("num_clips", None)
        self.num_speakers = kwargs.pop("num_speakers", None)
        self.total_hr = kwargs.pop("total_hr", None)
        self.size_bytes = kwargs.pop("size_bytes", None)
        self.size_human = size_str(self.size_bytes)
        description = (
            f"LibriVox-Indonesia speech to text dataset in {self.language} released on {self.release_date}. "
            f"The dataset comprises {self.total_hr} hours of transcribed speech data"
        )
        super(LibriVoxIndonesiaConfig, self).__init__(
            name=name,
            version=datasets.Version(version),
            description=description,
            **kwargs,
        )


class LibriVoxIndonesia(datasets.GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "_all_"

    BUILDER_CONFIGS = [
        LibriVoxIndonesiaConfig(
            name=lang,
            version=STATS["version"],
            language=LANGUAGES[lang],
            release_date=STATS["date"],
            num_clips=lang_stats["clips"],
            num_speakers=lang_stats["users"],
            total_hr=float(lang_stats["totalHrs"]) if lang_stats["totalHrs"] else None,
            size_bytes=int(lang_stats["size"]) if lang_stats["size"] else None,
        )
        for lang, lang_stats in STATS["locales"].items()
    ]

    def _info(self):
        total_languages = len(STATS["locales"])
        total_hours = self.config.total_hr
        description = (
            "LibriVox-Indonesia is a speech dataset generated from LibriVox with only languages from Indonesia."
            f"The dataset currently consists of {total_hours} hours of speech "
            f"in {total_languages} languages, but more voices and languages are always added."
        )
        features = datasets.Features(
            {
                "path": datasets.Value("string"),
                "language": datasets.Value("string"),
                "reader": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=44100)
            }
        )

        return datasets.DatasetInfo(
            description=description,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=self.config.version,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager.download_config.ignore_url_params = True
        audio_path = {}
        local_extracted_archive = {}
        metadata_path = {}
        split_type = {"train": datasets.Split.TRAIN, "test": datasets.Split.TEST}
        for split in split_type:
            audio_path[split] = dl_manager.download(f"{_DATA_URL}/audio_{split}.tgz")
            local_extracted_archive[split] = dl_manager.extract(audio_path[split]) if not dl_manager.is_streaming else None
            metadata_path[split] = dl_manager.download_and_extract(f"{_DATA_URL}/metadata_{split}.csv.gz")
        path_to_clips = "librivox-indonesia"

        return [
            datasets.SplitGenerator(
                name=split_type[split],
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive[split],
                    "audio_files": dl_manager.iter_archive(audio_path[split]),
                    "metadata_path": dl_manager.download_and_extract(metadata_path[split]),
                    "path_to_clips": path_to_clips,
                },
            ) for split in split_type
        ]

    def _generate_examples(
        self,
        local_extracted_archive,
        audio_files,
        metadata_path,
        path_to_clips,
    ):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        metadata = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.config.name == "_all_" or self.config.name == row["language"]:
                    row["path"] = os.path.join(path_to_clips, row["path"])
                    # if data is incomplete, fill with empty values
                    for field in data_fields:
                        if field not in row:
                            row[field] = ""
                    metadata[row["path"]] = row
        id_ = 0
        for path, f in audio_files:
            if path in metadata:
                result = dict(metadata[path])
                # set the audio feature and the path to the extracted file
                path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                result["audio"] = {"path": path, "bytes": f.read()}
                result["path"] = path
                yield id_, result
                id_ += 1
