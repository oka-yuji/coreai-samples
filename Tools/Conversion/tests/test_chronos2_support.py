import copy
from pathlib import Path
import shutil
import tempfile

from Tools.Conversion.adapters.chronos2.artifacts import (
    CORE_CONFIG, FORECAST_CONFIG, REVISION, compare_documents, input_identity, inventory, load_fixtures,
    vector_document,
)
from Tools.Conversion.adapters.chronos2.verify import validate_inspection, verification_record
from Tools.Conversion.storage import file_hash, read_json, write_json


REPO = Path(__file__).resolve().parents[3]
RECIPE = REPO / "Tools/Conversion/recipes/chronos2"


def predictions(index=0):
    return [[[index / 8 + quantile / 2 + step / 32 for step in range(16)] for quantile in range(21)]]


def inspected_asset():
    return {
        "summary": {
            "computeTypes": ["Bool", "Float32", "Int32", "Int64", "UInt32", "UInt64"],
            "storageTypes": [{"name": "Float32", "count": 32}, {"name": "UInt32", "count": 3}],
            "functions": [{"name": "main", "inputs": [{"name": "context", "type": "NDArray (Float32, 1 × ?)"}],
                           "outputs": [{"name": "quantiles", "type": "NDArray (Float32, 1 × 21 × 16)"}]}],
            "compatibilityVersion": "27.0",
        },
        "specializations": [],
        "metadata": {"producer": "coreai-core 1.0.0b3", "author": "Amazon", "license": "Apache-2.0",
                     "creationDate": "2026-11-11T00:00:00Z",
                     "creatorDefinedMetadata": {"source_revision": REVISION}},
        "path": str(REPO / ".local/private/Model.aimodel"),
    }


class ChronosFiles:
    def setUp(self):
        scratch = Path(tempfile.gettempdir()).resolve()
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recipe = self.root / "recipe"
        shutil.copytree(RECIPE, self.recipe)
        self.model_path = self.recipe / "model.json"
        self.fixture_path = self.recipe / "fixtures.json"
        self.fixtures = load_fixtures(self.fixture_path)
        self.snapshot = self.root / REVISION
        self.snapshot.mkdir()
        self.config = {**copy.deepcopy(CORE_CONFIG), "chronos_config": copy.deepcopy(FORECAST_CONFIG),
                       "torch_dtype": "float32", "transformers_version": "4.49.0"}
        write_json(self.snapshot / "config.json", self.config)
        (self.snapshot / "model.safetensors").write_bytes(b"synthetic checkpoint for hash-only tests")
        (self.snapshot / "README.md").write_text("Official model card stand-in for CPU tests.\n")
        self.contract = read_json(self.model_path)
        self.contract["files"] = {name: file_hash(self.snapshot / name) for name in self.contract["files"]}
        write_json(self.model_path, self.contract)
        self.identity = input_identity(self.model_path, self.fixture_path)
        self.ready = self.root / "ready.json"
        write_json(self.ready, {"ready": True, "identity": self.identity,
                                "source_files": self.contract["files"], "setup": {}})
        self.values = [predictions(index) for index in range(18)]
        self.reference = vector_document("reference", self.identity, self.fixtures, self.values)
        self.reference_path = self.root / "reference.json"
        write_json(self.reference_path, self.reference)

    def make_export(self):
        self.export = self.root / "export"
        asset = self.export / "Model.aimodel"
        (asset / "functions").mkdir(parents=True)
        (asset / "functions/main.mlirb").write_bytes(b"synthetic asset")
        (asset / "metadata.json").write_text("{}\n")
        wrapper = vector_document("wrapper", self.identity, self.fixtures, self.values)
        write_json(self.export / "wrapper-actuals.json", wrapper)
        write_json(self.export / "wrapper-comparison.json", compare_documents(
            self.reference, wrapper, self.identity, self.fixtures, kind="wrapper"))
        write_json(self.export / "export.json", {
            "format": "coreai-samples-chronos2-export/1", "identity": self.identity,
            "reference": file_hash(self.reference_path), "asset": inventory(asset),
            "wrapper_actuals": file_hash(self.export / "wrapper-actuals.json"),
            "wrapper_comparison": file_hash(self.export / "wrapper-comparison.json"),
        })
        return self.export

    def make_verified(self):
        self.make_export()
        self.verified = self.root / "verified"
        self.verified.mkdir()
        actuals = vector_document("actuals", self.identity, self.fixtures, self.values)
        raw = inspected_asset()
        inspection = validate_inspection(raw)
        write_json(self.verified / "actuals.json", actuals)
        write_json(self.verified / "inspection-raw.json", raw)
        write_json(self.verified / "inspection.json", inspection)
        write_json(self.verified / "comparison.json", verification_record(
            self.reference_path, self.reference, actuals, self.identity, self.fixtures,
            inventory(self.export / "Model.aimodel"), inspection))
        return self.verified
