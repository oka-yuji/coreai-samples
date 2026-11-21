import json
from pathlib import Path
import shutil
import subprocess
import unittest
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "CoreAISamples.xcodeproj" / "xcshareddata" / "xcschemes"
APP_SCHEMES = (
    "CoreAISamples",
    "CoreAISamples-Debug",
    "CoreAISamples-iOS",
    "CoreAISamples-ImageTests",
    "SpeechPhone",
)


class MetalRunSchemeTests(unittest.TestCase):
    def test_app_run_actions_disable_metal_api_validation(self):
        for name in APP_SCHEMES:
            with self.subTest(scheme=name):
                launch = ElementTree.parse(SHARED / f"{name}.xcscheme").getroot().find("LaunchAction")
                self.assertIsNotNone(launch)
                self.assertEqual(launch.get("enableGPUValidationMode"), "1")

    def test_project_run_settings_disable_metal_api_validation(self):
        if xcodegen := shutil.which("xcodegen"):
            command = [xcodegen, "dump", "--spec", "project.yml", "--type", "parsed-json"]
        else:
            command = [
                "/usr/bin/ruby", "--disable-gems", "-rpsych", "-rjson", "-e",
                "puts JSON.generate(Psych.safe_load(File.read(ARGV.fetch(0))))", "project.yml",
            ]
        spec = json.loads(subprocess.check_output(command, cwd=ROOT))
        for name in APP_SCHEMES:
            with self.subTest(scheme=name):
                run = spec["schemes"][name].get("run") or {}
                self.assertIs(run.get("enableGPUValidationMode"), False)


if __name__ == "__main__":
    unittest.main()
