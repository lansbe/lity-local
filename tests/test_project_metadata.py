import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectMetadataTests(unittest.TestCase):
    def test_base_runtime_dependencies_include_core_and_audio(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        base_dependency_block = pyproject.split("[project.optional-dependencies]", 1)[0]

        # Core + voice (STT/TTS) + the local RAG reranker ship by default so
        # dictation/read-aloud and reranked retrieval work out of the box.
        for package in [
            "ollama",
            "platformdirs",
            "mcp",
            "faster-whisper",
            "numpy",
            "piper-tts",
            "sounddevice",
            "fastembed",
        ]:
            self.assertIn(f'"{package}>=', base_dependency_block)

        # GUI toolkits, image processing and packaging stay optional: they are
        # heavy and/or platform-specific, so they must not leak into the base.
        for package in ["PySide6", "pywebview", "Pillow", "pyinstaller"]:
            self.assertNotIn(f'"{package}>=', base_dependency_block)

    def test_optional_features_are_declared_as_extras(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        expected_extra_headers = [
            "desktop = [",
            "web = [",
            "image = [",
            "image-mlx = [",
            "full = [",
        ]
        for header in expected_extra_headers:
            self.assertIn(header, pyproject)

        # Audio is no longer an extra — it moved into the base dependencies.
        self.assertNotIn("audio = [", pyproject)

        optional_block = pyproject.split("[project.optional-dependencies]", 1)[1]
        for package in ["PySide6-Essentials", "pywebview", "Pillow", "pyinstaller"]:
            self.assertIn(f'"{package}>=', optional_block)

    def test_image_mlx_conflict_is_declared_for_uv(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[tool.uv]", pyproject)
        self.assertIn("conflicts = [", pyproject)
        self.assertIn('{ extra = "image" }', pyproject)
        self.assertIn('{ extra = "image-mlx" }', pyproject)
        self.assertIn('{ extra = "video" }', pyproject)

    def test_dev_extra_includes_document_test_dependencies(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        dev_block = pyproject.split("dev = [", 1)[1].split("]", 1)[0]

        for package in ["pypdf", "python-docx"]:
            self.assertIn(f'"{package}>=', dev_block)

    def test_packaging_scripts_and_pyinstaller_spec_exist(self):
        self.assertTrue((ROOT / "packaging" / "pyinstaller" / "lity.spec").exists())
        self.assertTrue((ROOT / "scripts" / "build_macos.sh").exists())
        self.assertTrue((ROOT / "scripts" / "build_windows.ps1").exists())

    def test_packaging_docs_explain_slim_bundle_profile(self):
        packaging_readme = (ROOT / "packaging" / "README.md").read_text(encoding="utf-8")

        self.assertIn("slim", packaging_readme.lower())
        self.assertIn("base development installs", packaging_readme)
        self.assertIn("runtime services", packaging_readme)

    def test_windows_scripts_do_not_require_python3_only(self):
        setup_script = (ROOT / "scripts" / "setup.ps1").read_text(encoding="utf-8")
        build_script = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

        for script in (setup_script, build_script):
            self.assertIn("py -3", script)
            self.assertIn("Get-Command python", script)
            self.assertNotIn("python3 -m uv --version", script)

    def test_setup_scripts_install_web_extra(self):
        unix_script = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
        windows_script = (ROOT / "scripts" / "setup.ps1").read_text(encoding="utf-8")

        for script in (unix_script, windows_script):
            self.assertIn("--extra web", script)


if __name__ == "__main__":
    unittest.main()
