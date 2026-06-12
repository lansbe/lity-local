import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lity.services.files.documents import extract_document


class ExtractDocumentTests(unittest.TestCase):
    def test_plain_text_fallback(self):
        result = extract_document("notes.txt", b"Bonjour le monde")
        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "Bonjour le monde")

    def test_docx_round_trip(self):
        docx = __import__("docx")
        document = docx.Document()
        document.add_paragraph("Première ligne du document.")
        document.add_paragraph("Deuxième ligne.")
        buffer = io.BytesIO()
        document.save(buffer)

        result = extract_document("rapport.docx", buffer.getvalue())
        self.assertTrue(result["ok"], result.get("error"))
        self.assertIn("Première ligne du document.", result["text"])
        self.assertIn("Deuxième ligne.", result["text"])

    def test_pdf_without_text_is_flagged(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)

        result = extract_document("scan.pdf", buffer.getvalue())
        self.assertFalse(result["ok"])  # blank/scanned PDF → no extractable text
        self.assertIn("texte", (result["error"] or "").lower())

    def test_truncates_huge_text(self):
        result = extract_document("big.txt", ("a" * 60_000).encode())
        self.assertLessEqual(len(result["text"]), 40_100)
        self.assertIn("tronqué", result["text"])


if __name__ == "__main__":
    unittest.main()
