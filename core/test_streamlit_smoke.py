import compileall
from pathlib import Path
from unittest import TestCase


class StreamlitSmokeTests(TestCase):
	def setUp(self):
		self.project_root = Path(__file__).resolve().parent.parent
		self.streamlit_root = self.project_root / 'streamlit_app'
		self.page_files = [
			self.streamlit_root / 'pages' / 'ceo' / '00_Tong_quan_CEO.py',
			self.streamlit_root / 'pages' / 'ceo' / '01_Thu_hut_Kich_hoat.py',
			self.streamlit_root / 'pages' / 'ceo' / '02_Tin_dung.py',
			self.streamlit_root / 'pages' / 'ceo' / '03_He_sinh_thai.py',
			self.streamlit_root / 'pages' / 'ceo' / '04_Merchant.py',
			self.streamlit_root / 'pages' / 'tech' / '02_Hang_doi_Xet_duyet.py',
			self.streamlit_root / 'pages' / 'tech' / '03_Tinh_chinh_He_thong.py',
			self.streamlit_root / 'pages' / 'tech' / '04_Phan_tich_Ky_thuat.py',
		]

	def test_streamlit_sources_compile(self):
		compiled = compileall.compile_dir(str(self.streamlit_root), force=True, quiet=1)
		self.assertTrue(compiled, 'Streamlit modules must compile without syntax errors.')

	def test_unified_entrypoint_registers_all_pages(self):
		app_path = self.streamlit_root / 'app.py'
		content = app_path.read_text(encoding='utf-8')
		self.assertIn('st.Page("pages/ceo/', content)
		self.assertIn('st.Page("pages/tech/', content)